"""Turn verified source solutions into protocol-valid DIRECT agent traces."""

import json
import re
from collections import Counter
from dataclasses import dataclass

import orjson

from adaptive_math.agent.actions import FinalAction, ToolAction
from adaptive_math.agent.environment import OfflineMathEnv
from adaptive_math.agent.parser import parse_action
from adaptive_math.agent.state import EventKind, TerminationReason, TraceEvent, Usage
from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.hashing import make_source_hash
from adaptive_math.core.types import Budget, JSONValue, LabeledMathTask
from adaptive_math.data.canonicalize import canonicalize_source
from adaptive_math.data.sources import SourceSpec
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.training.sft_builder import build_sft_record
from adaptive_math.training.sft_records import SFTTrajectoryRecord
from adaptive_math.verifier import (
    ExtractStatus,
    VerifierStatus,
    extract_solution_answer,
    verify_answer,
)


def direct_trace_from_solution(labeled_task: LabeledMathTask, solution: str) -> Trajectory:
    """Build a DIRECT trace only when a source solution proves the reference answer.

    This preserves source reasoning verbatim.  Rows without an extractable,
    verifier-correct answer are rejected rather than converted into answer-only
    demonstrations.
    """
    reasoning = solution.strip()
    if not reasoning:
        raise ValueError("source solution is empty")
    if any(tag in reasoning for tag in ("<think>", "</think>", "<tool_call>", "</tool_call>", "<final>", "</final>")):
        raise ValueError("source solution contains a reserved protocol tag")
    extracted = extract_solution_answer(reasoning)
    if extracted.status is not ExtractStatus.OK or extracted.value is None:
        raise ValueError("source solution has no unambiguous terminal answer")
    verdict = verify_answer(
        extracted.value, labeled_task.reference, task_id=labeled_task.task.task_id
    )
    if verdict.status is not VerifierStatus.CORRECT:
        raise ValueError("source solution is not verifier-correct")
    raw = (
        f"<think>{reasoning}</think><final>"
        + orjson.dumps({"answer": extracted.value}).decode()
        + "</final>"
    )
    parsed = parse_action(raw)
    if not isinstance(parsed.action, FinalAction) or parsed.action.answer != extracted.value:
        raise ValueError("constructed DIRECT trace violates the production action protocol")
    return Trajectory(
        trace_id=f"source:{labeled_task.task.source_hash[:20]}",
        task_id=labeled_task.task.task_id,
        events=(
            TraceEvent(
                sequence=0,
                kind=EventKind.MODEL_OUTPUT,
                monotonic_ms=0,
                payload={"raw": raw, "generated_tokens": 0},
            ),
            TraceEvent(
                sequence=1,
                kind=EventKind.FINAL,
                monotonic_ms=1,
                payload={"answer": extracted.value},
            ),
        ),
        final_answer=extracted.value,
        termination_reason=TerminationReason.FINAL,
        usage=Usage(steps=1),
        runtime_version="runtime-v1+source-solution-v1",
    )


@dataclass(frozen=True)
class DirectMaterialization:
    records: tuple[SFTTrajectoryRecord, ...]
    traces: tuple[Trajectory, ...]
    rejected: dict[str, int]


def materialize_direct_records(
    spec: SourceSpec,
    raw_records: list[dict[str, JSONValue]],
    budget: Budget,
    registry: ToolRegistry,
    *,
    tokenizer_revision: str = "unresolved",
) -> DirectMaterialization:
    """Convert one source's retained solution rows into audited DIRECT records."""
    solution_column = spec.loader_params.get("solution_column")
    if not isinstance(solution_column, str):
        raise TypeError(f"source {spec.name} has no solution_column for DIRECT materialization")
    labeled, _ = canonicalize_source(spec, raw_records)
    labeled_by_source_hash = {item.task.source_hash: item for item in labeled}
    records: list[SFTTrajectoryRecord] = []
    traces: list[Trajectory] = []
    rejected: Counter[str] = Counter()
    seen_task_ids: set[str] = set()
    for raw_record in sorted(
        raw_records, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False)
    ):
        source_hash = make_source_hash(
            json.dumps(raw_record, sort_keys=True, ensure_ascii=False).encode()
        )
        labeled_task = labeled_by_source_hash.get(source_hash)
        if labeled_task is None:
            rejected["canonicalize"] += 1
            continue
        if labeled_task.task.task_id in seen_task_ids:
            rejected["duplicate_task"] += 1
            continue
        solution = raw_record.get(solution_column)
        if not isinstance(solution, str):
            rejected["missing_solution"] += 1
            continue
        try:
            trace = direct_trace_from_solution(labeled_task, solution)
            record = build_sft_record(
                labeled_task,
                trace,
                budget,
                registry,
                tokenizer_revision=tokenizer_revision,
            )
        except ValueError:
            rejected["invalid_solution"] += 1
            continue
        seen_task_ids.add(labeled_task.task.task_id)
        records.append(record)
        traces.append(trace)
    return DirectMaterialization(
        records=tuple(records), traces=tuple(traces), rejected=dict(sorted(rejected.items()))
    )


def materialize_direct_records_batched(
    spec: SourceSpec,
    raw_records: list[dict[str, JSONValue]],
    budget: Budget,
    registry: ToolRegistry,
    *,
    batch_size: int,
    tokenizer_revision: str = "unresolved",
) -> DirectMaterialization:
    """Bound peak memory while retaining deterministic, cross-batch deduplication."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rejected: Counter[str] = Counter()
    selected: dict[str, tuple[SFTTrajectoryRecord, Trajectory]] = {}
    for start in range(0, len(raw_records), batch_size):
        batch = materialize_direct_records(
            spec,
            raw_records[start : start + batch_size],
            budget,
            registry,
            tokenizer_revision=tokenizer_revision,
        )
        rejected.update(batch.rejected)
        for record, trace in zip(batch.records, batch.traces, strict=True):
            if record.task_id in selected:
                rejected["duplicate_task"] += 1
                continue
            selected[record.task_id] = (record, trace)
    ordered = [selected[task_id] for task_id in sorted(selected)]
    return DirectMaterialization(
        records=tuple(record for record, _ in ordered),
        traces=tuple(trace for _, trace in ordered),
        rejected=dict(sorted(rejected.items())),
    )


_PYTHON_FENCE = re.compile(r"```python\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL)


async def python_tir_trace_from_solution(
    labeled_task: LabeledMathTask,
    solution: str,
    budget: Budget,
    registry: ToolRegistry,
) -> Trajectory:
    """Replay one fenced Python calculation through the production environment."""
    matches = _PYTHON_FENCE.findall(solution)
    if len(matches) != 1 or not matches[0].strip():
        raise ValueError("TIR solution requires exactly one non-empty fenced Python block")
    extracted = extract_solution_answer(solution)
    if extracted.status is not ExtractStatus.OK or extracted.value is None:
        raise ValueError("TIR solution has no unambiguous terminal answer")
    verdict = verify_answer(
        extracted.value, labeled_task.reference, task_id=labeled_task.task.task_id
    )
    if verdict.status is not VerifierStatus.CORRECT:
        raise ValueError("TIR solution is not verifier-correct")
    environment = OfflineMathEnv(
        labeled_task, budget, registry, trace_id=f"tir:{labeled_task.task.source_hash[:20]}"
    )
    tool_raw = "<think>Execute the supplied calculation.</think><tool_call>" + orjson.dumps(
        {"name": "python", "arguments": {"code": matches[0].strip()}}
    ).decode() + "</tool_call>"
    parsed_tool = parse_action(tool_raw)
    if not isinstance(parsed_tool.action, ToolAction):
        raise TypeError("constructed TIR tool action violates the production action protocol")
    environment.record_model_output(tool_raw, generated_tokens=0, monotonic_ms=0)
    tool_step = await environment.step(parsed_tool.action, monotonic_ms=1)
    if tool_step.terminated:
        raise ValueError("TIR tool call exhausted its budget before the final answer")
    tool_events = tool_step.state.events
    result_payload = tool_events[-1].payload.get("result") if tool_events else None
    if not isinstance(result_payload, dict) or result_payload.get("ok") is not True:
        raise ValueError("TIR tool call did not execute successfully")
    final_raw = "<final>" + orjson.dumps({"answer": extracted.value}).decode() + "</final>"
    parsed_final = parse_action(final_raw)
    if not isinstance(parsed_final.action, FinalAction):
        raise TypeError("constructed TIR final action violates the production action protocol")
    environment.record_model_output(final_raw, generated_tokens=0, monotonic_ms=2)
    final_step = await environment.step(parsed_final.action, monotonic_ms=3)
    evaluation = environment.evaluate()
    if not final_step.terminated or evaluation is None or evaluation.status is not VerifierStatus.CORRECT:
        raise ValueError("TIR final answer is not verifier-correct after replay")
    state = final_step.state
    if state.termination_reason is None:
        raise ValueError("TIR replay terminated without a termination reason")
    return Trajectory(
        trace_id=environment.trace_id,
        task_id=labeled_task.task.task_id,
        events=state.events,
        final_answer=state.final_answer,
        termination_reason=state.termination_reason,
        usage=state.usage,
        runtime_version="runtime-v1+tir-replay-v1",
    )


async def materialize_python_tir_records(
    spec: SourceSpec,
    raw_records: list[dict[str, JSONValue]],
    budget: Budget,
    registry: ToolRegistry,
    *,
    tokenizer_revision: str = "unresolved",
) -> DirectMaterialization:
    """Replay source TIR Python snippets and retain only correct executions."""
    solution_column = spec.loader_params.get("solution_column")
    if not isinstance(solution_column, str):
        raise TypeError(f"source {spec.name} has no solution_column for TIR materialization")
    labeled, _ = canonicalize_source(spec, raw_records)
    labeled_by_source_hash = {item.task.source_hash: item for item in labeled}
    records: list[SFTTrajectoryRecord] = []
    traces: list[Trajectory] = []
    rejected: Counter[str] = Counter()
    seen_task_ids: set[str] = set()
    for raw_record in sorted(
        raw_records, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False)
    ):
        source_hash = make_source_hash(
            json.dumps(raw_record, sort_keys=True, ensure_ascii=False).encode()
        )
        labeled_task = labeled_by_source_hash.get(source_hash)
        if labeled_task is None:
            rejected["canonicalize"] += 1
            continue
        if labeled_task.task.task_id in seen_task_ids:
            rejected["duplicate_task"] += 1
            continue
        solution = raw_record.get(solution_column)
        if not isinstance(solution, str):
            rejected["missing_solution"] += 1
            continue
        try:
            trace = await python_tir_trace_from_solution(labeled_task, solution, budget, registry)
            record = build_sft_record(
                labeled_task,
                trace,
                budget,
                registry,
                tokenizer_revision=tokenizer_revision,
            )
        except ValueError:
            rejected["invalid_or_unreplayable_tir"] += 1
            continue
        seen_task_ids.add(labeled_task.task.task_id)
        records.append(record)
        traces.append(trace)
    return DirectMaterialization(
        records=tuple(records), traces=tuple(traces), rejected=dict(sorted(rejected.items()))
    )
