import asyncio

import pytest
from pydantic import BaseModel

from adaptive_math.agent.parser import parse_action
from adaptive_math.agent.state import EventKind, TerminationReason
from adaptive_math.core.types import AnswerType, Budget, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.data.sources import SourceSpec
from adaptive_math.tools.base import ToolContext, ToolResult
from adaptive_math.tools.python_tool import PythonArguments
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.training.solution_traces import (
    direct_trace_from_solution,
    materialize_direct_records,
    materialize_direct_records_batched,
    materialize_python_tir_records,
    python_tir_trace_from_solution,
)


def _task() -> LabeledMathTask:
    return LabeledMathTask(
        task=MathTask(
            task_id="unit:solution",
            problem="What is 1+1?",
            answer_type=AnswerType.INTEGER,
            dataset="unit",
            split="train",
            source_hash="a" * 64,
            pipeline_version="v1",
        ),
        reference=ReferenceAnswer(value="2", answer_type=AnswerType.INTEGER),
    )


def test_direct_trace_uses_the_verified_source_solution_as_reasoning() -> None:
    trace = direct_trace_from_solution(_task(), "Adding gives \\boxed{2}.")

    raw = trace.events[0].payload["raw"]
    assert isinstance(raw, str)
    assert parse_action(raw).action is not None
    assert trace.events[-1].kind is EventKind.FINAL
    assert trace.final_answer == "2"
    assert trace.termination_reason is TerminationReason.FINAL


def test_direct_trace_rejects_solution_whose_extracted_answer_is_incorrect() -> None:
    with pytest.raises(ValueError, match="verifier-correct"):
        direct_trace_from_solution(_task(), "Adding gives \\boxed{3}.")


def test_direct_trace_accepts_a_verifier_correct_terminal_assignment() -> None:
    trace = direct_trace_from_solution(
        _task(),
        "First calculate 1+1.\nx = 2\nTOTAL 2 POINTS",
    )

    assert trace.final_answer == "2"
    assert '"answer":"2"' in str(trace.events[-2].payload["raw"])


def test_direct_trace_does_not_fallback_after_a_semantically_incorrect_strict_answer() -> None:
    with pytest.raises(ValueError, match="verifier-correct"):
        direct_trace_from_solution(_task(), "The answer is \\boxed{3}.\nx = 2")


def test_direct_trace_rejects_protocol_tag_in_source_reasoning() -> None:
    with pytest.raises(ValueError, match="protocol tag"):
        direct_trace_from_solution(_task(), "<final>{\"answer\": \"2\"}</final>")


def test_direct_materialization_canonicalizes_and_builds_a_verified_sft_record() -> None:
    spec = SourceSpec(
        name="unit_source",
        uri="file://unit",
        revision="a" * 40,
        license="apache-2.0",
        intended_use="train",
        citation="unit",
        loader="synthetic",
        loader_params={"problem_column": "problem", "solution_column": "solution"},
        answer_type=AnswerType.INTEGER,
    )

    result = materialize_direct_records(
        spec,
        [{"problem": "1+1", "solution": "Adding gives \\boxed{2}."}],
        Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100),
        ToolRegistry([]),
    )

    assert len(result.records) == 1
    assert len(result.traces) == 1
    assert result.records[0].behavior.value == "direct"
    assert result.rejected == {}


def test_direct_materialization_records_per_row_rejection_diagnostics() -> None:
    spec = SourceSpec(
        name="unit_diagnostics",
        uri="file://unit",
        revision="a" * 40,
        license="apache-2.0",
        intended_use="train",
        citation="unit",
        loader="synthetic",
        loader_params={
            "problem_column": "problem",
            "answer_column": "answer",
            "solution_column": "solution",
        },
        answer_type=AnswerType.INTEGER,
    )

    result = materialize_direct_records(
        spec,
        [
            {"problem": "1+1", "answer": "2", "solution": "Answer: 2"},
            {"problem": "2+2", "answer": "4", "solution": "The answer is 3"},
            {"problem": "3+3", "answer": "6", "solution": "The calculation is complete."},
            {"problem": "", "answer": "7", "solution": "Answer: 7"},
        ],
        Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100),
        ToolRegistry([]),
    )

    assert len(result.records) == 1
    assert {item.outcome for item in result.diagnostics} == {
        "accepted_strict",
        "canonicalize_missing_problem",
        "strict_verifier_incorrect",
        "terminal_extract_missing",
    }


def test_batched_direct_materialization_keeps_deterministic_unique_records() -> None:
    spec = SourceSpec(
        name="unit_batched",
        uri="file://unit",
        revision="a" * 40,
        license="apache-2.0",
        intended_use="train",
        citation="unit",
        loader="synthetic",
        loader_params={"problem_column": "problem", "solution_column": "solution"},
        answer_type=AnswerType.INTEGER,
    )
    rows = [
        {"problem": "1+1", "solution": "\\boxed{2}"},
        {"problem": "2+2", "solution": "\\boxed{4}"},
        {"problem": "1+1", "solution": "\\boxed{2}"},
    ]

    result = materialize_direct_records_batched(
        spec,
        rows,
        Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100),
        ToolRegistry([]),
        batch_size=1,
    )

    assert {record.final_answer for record in result.records} == {"2", "4"}
    assert result.rejected == {"duplicate_task": 1}


class _PythonTool:
    name = "python"
    description = "test python"
    arguments_model: type[BaseModel] = PythonArguments

    async def execute(self, arguments: BaseModel, context: ToolContext) -> ToolResult:
        assert isinstance(arguments, PythonArguments)
        assert arguments.code == "print(1 + 1)"
        return ToolResult(ok=True, output="2\n", latency_ms=1)


def test_tir_trace_executes_fenced_python_through_the_tool_registry() -> None:
    trace = asyncio.run(
        python_tir_trace_from_solution(
            _task(),
            "```python\nprint(1 + 1)\n```\nTherefore \\boxed{2}.",
            Budget(max_steps=3, max_tool_calls=1, max_python_seconds=5, max_observation_chars=100),
            ToolRegistry([_PythonTool()]),
        )
    )

    assert [event.kind for event in trace.events] == [
        EventKind.MODEL_OUTPUT,
        EventKind.TOOL_CALL,
        EventKind.TOOL_RESULT,
        EventKind.MODEL_OUTPUT,
        EventKind.FINAL,
    ]
    assert trace.final_answer == "2"
    assert trace.termination_reason is TerminationReason.FINAL


def test_tir_materialization_emits_a_python_behavior_sft_record() -> None:
    spec = SourceSpec(
        name="unit_tir",
        uri="file://unit",
        revision="a" * 40,
        license="apache-2.0",
        intended_use="train",
        citation="unit",
        loader="synthetic",
        loader_params={"problem_column": "problem", "solution_column": "solution"},
        answer_type=AnswerType.INTEGER,
    )

    result = asyncio.run(
        materialize_python_tir_records(
            spec,
            [{"problem": "1+1", "solution": "```python\nprint(1 + 1)\n```\n\\boxed{2}"}],
            Budget(max_steps=3, max_tool_calls=1, max_python_seconds=5, max_observation_chars=100),
            ToolRegistry([_PythonTool()]),
        )
    )

    assert len(result.records) == 1
    assert len(result.traces) == 1
    assert result.records[0].behavior.value == "python"
