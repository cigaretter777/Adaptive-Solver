"""Map source-specific records to LabeledMathTask, quarantining bad rows.

Every kept task passes a reference self-check (the reference answer verifies
against itself); missing, ambiguous or unverifiable answers are quarantined
with a machine-readable reason. No model output is involved anywhere.
"""

import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from adaptive_math.core.hashing import make_source_hash, make_task_id
from adaptive_math.core.types import (
    JSONValue,
    LabeledMathTask,
    MathTask,
    ReferenceAnswer,
)
from adaptive_math.data.sources import SourceSpec
from adaptive_math.verifier import (
    ExtractResult,
    ExtractStatus,
    VerifierStatus,
    extract,
    normalize_surface,
    verify_answer,
)


class QuarantineReason(StrEnum):
    MISSING_PROBLEM = "missing_problem"
    MISSING_ANSWER = "missing_answer"
    AMBIGUOUS_ANSWER = "ambiguous_answer"
    UNVERIFIABLE_REFERENCE = "unverifiable_reference"
    PARSE_ERROR = "parse_error"


class QuarantineRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    source_hash: str
    reason: QuarantineReason
    detail: str


def canonicalize_source(
    spec: SourceSpec, records: list[dict[str, JSONValue]]
) -> tuple[list[LabeledMathTask], list[QuarantineRecord]]:
    """Canonicalize all records of one source, in deterministic order."""
    kept: list[LabeledMathTask] = []
    quarantined: list[QuarantineRecord] = []
    for record in sorted(
        records, key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False)
    ):
        outcome = _canonicalize_one(spec, record)
        if isinstance(outcome, LabeledMathTask):
            kept.append(outcome)
        else:
            quarantined.append(outcome)
    return kept, quarantined


def _canonicalize_one(
    spec: SourceSpec, record: dict[str, JSONValue]
) -> LabeledMathTask | QuarantineRecord:
    source_hash = make_source_hash(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
    )
    problem = record.get(str(spec.loader_params.get("problem_column", "problem")))
    if not isinstance(problem, str) or not problem.strip():
        return _quarantine(spec, source_hash, QuarantineReason.MISSING_PROBLEM, "empty problem")

    answer = _extract_answer(spec, record)
    if answer.status is ExtractStatus.MISSING:
        return _quarantine(
            spec, source_hash, QuarantineReason.MISSING_ANSWER, str(answer.details)
        )
    if answer.status is ExtractStatus.AMBIGUOUS:
        return _quarantine(
            spec, source_hash, QuarantineReason.AMBIGUOUS_ANSWER, str(answer.details)
        )
    assert answer.value is not None

    task = MathTask(
        task_id=make_task_id(spec.name, problem),
        problem=problem,
        answer_type=spec.answer_type,
        dataset=spec.name,
        split="unassigned",
        source_hash=source_hash,
        pipeline_version="canonicalize-v1",
        metadata=_metadata(spec, record),
    )
    reference = ReferenceAnswer(value=answer.value, answer_type=spec.answer_type)
    result = verify_answer(answer.value, reference, task_id=task.task_id)
    if result.status is not VerifierStatus.CORRECT:
        return _quarantine(
            spec,
            source_hash,
            QuarantineReason.UNVERIFIABLE_REFERENCE,
            f"self-check failed with {result.status.value}",
        )
    return LabeledMathTask(task=task, reference=reference)


def _extract_answer(
    spec: SourceSpec, record: dict[str, JSONValue]
) -> ExtractResult:
    answer_column = spec.loader_params.get("answer_column")
    if answer_column is not None:
        value = record.get(str(answer_column))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str) or not value.strip():
            return ExtractResult(
                status=ExtractStatus.MISSING,
                details={"reason": "empty source answer field"},
            )
        # A source answer column is a label, not an Agent response.  In
        # particular it need not contain <final>, $...$, or \\boxed{...}.
        # Its mathematical validity is established by the reference
        # self-check below, rather than by the trajectory-output extractor.
        return ExtractResult(status=ExtractStatus.OK, value=normalize_surface(value))
    solution_column = spec.loader_params.get("solution_column")
    if solution_column is not None:
        solution = record.get(str(solution_column))
        if not isinstance(solution, str) or not solution.strip():
            return ExtractResult(
                status=ExtractStatus.MISSING,
                details={"reason": "empty source solution field"},
            )
        result = extract(solution)
        if result.status is ExtractStatus.OK and result.value is not None:
            # The solution extractor has already identified a single answer.
            # Normalize its surface form, but do not feed the plain value back
            # through the trajectory extractor ("42" has no final marker).
            return result.model_copy(update={"value": normalize_surface(result.value)})
        return result
    return ExtractResult(
        status=ExtractStatus.MISSING,
        details={"reason": "source has neither answer_column nor solution_column"},
    )


def _metadata(spec: SourceSpec, record: dict[str, JSONValue]) -> dict[str, JSONValue]:
    mapping = spec.loader_params.get("metadata_columns")
    if not isinstance(mapping, dict):
        return {}
    metadata: dict[str, JSONValue] = {}
    for source_column, target_key in mapping.items():
        value = record.get(str(source_column))
        if isinstance(value, (str, int, float, bool)):
            metadata[str(target_key)] = value
    return metadata


def _quarantine(
    spec: SourceSpec, source_hash: str, reason: QuarantineReason, detail: str
) -> QuarantineRecord:
    return QuarantineRecord(
        source=spec.name, source_hash=source_hash, reason=reason, detail=detail
    )
