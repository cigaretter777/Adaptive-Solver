import json

from adaptive_math.core.hashing import make_source_hash, make_task_id
from adaptive_math.core.types import AnswerType
from adaptive_math.data.canonicalize import QuarantineReason, canonicalize_source
from adaptive_math.data.sources import SourceSpec


def make_spec(**overrides: object) -> SourceSpec:
    fields: dict[str, object] = {
        "name": "test_source",
        "uri": "file://synthetic",
        "revision": "a" * 40,
        "license": "apache-2.0",
        "intended_use": "train",
        "citation": "synthetic test source",
        "loader": "synthetic",
        "loader_params": {"problem_column": "problem", "answer_column": "answer"},
        "answer_type": AnswerType.INTEGER,
    }
    fields.update(overrides)
    return SourceSpec.model_validate(fields)


def test_records_map_to_labeled_tasks() -> None:
    spec = make_spec()
    records = [{"problem": "What is 1+1?", "answer": "2"}]
    kept, quarantined = canonicalize_source(spec, records)
    assert quarantined == []
    assert len(kept) == 1
    task = kept[0]
    assert task.task.task_id == make_task_id("test_source", "What is 1+1?")
    assert task.reference.value == "2"
    assert task.task.answer_type is AnswerType.INTEGER
    assert task.task.source_hash == make_source_hash(
        json.dumps(records[0], sort_keys=True).encode()
    )


def test_solution_column_boxed_extraction() -> None:
    spec = make_spec(
        loader_params={"problem_column": "problem", "solution_column": "solution"}
    )
    records = [{"problem": "Compute 6*7.", "solution": "We compute \\boxed{42}."}]
    kept, quarantined = canonicalize_source(spec, records)
    assert quarantined == []
    assert kept[0].reference.value == "42"


def test_missing_problem_is_quarantined() -> None:
    spec = make_spec()
    records = [{"answer": "2"}]
    kept, quarantined = canonicalize_source(spec, records)
    assert kept == []
    assert quarantined[0].reason is QuarantineReason.MISSING_PROBLEM


def test_missing_answer_is_quarantined() -> None:
    spec = make_spec()
    records = [{"problem": "What is 1+1?"}]
    kept, quarantined = canonicalize_source(spec, records)
    assert kept == []
    assert quarantined[0].reason is QuarantineReason.MISSING_ANSWER


def test_solution_with_multiple_boxed_keeps_the_last_one() -> None:
    # Solution-mode extraction convention: in a worked solution the final
    # answer is the LAST \boxed{...}; earlier ones are intermediate steps.
    # (This replaced an ambiguous-quarantine expectation when solution-mode
    # extraction was introduced; the strict trajectory extractor still treats
    # multiple boxed groups as AMBIGUOUS.)
    spec = make_spec(
        loader_params={"problem_column": "problem", "solution_column": "solution"}
    )
    records = [
        {"problem": "What is 1+1?", "solution": "It is \\boxed{1} or \\boxed{2}."}
    ]
    kept, quarantined = canonicalize_source(spec, records)
    assert quarantined == []
    assert len(kept) == 1
    assert kept[0].reference.value == "2"


def test_unverifiable_reference_is_quarantined() -> None:
    spec = make_spec()  # INTEGER answers
    records = [{"problem": "What is 1+1?", "answer": "not a number"}]
    kept, quarantined = canonicalize_source(spec, records)
    assert kept == []
    assert quarantined[0].reason is QuarantineReason.UNVERIFIABLE_REFERENCE


def test_quarantine_reasons_are_machine_readable() -> None:
    spec = make_spec()
    records = [
        {"answer": "2"},
        {"problem": "What is 1+1?"},
        {"problem": "What is 1+1?", "answer": "nope"},
        {"problem": "What is 1+1?", "answer": "2"},
    ]
    kept, quarantined = canonicalize_source(spec, records)
    assert len(kept) == 1
    assert {q.reason.value for q in quarantined} == {
        "missing_problem",
        "missing_answer",
        "unverifiable_reference",
    }
    for q in quarantined:
        assert q.source == "test_source"
        assert len(q.source_hash) == 64
        assert isinstance(q.detail, str)


def test_records_are_processed_in_deterministic_order() -> None:
    spec = make_spec()
    records = [{"problem": f"What is {i}+{i}?", "answer": str(2 * i)} for i in range(10)]
    first = canonicalize_source(spec, records)[0]
    second = canonicalize_source(spec, list(reversed(records)))[0]
    assert [t.task.task_id for t in first] == [t.task.task_id for t in second]
