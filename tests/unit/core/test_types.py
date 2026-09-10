import copy
import json
from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from adaptive_math.core.types import (
    AnswerType,
    Budget,
    LabeledMathTask,
    MathTask,
    ReferenceAnswer,
)


def make_task(**overrides: object) -> MathTask:
    fields: dict[str, object] = {
        "task_id": "aime_2024:0123456789abcdef0123",
        "problem": "What is 1+1?",
        "answer_type": AnswerType.INTEGER,
        "dataset": "aime_2024",
        "split": "train",
        "source_hash": "a" * 64,
        "pipeline_version": "canonicalize-v1",
    }
    fields.update(overrides)
    return MathTask(**fields)


@pytest.fixture
def labeled_task() -> LabeledMathTask:
    return LabeledMathTask(
        task=make_task(),
        reference=ReferenceAnswer(value="2", answer_type=AnswerType.INTEGER),
    )


def test_public_view_removes_reference_answer(labeled_task: LabeledMathTask) -> None:
    public = labeled_task.public_view()
    assert isinstance(public, MathTask)
    assert "reference" not in public.model_dump()
    assert "answer" not in public.model_dump()


def test_public_view_preserves_public_fields(labeled_task: LabeledMathTask) -> None:
    assert labeled_task.public_view() == labeled_task.task


def test_task_requires_all_public_fields() -> None:
    with pytest.raises(ValidationError):
        MathTask(
            problem="What is 1+1?",
            answer_type=AnswerType.INTEGER,
            dataset="aime_2024",
            split="train",
            source_hash="a" * 64,
        )


def test_metadata_defaults_to_empty() -> None:
    assert make_task().metadata == {}


def test_task_lineage_and_nested_metadata_are_immutable() -> None:
    task = make_task(metadata={"provenance": {"tags": ["aime"]}})

    for field in ("task_id", "dataset", "split", "source_hash", "pipeline_version"):
        with pytest.raises(ValidationError):
            setattr(task, field, "changed")
    provenance = task.metadata["provenance"]
    assert isinstance(provenance, Mapping)
    with pytest.raises(TypeError):
        provenance["source"] = "mutated"
    tags = provenance["tags"]
    assert isinstance(tags, tuple)
    with pytest.raises(AttributeError):
        tags.append("mutated")


def test_immutable_metadata_survives_deep_copy_and_json_serialization() -> None:
    task = make_task(metadata={"nested": {"values": [1, 2]}})

    copied = copy.deepcopy(task)
    model_copied = task.model_copy(deep=True)
    assert copied == task
    assert model_copied == task
    assert json.loads(task.model_dump_json())["metadata"] == {"nested": {"values": [1, 2]}}


def test_pipeline_version_is_required() -> None:
    values = make_task().model_dump()
    values.pop("pipeline_version")
    with pytest.raises(ValidationError):
        MathTask.model_validate(values)


def test_zero_or_negative_budgets_are_invalid() -> None:
    base = {"max_steps": 1, "max_tool_calls": 0, "max_python_seconds": 0.0, "max_observation_chars": 1}
    for invalid in (
        {"max_steps": 0},
        {"max_steps": -1},
        {"max_observation_chars": 0},
        {"max_observation_chars": -5},
        {"max_tool_calls": -1},
        {"max_python_seconds": -0.1},
    ):
        with pytest.raises(ValidationError):
            Budget(**{**base, **invalid})


def test_zero_tool_calls_and_python_seconds_are_allowed() -> None:
    budget = Budget(max_steps=1, max_tool_calls=0, max_python_seconds=0.0, max_observation_chars=1)
    assert budget.max_tool_calls == 0
    assert budget.max_python_seconds == 0.0


def test_metadata_rejects_non_finite_floats() -> None:
    with pytest.raises(ValidationError):
        make_task(metadata={"x": float("nan")})
    with pytest.raises(ValidationError):
        make_task(metadata={"nested": [1.0, {"y": float("inf")}]})
    with pytest.raises(ValidationError):
        make_task(metadata={"z": float("-inf")})


def test_json_round_trip_is_stable(labeled_task: LabeledMathTask) -> None:
    dumped = labeled_task.model_dump_json()
    restored = LabeledMathTask.model_validate_json(dumped)
    assert restored == labeled_task
    assert json.loads(dumped)["reference"]["value"] == "2"


def test_answer_type_members() -> None:
    assert {member.value for member in AnswerType} == {
        "integer",
        "rational",
        "real",
        "expression",
        "set",
        "tuple",
        "interval",
    }


def test_reference_answer_defaults() -> None:
    reference = ReferenceAnswer(value="2", answer_type=AnswerType.INTEGER)
    assert reference.acceptable_forms == ()
