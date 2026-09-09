"""Public and hidden task contracts shared by data, training and serving.

``MathTask`` is the only agent-facing task type; ``ReferenceAnswer`` and
``LabeledMathTask`` may only exist in offline data, training and evaluation
code paths.
"""

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]


class AnswerType(StrEnum):
    INTEGER = "integer"
    RATIONAL = "rational"
    REAL = "real"
    EXPRESSION = "expression"
    SET = "set"
    TUPLE = "tuple"
    INTERVAL = "interval"


class MathTask(BaseModel):
    """Public, agent-facing task description. Never carries a reference answer."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    problem: str
    answer_type: AnswerType
    dataset: str
    split: str
    source_hash: str
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _reject_non_finite(cls, value: dict[str, JSONValue]) -> dict[str, JSONValue]:
        _ensure_finite(value)
        return value


class ReferenceAnswer(BaseModel):
    """Hidden reference answer; only offline training/evaluation may hold it."""

    model_config = ConfigDict(extra="forbid")

    value: str
    answer_type: AnswerType
    acceptable_forms: tuple[str, ...] = ()


class LabeledMathTask(BaseModel):
    """Task with its hidden reference. ``public_view()`` strips the answer."""

    model_config = ConfigDict(extra="forbid")

    task: MathTask
    reference: ReferenceAnswer

    def public_view(self) -> MathTask:
        return self.task


class Budget(BaseModel):
    """Per-episode resource limits for one agent trajectory."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(gt=0)
    max_tool_calls: int = Field(ge=0)
    max_python_seconds: float = Field(ge=0)
    max_observation_chars: int = Field(gt=0)


def _ensure_finite(value: JSONValue) -> None:
    """Raise ValueError when any float in a JSON value is NaN or infinite."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is not valid JSON metadata: {value!r}")
    elif isinstance(value, list):
        for item in value:
            _ensure_finite(item)
    elif isinstance(value, dict):
        for item in value.values():
            _ensure_finite(item)
