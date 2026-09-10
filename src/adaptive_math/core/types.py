"""Public and hidden task contracts shared by data, training and serving.

``MathTask`` is the only agent-facing task type; ``ReferenceAnswer`` and
``LabeledMathTask`` may only exist in offline data, training and evaluation
code paths.
"""

import math
from collections.abc import Iterator, Mapping
from enum import StrEnum
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]


class AnswerType(StrEnum):
    INTEGER = "integer"
    RATIONAL = "rational"
    REAL = "real"
    EXPRESSION = "expression"
    SET = "set"
    TUPLE = "tuple"
    INTERVAL = "interval"


class _FrozenJSONMapping(Mapping[str, object]):
    """Recursively immutable metadata storage compatible with deep copies."""

    __slots__ = ("_items",)
    _items: tuple[tuple[str, object], ...]

    def __init__(self, values: dict[str, object]) -> None:
        object.__setattr__(self, "_items", tuple(values.items()))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("_FrozenJSONMapping is immutable")

    def __getitem__(self, key: str) -> object:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, memo: dict[int, object]) -> "_FrozenJSONMapping":
        return self


class MathTask(BaseModel):
    """Public, agent-facing task description. Never carries a reference answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    problem: str
    answer_type: AnswerType
    dataset: str
    split: str
    source_hash: str
    pipeline_version: str
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _reject_non_finite(cls, value: dict[str, JSONValue]) -> dict[str, JSONValue]:
        _ensure_finite(value)
        return cast(dict[str, JSONValue], _freeze_json(value))

    @field_serializer("metadata")
    def _serialize_metadata(self, value: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return cast(dict[str, JSONValue], _thaw_json(value))


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


def _freeze_json(value: JSONValue) -> object:
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, dict):
        return _FrozenJSONMapping({key: _freeze_json(item) for key, item in value.items()})
    return value


def _thaw_json(value: object) -> JSONValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return cast(JSONValue, value)
