"""Core contracts: task types, budgets and content-addressed hashing."""

from adaptive_math.core.hashing import canonical_text, make_source_hash, make_task_id, sha256_hex
from adaptive_math.core.types import (
    AnswerType,
    Budget,
    JSONValue,
    LabeledMathTask,
    MathTask,
    ReferenceAnswer,
)

__all__ = [
    "AnswerType",
    "Budget",
    "JSONValue",
    "LabeledMathTask",
    "MathTask",
    "ReferenceAnswer",
    "canonical_text",
    "make_source_hash",
    "make_task_id",
    "sha256_hex",
]
