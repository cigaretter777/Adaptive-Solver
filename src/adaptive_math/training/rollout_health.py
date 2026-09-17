"""Pure helpers for GRPO rollout-health experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from random import Random
from typing import Any

from adaptive_math.core.types import LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.training.reward_bridge import group_advantages


def select_task_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    return Random(seed).sample(list(rows), count)


def _normalize_acceptable_forms(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()

    if isinstance(value, str):
        return (value,)

    return tuple(value)


def row_to_labeled_task(row: Mapping[str, Any]) -> LabeledMathTask:
    metadata = row.get("metadata")

    if isinstance(metadata, str):
        metadata = json.loads(metadata) if metadata else {}

    if metadata is None:
        metadata = {}

    task = MathTask(
        task_id=row["task_id"],
        problem=row["problem"],
        answer_type=row["answer_type"],
        dataset=row["dataset"],
        split=row["split"],
        source_hash=row["source_hash"],
        pipeline_version=row["pipeline_version"],
        metadata=metadata,
    )

    reference = ReferenceAnswer(
        value=row["reference_value"],
        answer_type=row["answer_type"],
        acceptable_forms=_normalize_acceptable_forms(row.get("reference_acceptable_forms")),
    )

    return LabeledMathTask(
        task=task,
        reference=reference,
    )


def summarize_group_rewards(
    groups: Sequence[Sequence[float]],
) -> dict[str, float | int]:
    if not groups:
        raise ValueError("groups must not be empty")

    all_zero = 0
    all_one = 0
    mixed = 0
    effective = 0

    for rewards in groups:
        values = list(rewards)

        if not values:
            raise ValueError("reward group must not be empty")

        if all(r == 0.0 for r in values):
            all_zero += 1
        elif all(r == 1.0 for r in values):
            all_one += 1
        else:
            mixed += 1

        if group_advantages(values).effective:
            effective += 1

    n = len(groups)

    return {
        "group_count": n,
        "all_zero_group_count": all_zero,
        "mixed_group_count": mixed,
        "all_one_group_count": all_one,
        "effective_group_count": effective,
        "all_zero_group_rate": all_zero / n,
        "mixed_group_rate": mixed / n,
        "all_one_group_rate": all_one / n,
        "effective_group_rate": effective / n,
    }
