"""Deterministic seeded split assignment.

Tasks from eval-intended sources all go to frozen_eval; tasks from train
sources are seeded-shuffled and cut into train/sft_dev/rl_dev with exact
counts derived from the configured ratios. Rebuilds with the same seed are
identical, and frozen_eval never overlaps any training split.
"""

import random

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_math.core.types import LabeledMathTask
from adaptive_math.data.sources import SourceRegistry

TRAINING_SPLITS = ("train", "sft_dev", "rl_dev")
SPLIT_NAMES = (*TRAINING_SPLITS, "frozen_eval")


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_ratio: float = Field(default=0.90, gt=0, lt=1)
    sft_dev_ratio: float = Field(default=0.05, ge=0, lt=1)
    rl_dev_ratio: float = Field(default=0.05, ge=0, lt=1)

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> "SplitConfig":
        total = self.train_ratio + self.sft_dev_ratio + self.rl_dev_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        return self


def assign_splits(
    tasks: list[LabeledMathTask],
    registry: SourceRegistry,
    *,
    seed: int,
    config: SplitConfig | None = None,
) -> dict[str, list[LabeledMathTask]]:
    config = config or SplitConfig()
    eval_names = registry.eval_source_names()
    ordered = sorted(tasks, key=lambda t: t.task.task_id)
    eval_tasks = [t for t in ordered if t.task.dataset in eval_names]
    train_tasks = [t for t in ordered if t.task.dataset not in eval_names]

    train_count = round(len(train_tasks) * config.train_ratio)
    sft_dev_count = round(len(train_tasks) * config.sft_dev_ratio)
    shuffled = random.Random(seed).sample(train_tasks, k=len(train_tasks))
    splits: dict[str, list[LabeledMathTask]] = {
        "train": sorted(shuffled[:train_count], key=lambda t: t.task.task_id),
        "sft_dev": sorted(
            shuffled[train_count : train_count + sft_dev_count], key=lambda t: t.task.task_id
        ),
        "rl_dev": sorted(
            shuffled[train_count + sft_dev_count :], key=lambda t: t.task.task_id
        ),
        "frozen_eval": sorted(eval_tasks, key=lambda t: t.task.task_id),
    }
    return {name: [_relabel(t, name) for t in members] for name, members in splits.items()}


def _relabel(task: LabeledMathTask, split: str) -> LabeledMathTask:
    if task.task.split == split:
        return task
    return task.model_copy(update={"task": task.task.model_copy(update={"split": split})})
