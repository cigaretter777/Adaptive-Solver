"""Shared reward normalization primitives for synchronous GRPO rollout groups."""

import math

from pydantic import BaseModel, ConfigDict


class GroupAdvantages(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    advantages: tuple[float, ...]
    effective: bool
    mean_reward: float
    std_reward: float


def group_advantages(rewards: list[float], *, epsilon: float = 1e-6) -> GroupAdvantages:
    """Compute GRPO group-relative advantages without inventing signal for ties."""
    if not rewards:
        raise ValueError("rewards must not be empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not all(math.isfinite(reward) for reward in rewards):
        raise ValueError("rewards must be finite")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std == 0:
        return GroupAdvantages(
            advantages=tuple(0.0 for _ in rewards), effective=False, mean_reward=mean, std_reward=std
        )
    return GroupAdvantages(
        advantages=tuple((reward - mean) / (std + epsilon) for reward in rewards),
        effective=True,
        mean_reward=mean,
        std_reward=std,
    )
