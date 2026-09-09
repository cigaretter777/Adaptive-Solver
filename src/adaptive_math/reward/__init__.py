"""Pure, versioned reward functions (R0-R3 ablations)."""

from adaptive_math.reward.functions import (
    R0_DEFAULT,
    R1_DEFAULT,
    R2_DEFAULT,
    R3_DEFAULT,
    compute_reward,
    reward_r0,
    reward_r1,
    reward_r2,
    reward_r3,
)
from adaptive_math.reward.types import RewardBreakdown, RewardConfig, RewardContext

__all__ = [
    "R0_DEFAULT",
    "R1_DEFAULT",
    "R2_DEFAULT",
    "R3_DEFAULT",
    "RewardBreakdown",
    "RewardConfig",
    "RewardContext",
    "compute_reward",
    "reward_r0",
    "reward_r1",
    "reward_r2",
    "reward_r3",
]
