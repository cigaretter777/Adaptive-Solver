"""
强化学习模块

使用 GRPO 算法进行在线强化学习优化，提供：
- 轨迹数据收集
- 多维度奖励函数
- GRPO 训练器（框架）
- 模型评估
"""

from .trajectory_collector import (
    TrajectoryCollector,
    Trajectory,
    Transition
)

from .reward_function import (
    RewardFunction,
    CompositeReward,
    AccuracyReward,
    EfficiencyReward,
    SafetyReward,
    ConsistencyReward,
    create_default_reward_function,
    get_reward_function
)

from .grpo_trainer import (
    GRPOTrainer,
    GRPOConfig
)

__all__ = [
    # 数据收集
    "TrajectoryCollector",
    "Trajectory",
    "Transition",
    # 奖励函数
    "RewardFunction",
    "CompositeReward",
    "AccuracyReward",
    "EfficiencyReward",
    "SafetyReward",
    "ConsistencyReward",
    "create_default_reward_function",
    "get_reward_function",
    # 训练
    "GRPOTrainer",
    "GRPOConfig",
]
