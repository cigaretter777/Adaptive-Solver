"""
奖励函数模块

定义多维度奖励函数，用于评估状态机执行质量
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import re


class RewardFunction(ABC):
    """奖励函数基类"""

    @abstractmethod
    def compute(self, trajectory: Dict[str, Any]) -> float:
        """
        计算奖励

        Args:
            trajectory: 轨迹数据

        Returns:
            float: 奖励值
        """
        pass

    def __call__(self, trajectory: Dict[str, Any]) -> float:
        """便捷调用"""
        return self.compute(trajectory)


class AccuracyReward(RewardFunction):
    """
    准确性奖励

    评估答案的正确性和准确性
    """

    def __init__(self, weight: float = 1.0):
        """
        初始化

        Args:
            weight: 权重
        """
        self.weight = weight

    def compute(self, trajectory: Dict[str, Any]) -> float:
        """
        计算准确性奖励

        Args:
            trajectory: 轨迹数据

        Returns:
            float: 奖励值 (0-1)
        """
        # 基本成功奖励
        if trajectory.get("success", False):
            base_reward = 1.0
        else:
            base_reward = -0.5

        # 检查答案质量
        answer = trajectory.get("final_answer", "")
        query = trajectory.get("query", "")

        # 计算任务：检查是否包含数字结果
        if "计算" in query and re.search(r'\d+', answer):
            base_reward += 0.2

        # 证明任务：检查是否包含证明逻辑
        if "证明" in query and any(kw in answer for kw in ["证明", "因为", "所以", "因此"]):
            base_reward += 0.2

        # 答案长度适中奖励
        answer_len = len(answer)
        if 50 <= answer_len <= 500:
            base_reward += 0.1

        return base_reward * self.weight


class EfficiencyReward(RewardFunction):
    """
    效率奖励

    评估执行效率，奖励更少的步骤和工具调用
    """

    def __init__(self, weight: float = 0.5):
        """
        初始化

        Args:
            weight: 权重
        """
        self.weight = weight
        self.optimal_steps = 5  # 最优步骤数

    def compute(self, trajectory: Dict[str, Any]) -> float:
        """
        计算效率奖励

        Args:
            trajectory: 轨迹数据

        Returns:
            float: 奖励值 (0-1)
        """
        steps = trajectory.get("steps", 0)
        metadata = trajectory.get("metadata", {})
        tool_calls = metadata.get("tool_calls", [])

        # 步骤数奖励（越接近最优越好）
        if steps == 0:
            steps_reward = 0.0
        elif steps <= self.optimal_steps:
            steps_reward = 1.0
        else:
            steps_reward = max(0.0, 1.0 - (steps - self.optimal_steps) / 10.0)

        # 工具调用次数奖励
        optimal_tool_calls = 2  # 假设最优工具调用次数
        tool_count = len(tool_calls)
        if tool_count <= optimal_tool_calls:
            tool_reward = 1.0
        else:
            tool_reward = max(0.0, 1.0 - (tool_count - optimal_tool_calls) / 5.0)

        # 综合奖励
        total_reward = (steps_reward + tool_reward) / 2.0

        return total_reward * self.weight


class SafetyReward(RewardFunction):
    """
    安全性奖励

    评估执行过程的安全性和稳定性
    """

    def __init__(self, weight: float = 0.3):
        """
        初始化

        Args:
            weight: 权重
        """
        self.weight = weight

    def compute(self, trajectory: Dict[str, Any]) -> float:
        """
        计算安全性奖励

        Args:
            trajectory: 轨迹数据

        Returns:
            float: 奖励值 (0-1)
        """
        reward = 1.0

        # 检查是否有错误
        transitions = trajectory.get("transitions", [])
        error_count = 0
        for trans in transitions:
            if trans.get("info", {}).get("error"):
                error_count += 1
                reward -= 0.2

        # 检查是否有回退（重复访问同一状态）
        states = [trans["state"].get("status") for trans in transitions]
        if len(states) != len(set(states)):
            reward -= 0.1  # 有回退

        # 确保奖励不为负
        reward = max(0.0, reward)

        return reward * self.weight


class ConsistencyReward(RewardFunction):
    """
    一致性奖励

    评估结果与预期的一致性
    """

    def __init__(self, weight: float = 0.2):
        """
        初始化

        Args:
            weight: 权重
        """
        self.weight = weight

    def compute(self, trajectory: Dict[str, Any]) -> float:
        """
        计算一致性奖励

        Args:
            trajectory: 轨迹数据

        Returns:
            float: 奖励值 (0-1)
        """
        query = trajectory.get("query", "")
        answer = trajectory.get("final_answer", "")

        reward = 0.0

        # 检查答案是否包含查询中的关键信息
        query_keywords = re.findall(r'[\w]+', query)
        answer_text = answer.lower()
        matched_keywords = sum(1 for kw in query_keywords if kw.lower() in answer_text)

        if query_keywords:
            keyword_ratio = matched_keywords / len(query_keywords)
            reward = keyword_ratio

        return reward * self.weight


class CompositeReward(RewardFunction):
    """
    组合奖励函数

    组合多个奖励函数
    """

    def __init__(self, reward_functions: List[RewardFunction]):
        """
        初始化

        Args:
            reward_functions: 奖励函数列表
        """
        self.reward_functions = reward_functions

    def compute(self, trajectory: Dict[str, Any]) -> float:
        """
        计算组合奖励

        Args:
            trajectory: 轨迹数据

        Returns:
            float: 总奖励值
        """
        total_reward = 0.0
        details = {}

        for i, func in enumerate(self.reward_functions):
            reward = func.compute(trajectory)
            total_reward += reward
            details[f"reward_{i}_{func.__class__.__name__}"] = reward

        return total_reward

    def get_details(self, trajectory: Dict[str, Any]) -> Dict[str, float]:
        """
        获取各奖励函数的详细结果

        Args:
            trajectory: 轨迹数据

        Returns:
            Dict: 各奖励值
        """
        details = {}
        for func in self.reward_functions:
            name = func.__class__.__name__
            details[name] = func.compute(trajectory)
        return details


def create_default_reward_function() -> CompositeReward:
    """
    创建默认的组合奖励函数

    Returns:
        CompositeReward: 默认奖励函数
    """
    return CompositeReward([
        AccuracyReward(weight=1.0),
        EfficiencyReward(weight=0.5),
        SafetyReward(weight=0.3),
        ConsistencyReward(weight=0.2)
    ])


# 预定义的奖励函数集合
REWARD_FUNCTIONS = {
    "accuracy": AccuracyReward,
    "efficiency": EfficiencyReward,
    "safety": SafetyReward,
    "consistency": ConsistencyReward,
    "default": create_default_reward_function
}


def get_reward_function(name: str = "default", **kwargs) -> RewardFunction:
    """
    获取奖励函数

    Args:
        name: 奖励函数名称
        **kwargs: 额外参数

    Returns:
        RewardFunction: 奖励函数实例
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"未知的奖励函数: {name}. 可用: {list(REWARD_FUNCTIONS.keys())}")

    func_class = REWARD_FUNCTIONS[name]

    if name == "default":
        return func_class()
    else:
        return func_class(**kwargs)


if __name__ == "__main__":
    # 测试奖励函数
    print("测试奖励函数...\n")

    # 创建测试轨迹
    test_trajectory = {
        "query": "计算 1+1",
        "route_decision": "direct_turbo",
        "transitions": [
            {"state": {"status": "planning"}, "action": "planner", "next_state": {"status": "executing"}, "info": {}},
            {"state": {"status": "executing"}, "action": "tool_caller", "next_state": {"status": "completed"}, "info": {}}
        ],
        "final_answer": "计算结果: 2",
        "success": True,
        "steps": 2,
        "metadata": {
            "tool_calls": [{"tool": "calculator"}]
        }
    }

    # 测试各个奖励函数
    accuracy = AccuracyReward()
    efficiency = EfficiencyReward()
    safety = SafetyReward()
    consistency = ConsistencyReward()

    print("准确性奖励:", accuracy.compute(test_trajectory))
    print("效率奖励:", efficiency.compute(test_trajectory))
    print("安全性奖励:", safety.compute(test_trajectory))
    print("一致性奖励:", consistency.compute(test_trajectory))

    # 测试组合奖励
    composite = create_default_reward_function()
    print("\n组合奖励:", composite.compute(test_trajectory))
    print("详细:", composite.get_details(test_trajectory))
