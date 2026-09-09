"""
轨迹数据收集器

收集状态机运行过程中的轨迹数据，用于强化学习训练
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Transition:
    """
    单步转移数据

    记录从状态 s 采取行动 a 到达状态 s' 的转移
    """
    state: Dict[str, Any]
    action: str  # 采取的动作（节点名称）
    next_state: Dict[str, Any]
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "done": self.done,
            "info": self.info
        }


@dataclass
class Trajectory:
    """
    完整轨迹

    记录从开始到结束的完整执行序列
    """
    query: str
    route_decision: str
    transitions: List[Transition] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    total_reward: float = 0.0
    steps: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_transition(self, transition: Transition):
        """添加转移步"""
        self.transitions.append(transition)
        self.total_reward += transition.reward
        self.steps += 1

    def get_states(self) -> List[Dict[str, Any]]:
        """获取所有状态序列"""
        states = [t.state for t in self.transitions]
        if self.transitions:
            states.append(self.transitions[-1].next_state)
        return states

    def get_actions(self) -> List[str]:
        """获取所有动作序列"""
        return [t.action for t in self.transitions]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query": self.query,
            "route_decision": self.route_decision,
            "transitions": [t.to_dict() for t in self.transitions],
            "final_answer": self.final_answer,
            "success": self.success,
            "total_reward": self.total_reward,
            "steps": self.steps,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class TrajectoryCollector:
    """
    轨迹收集器

    收集和存储状态机运行轨迹
    """

    def __init__(self, max_buffer_size: int = 10000):
        """
        初始化收集器

        Args:
            max_buffer_size: 最大缓冲区大小
        """
        self.trajectories: List[Trajectory] = []
        self.max_buffer_size = max_buffer_size
        self.current_trajectory: Optional[Trajectory] = None

    def start_trajectory(self, query: str, route_decision: str) -> Trajectory:
        """
        开始收集新轨迹

        Args:
            query: 用户查询
            route_decision: 路由决策

        Returns:
            Trajectory: 新创建的轨迹
        """
        self.current_trajectory = Trajectory(
            query=query,
            route_decision=route_decision
        )
        return self.current_trajectory

    def add_transition(self, state: Dict[str, Any], action: str,
                      next_state: Dict[str, Any], reward: float = 0.0,
                      done: bool = False, info: Dict = None):
        """
        添加转移步

        Args:
            state: 当前状态
            action: 采取的动作
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        if self.current_trajectory is None:
            raise ValueError("没有活跃的轨迹，请先调用 start_trajectory")

        transition = Transition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
            info=info or {}
        )

        self.current_trajectory.add_transition(transition)

    def end_trajectory(self, final_answer: str, success: bool,
                     metadata: Dict = None) -> Trajectory:
        """
        结束轨迹收集

        Args:
            final_answer: 最终答案
            success: 是否成功
            metadata: 额外元数据

        Returns:
            Trajectory: 完整的轨迹
        """
        if self.current_trajectory is None:
            raise ValueError("没有活跃的轨迹")

        self.current_trajectory.final_answer = final_answer
        self.current_trajectory.success = success
        if metadata:
            self.current_trajectory.metadata.update(metadata)

        # 保存轨迹
        self.trajectories.append(self.current_trajectory)

        # 检查缓冲区大小
        if len(self.trajectories) > self.max_buffer_size:
            self.trajectories.pop(0)

        trajectory = self.current_trajectory
        self.current_trajectory = None
        return trajectory

    def get_trajectories(self, success: Optional[bool] = None) -> List[Trajectory]:
        """
        获取轨迹

        Args:
            success: 筛选成功/失败的轨迹，None 表示全部

        Returns:
            List[Trajectory]: 轨迹列表
        """
        if success is None:
            return self.trajectories.copy()
        return [t for t in self.trajectories if t.success == success]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.trajectories)
        if total == 0:
            return {
                "total_trajectories": 0,
                "success_rate": 0.0,
                "average_steps": 0.0,
                "average_reward": 0.0
            }

        success_count = sum(1 for t in self.trajectories if t.success)
        total_steps = sum(t.steps for t in self.trajectories)
        total_reward = sum(t.total_reward for t in self.trajectories)

        return {
            "total_trajectories": total,
            "success_count": success_count,
            "success_rate": success_count / total * 100,
            "average_steps": total_steps / total,
            "average_reward": total_reward / total
        }

    def clear(self):
        """清空收集器"""
        self.trajectories.clear()
        self.current_trajectory = None

    def save_to_file(self, filepath: str):
        """
        保存轨迹到文件

        Args:
            filepath: 文件路径
        """
        data = {
            "trajectories": [t.to_dict() for t in self.trajectories],
            "statistics": self.get_statistics(),
            "exported_at": datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'TrajectoryCollector':
        """
        从文件加载轨迹

        Args:
            filepath: 文件路径

        Returns:
            TrajectoryCollector: 加载后的收集器
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        collector = cls()

        for traj_data in data["trajectories"]:
            trajectory = Trajectory(
                query=traj_data["query"],
                route_decision=traj_data["route_decision"],
                final_answer=traj_data["final_answer"],
                success=traj_data["success"],
                total_reward=traj_data["total_reward"],
                steps=traj_data["steps"],
                timestamp=traj_data["timestamp"],
                metadata=traj_data.get("metadata", {})
            )

            for trans_data in traj_data["transitions"]:
                transition = Transition(
                    state=trans_data["state"],
                    action=trans_data["action"],
                    next_state=trans_data["next_state"],
                    reward=trans_data["reward"],
                    done=trans_data["done"],
                    info=trans_data.get("info", {})
                )
                trajectory.add_transition(transition)

            collector.trajectories.append(trajectory)

        return collector


# 工作流装饰器：自动收集轨迹
def collect_trajectories(workflow_class):
    """
    装饰器：为工作流类添加轨迹收集功能

    Args:
        workflow_class: 工作流类

    Returns:
        装饰后的工作流类
    """
    original_run = workflow_class.run

    def run_with_collection(self, query, max_retries=3, verbose=True, collect=True):
        """
        带轨迹收集的运行方法

        Args:
            query: 用户查询
            max_retries: 最大重试次数
            verbose: 是否输出日志
            collect: 是否收集轨迹

        Returns:
            Dict: 执行结果
        """
        if not collect or not hasattr(self, 'trajectory_collector'):
            return original_run(self, query, max_retries, verbose)

        # 开始收集轨迹
        collector = self.trajectory_collector
        route_info = {"type": "workflow"}  # 简化版

        collector.start_trajectory(query, route_info.get("type", "unknown"))

        # 执行工作流（需要修改工作流以支持回调）
        result = original_run(self, query, max_retries, verbose)

        # 结束轨迹收集
        success = result.get("status") == "completed"
        collector.end_trajectory(
            final_answer=result.get("answer", ""),
            success=success,
            metadata={
                "step_count": result.get("step_count", 0),
                "tool_calls": result.get("tool_calls", [])
            }
        )

        return result

    workflow_class.run = run_with_collection

    # 添加轨迹收集器属性
    if not hasattr(workflow_class, 'trajectory_collector'):
        workflow_class.trajectory_collector = TrajectoryCollector()

    return workflow_class


if __name__ == "__main__":
    # 测试轨迹收集
    print("测试轨迹收集器...\n")

    collector = TrajectoryCollector()

    # 模拟一条轨迹
    collector.start_trajectory("计算 1+1", "direct_turbo")

    state = {"status": "planning", "step": 0}
    next_state = {"status": "executing", "step": 1}
    collector.add_transition(state, "planner", next_state, reward=0.1)

    state = next_state
    next_state = {"status": "completed", "step": 2}
    collector.add_transition(state, "tool_caller", next_state, reward=0.8)

    collector.end_trajectory("1+1=2", success=True, metadata={"model": "turbo"})

    print(f"收集到轨迹: {len(collector.trajectories)}")
    print(f"统计: {collector.get_statistics()}")

    # 保存到文件
    collector.save_to_file("trajectories.json")
    print(f"\n轨迹已保存到 trajectories.json")

    # 加载文件
    loaded = TrajectoryCollector.load_from_file("trajectories.json")
    print(f"加载轨迹: {len(loaded.trajectories)}")
