"""
强化学习模块测试
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl import (
    TrajectoryCollector,
    Trajectory,
    create_default_reward_function,
    GRPOTrainer,
    GRPOConfig
)


def test_trajectory_collector():
    """测试轨迹收集器"""
    print("1. 测试轨迹收集器...")

    collector = TrajectoryCollector(max_buffer_size=100)

    # 创建测试轨迹
    collector.start_trajectory("测试查询", "workflow")
    collector.add_transition({"s": "planning"}, "planner", {"s": "executing"}, 0.1)
    collector.end_trajectory("答案", success=True)

    assert len(collector.trajectories) == 1
    assert collector.trajectories[0].success == True
    print("  [OK] 轨迹收集器")

    # 测试统计
    stats = collector.get_statistics()
    assert stats["total_trajectories"] == 1
    print("  [OK] 统计功能")

    # 测试保存和加载
    collector.save_to_file("test_trajectories.json")
    loaded = TrajectoryCollector.load_from_file("test_trajectories.json")
    assert len(loaded.trajectories) == 1
    print("  [OK] 保存和加载")


def test_reward_function():
    """测试奖励函数"""
    print("\n2. 测试奖励函数...")

    reward_func = create_default_reward_function()

    # 创建测试轨迹
    test_traj = {
        "query": "计算 1+1",
        "final_answer": "计算结果: 2",
        "success": True,
        "steps": 2,
        "transitions": [],
        "metadata": {"tool_calls": [{"tool": "calculator"}]}
    }

    reward = reward_func.compute(test_traj)
    assert isinstance(reward, float)
    print("  [OK] 奖励计算")

    # 测试详细奖励
    details = reward_func.get_details(test_traj)
    assert len(details) > 0
    print("  [OK] 详细奖励")


def test_grpo_trainer():
    """测试 GRPO 训练器"""
    print("\n3. 测试 GRPO 训练器...")

    config = GRPOConfig(batch_size=4, num_epochs=2)
    trainer = GRPOTrainer(config, reward_function=create_default_reward_function())

    # 创建测试数据
    test_data = []
    for i in range(10):
        test_data.append({
            "query": f"查询 {i}",
            "final_answer": f"答案 {i}",
            "success": i % 2 == 0,
            "steps": 3,
            "transitions": [],
            "metadata": {}
        })

    # 训练
    metrics = trainer.train(test_data, verbose=False)
    assert len(metrics) > 0
    print("  [OK] 训练流程")

    # 评估
    eval_result = trainer.evaluate(test_data[:5])
    assert "success_rate" in eval_result
    print("  [OK] 评估功能")

    # 摘要
    summary = trainer.get_training_summary()
    assert "total_episodes" in summary
    print("  [OK] 训练摘要")


def test_integration():
    """测试完整流程"""
    print("\n4. 测试完整流程...")

    # 1. 收集轨迹
    collector = TrajectoryCollector()
    for i in range(5):
        collector.start_trajectory(f"查询 {i}", "workflow")
        collector.add_transition({"s": "planning"}, "planner", {"s": "executing"}, 0.1)
        collector.add_transition({"s": "executing"}, "tool_caller", {"s": "completed"}, 0.8)
        collector.end_trajectory(f"答案 {i}", success=True)

    # 2. 准备数据
    training_data = [traj.to_dict() for traj in collector.trajectories]

    # 3. 训练
    config = GRPOConfig(batch_size=3, num_epochs=2)
    trainer = GRPOTrainer(config, reward_function=create_default_reward_function())
    trainer.train(training_data, verbose=False)

    # 4. 验证
    assert trainer.episode > 0
    print("  [OK] 完整流程")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("强化学习模块测试")
    print("="*60)

    try:
        test_trajectory_collector()
        test_reward_function()
        test_grpo_trainer()
        test_integration()

        print("\n" + "="*60)
        print("[OK] 所有测试通过!")
        print("="*60)

    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\n[ERROR] 测试错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
