"""
强化学习示例

演示如何使用轨迹收集、奖励函数和 GRPO 训练器
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl import (
    TrajectoryCollector,
    create_default_reward_function,
    GRPOTrainer,
    GRPOConfig
)


def demo_trajectory_collection():
    """演示轨迹收集"""
    print("=" * 60)
    print("1. 轨迹收集演示")
    print("=" * 60)

    collector = TrajectoryCollector(max_buffer_size=100)

    # 模拟收集几条轨迹
    test_queries = [
        ("计算 1+1", "direct_turbo"),
        ("证明勾股定理", "workflow"),
        ("分析算法复杂度", "workflow"),
        ("设计用户系统", "direct_turbo")
    ]

    for query, route in test_queries:
        collector.start_trajectory(query, route)

        # 添加一些转移步
        state = {"status": "planning", "step": 0}
        next_state = {"status": "executing", "step": 1}
        collector.add_transition(state, "planner", next_state, reward=0.1)

        state = next_state
        next_state = {"status": "completed", "step": 2}
        collector.add_transition(state, "tool_caller", next_state, reward=0.8)

        # 结束轨迹
        success = route == "direct_turbo"  # 简化：直接路由成功
        collector.end_trajectory(
            final_answer=f"已处理: {query}",
            success=success,
            metadata={"steps": 2}
        )

    # 显示统计
    stats = collector.get_statistics()
    print(f"\n收集统计:")
    print(f"  总轨迹数: {stats['total_trajectories']}")
    print(f"  成功率: {stats['success_rate']:.2f}%")
    print(f"  平均步数: {stats['average_steps']:.2f}")
    print(f"  平均奖励: {stats['average_reward']:.4f}")

    # 保存轨迹
    collector.save_to_file("demo_trajectories.json")
    print(f"\n轨迹已保存到 demo_trajectories.json")

    return collector


def demo_reward_function(collector):
    """演示奖励函数"""
    print("\n" + "=" * 60)
    print("2. 奖励函数演示")
    print("=" * 60)

    reward_func = create_default_reward_function()

    print("\n计算各轨迹的奖励:")
    for i, traj in enumerate(collector.trajectories, 1):
        traj_dict = traj.to_dict()
        total_reward = reward_func.compute(traj_dict)
        details = reward_func.get_details(traj_dict)

        print(f"\n轨迹 {i}: {traj_dict['query']}")
        print(f"  总奖励: {total_reward:.4f}")
        print(f"  详细奖励:")
        for name, value in details.items():
            print(f"    {name}: {value:.4f}")


def demo_grpo_training():
    """演示 GRPO 训练"""
    print("\n" + "=" * 60)
    print("3. GRPO 训练演示")
    print("=" * 60)

    # 创建配置
    config = GRPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        num_epochs=2,
        output_dir="./demo_checkpoints"
    )

    # 创建训练器
    trainer = GRPOTrainer(config, reward_function=create_default_reward_function())

    # 创建训练数据
    training_data = []
    for i in range(20):
        training_data.append({
            "query": f"训练查询 {i}",
            "route_decision": "workflow" if i % 2 == 0 else "direct_turbo",
            "final_answer": f"答案 {i}",
            "success": i % 3 != 0,  # 2/3 成功率
            "steps": 3 + (i % 5),
            "transitions": [],
            "metadata": {}
        })

    # 训练
    print("\n开始训练...")
    metrics = trainer.train(training_data, verbose=True)

    # 获取训练摘要
    summary = trainer.get_training_summary()
    print(f"\n训练摘要:")
    print(f"  总轮数: {summary['total_episodes']}")
    print(f"  最佳奖励: {summary['best_reward']:.4f}")
    print(f"  最佳成功率: {summary['best_success_rate']:.2%}")

    # 评估
    test_data = training_data[:5]
    eval_result = trainer.evaluate(test_data)
    print(f"\n评估结果:")
    print(f"  平均奖励: {eval_result['avg_reward']:.4f}")
    print(f"  成功率: {eval_result['success_rate']:.2%}")

    return trainer


def demo_full_pipeline():
    """演示完整流程：收集 -> 奖励 -> 训练"""
    print("\n" + "=" * 60)
    print("4. 完整流程演示")
    print("=" * 60)

    # 1. 收集轨迹
    collector = TrajectoryCollector()

    for i in range(10):
        collector.start_trajectory(f"查询 {i}", "workflow")
        collector.add_transition({"s": "planning"}, "planner", {"s": "executing"}, 0.1)
        collector.add_transition({"s": "executing"}, "tool_caller", {"s": "reviewing"}, 0.5)
        collector.add_transition({"s": "reviewing"}, "executor", {"s": "completed"}, 0.3)
        collector.end_trajectory(f"答案 {i}", success=True)

    print(f"收集了 {len(collector.trajectories)} 条轨迹")

    # 2. 创建训练数据
    reward_func = create_default_reward_function()
    training_data = [traj.to_dict() for traj in collector.trajectories]

    # 3. 训练
    config = GRPOConfig(batch_size=4, num_epochs=2)
    trainer = GRPOTrainer(config, reward_func)
    trainer.train(training_data, verbose=True)

    # 4. 评估
    eval_result = trainer.evaluate(training_data[:5])
    print(f"\n最终评估: 成功率 = {eval_result['success_rate']:.2%}")


def main():
    """主函数"""
    print("强化学习演示程序\n")

    # 演示各个组件
    collector = demo_trajectory_collection()
    demo_reward_function(collector)
    demo_grpo_training()
    demo_full_pipeline()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    print("\n说明:")
    print("- 轨迹收集: 收集状态机运行数据")
    print("- 奖励函数: 评估执行质量")
    print("- GRPO 训练: 优化策略模型（框架实现）")
    print("- 实际训练需要在阿里云 PAI 等平台进行")


if __name__ == "__main__":
    main()
