"""
GRPO 训练器
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import random


@dataclass
class GRPOConfig:
    """GRPO 训练配置"""
    # 基本配置
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 3

    # GRPO 特定参数
    group_size: int = 4  # 每组生成的样本数
    kl_penalty: float = 0.1  # KL 散度惩罚系数
    clip_range: float = 0.2  # 裁剪范围

    # 优势计算
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE 参数

    # 模型配置
    model_name: str = "qwen-turbo"
    lora_rank: int = 8  # LoRA 秩
    lora_alpha: int = 16  # LoRA alpha

    # 保存配置
    save_steps: int = 100
    output_dir: str = "./checkpoints"


@dataclass
class TrainingMetrics:
    """训练指标"""
    episode: int
    total_reward: float
    success_rate: float
    avg_steps: float
    policy_loss: float = 0.0
    value_loss: float = 0.0
    kl_divergence: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class GRPOTrainer:
    """
    GRPO 训练器

    使用 GRPO 算法训练策略模型
    """

    def __init__(self, config: GRPOConfig, reward_function=None):
        """
        初始化训练器

        Args:
            config: 训练配置
            reward_function: 奖励函数
        """
        self.config = config
        self.reward_function = reward_function

        # 训练状态
        self.episode = 0
        self.metrics_history: List[TrainingMetrics] = []

        # 模型相关（占位符，实际使用时需要连接真实模型）
        self.model = None
        self.ref_model = None
        self.policy_model = None

        # 数据缓冲区
        self.buffer: List[Dict[str, Any]] = []

        print("GRPO 训练器初始化完成")
        print(f"配置: {config}")

    def collect_experience(self, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        收集经验

        Args:
            trajectories: 轨迹列表

        Returns:
            带奖励的轨迹列表
        """
        processed = []

        for traj in trajectories:
            # 计算奖励
            if self.reward_function:
                reward = self.reward_function.compute(traj)
            else:
                reward = traj.get("total_reward", 0.0)

            # 计算优势（简化版）
            advantage = self._compute_advantage(traj, reward)

            processed.append({
                "trajectory": traj,
                "reward": reward,
                "advantage": advantage
            })

        return processed

    def _compute_advantage(self, trajectory: Dict[str, Any], reward: float) -> float:
        """
        计算优势值

        Args:
            trajectory: 轨迹
            reward: 总奖励

        Returns:
            float: 优势值
        """
        # 简化版：使用总奖励作为优势
        # 实际实现应该使用 GAE (Generalized Advantage Estimation)
        return reward

    def train_step(self, batch: List[Dict[str, Any]]) -> TrainingMetrics:
        """
        执行一步训练

        Args:
            batch: 批次数据

        Returns:
            TrainingMetrics: 训练指标
        """
        # 计算平均奖励
        avg_reward = sum(item["reward"] for item in batch) / len(batch)

        # 计算成功率
        success_count = sum(1 for item in batch if item["trajectory"].get("success", False))
        success_rate = success_count / len(batch)

        # 计算平均步数
        avg_steps = sum(item["trajectory"].get("steps", 0) for item in batch) / len(batch)

        # 模拟损失值（实际训练时从模型获取）
        policy_loss = random.uniform(0.1, 0.5)
        value_loss = random.uniform(0.1, 0.3)
        kl_divergence = random.uniform(0.0, 0.1)

        metrics = TrainingMetrics(
            episode=self.episode,
            total_reward=avg_reward,
            success_rate=success_rate,
            avg_steps=avg_steps,
            policy_loss=policy_loss,
            value_loss=value_loss,
            kl_divergence=kl_divergence
        )

        self.metrics_history.append(metrics)
        self.episode += 1

        return metrics

    def train(self, dataset: List[Dict[str, Any]], num_episodes: int = None,
              verbose: bool = True) -> List[TrainingMetrics]:
        """
        训练模型

        Args:
            dataset: 训练数据集
            num_episodes: 训练轮数
            verbose: 是否输出日志

        Returns:
            List[TrainingMetrics]: 训练指标历史
        """
        if num_episodes is None:
            num_episodes = self.config.num_epochs

        batch_size = self.config.batch_size

        if verbose:
            print(f"\n开始训练，总轮数: {num_episodes}")
            print(f"批次大小: {batch_size}")
            print(f"数据集大小: {len(dataset)}")
            print("=" * 60)

        for episode in range(num_episodes):
            # 随机打乱数据
            random.shuffle(dataset)

            # 分批
            num_batches = (len(dataset) + batch_size - 1) // batch_size
            episode_metrics = []

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = start + batch_size
                batch = dataset[start:end]

                # 处理批次数据（计算奖励）
                processed_batch = self.collect_experience(batch)

                # 训练一步
                metrics = self.train_step(processed_batch)
                episode_metrics.append(metrics)

            # 计算本轮平均指标
            avg_reward = sum(m.total_reward for m in episode_metrics) / len(episode_metrics)
            avg_success = sum(m.success_rate for m in episode_metrics) / len(episode_metrics)

            if verbose:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  平均奖励: {avg_reward:.4f}")
                print(f"  成功率: {avg_success:.4f}")
                print(f"  策略损失: {episode_metrics[-1].policy_loss:.4f}")
                print(f"  KL 散度: {episode_metrics[-1].kl_divergence:.4f}")

            # 保存检查点
            if (episode + 1) % self.config.save_steps == 0:
                self.save_checkpoint(f"checkpoint_{episode + 1}")

        if verbose:
            print("\n训练完成!")
            print("=" * 60)

        return self.metrics_history

    def save_checkpoint(self, name: str):
        """
        保存检查点

        Args:
            name: 检查点名称
        """
        checkpoint = {
            "config": self.config.__dict__,
            "episode": self.episode,
            "metrics": [m.__dict__ for m in self.metrics_history],
            "timestamp": datetime.now().isoformat()
        }

        filepath = f"{self.config.output_dir}/{name}.json"
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        if hasattr(self, 'buffer'):
            checkpoint["buffer_size"] = len(self.buffer)

    def load_checkpoint(self, filepath: str):
        """
        加载检查点

        Args:
            filepath: 检查点文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        self.episode = checkpoint.get("episode", 0)

        # 恢复指标
        for m_data in checkpoint.get("metrics", []):
            metrics = TrainingMetrics(**m_data)
            self.metrics_history.append(metrics)

        print(f"已加载检查点: {filepath}")
        print(f"当前轮数: {self.episode}")

    def evaluate(self, test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估模型

        Args:
            test_dataset: 测试数据集

        Returns:
            Dict: 评估结果
        """
        results = []

        for item in test_dataset:
            reward = self.reward_function.compute(item) if self.reward_function else item.get("total_reward", 0)
            success = item.get("success", False)
            steps = item.get("steps", 0)

            results.append({
                "reward": reward,
                "success": success,
                "steps": steps
            })

        # 计算平均指标
        avg_reward = sum(r["reward"] for r in results) / len(results)
        success_rate = sum(r["success"] for r in results) / len(results)
        avg_steps = sum(r["steps"] for r in results) / len(results)

        return {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "total_samples": len(results)
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.metrics_history:
            return {"status": "no training data"}

        latest = self.metrics_history[-1]
        best_reward = max(m.total_reward for m in self.metrics_history)
        best_success = max(m.success_rate for m in self.metrics_history)

        return {
            "total_episodes": len(self.metrics_history),
            "latest_reward": latest.total_reward,
            "latest_success_rate": latest.success_rate,
            "best_reward": best_reward,
            "best_success_rate": best_success,
            "final_kl_divergence": latest.kl_divergence
        }


def create_grpo_trainer(config: Optional[GRPOConfig] = None,
                       reward_function=None) -> GRPOTrainer:
    """
    创建 GRPO 训练器

    Args:
        config: 训练配置
        reward_function: 奖励函数

    Returns:
        GRPOTrainer: 训练器实例
    """
    if config is None:
        config = GRPOConfig()

    return GRPOTrainer(config, reward_function)


if __name__ == "__main__":
    # 测试 GRPO 训练器
    print("测试 GRPO 训练器...\n")

    # 创建配置
    config = GRPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        num_epochs=3,
        output_dir="./test_checkpoints"
    )

    # 创建训练器
    trainer = create_grpo_trainer(config)

    # 创建模拟数据
    mock_dataset = []
    for i in range(16):
        mock_dataset.append({
            "query": f"测试查询 {i}",
            "success": random.choice([True, False]),
            "steps": random.randint(2, 10),
            "final_answer": f"答案 {i}",
            "transitions": []
        })

    # 训练
    metrics = trainer.train(mock_dataset, verbose=True)

    # 获取摘要
    summary = trainer.get_training_summary()
    print(f"\n训练摘要: {json.dumps(summary, indent=2, ensure_ascii=False)}")

    # 评估
    test_data = mock_dataset[:4]
    eval_result = trainer.evaluate(test_data)
    print(f"\n评估结果: {json.dumps(eval_result, indent=2, ensure_ascii=False)}")
