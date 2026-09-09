"""
GSM8K 数据集加载器

GSM8K (Grade School Math 8K) 是一个包含 8.5K 个小学数学应用题的数据集
用于测试数学推理能力
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import re


class GSM8KLoader:
    """GSM8K 数据集加载器"""

    def __init__(self, data_path: str = None):
        """
        初始化加载器

        Args:
            data_path: GSM8K 数据集路径（JSON 格式）
        """
        self.data_path = data_path
        self.data = []

    def load(self, data_path: str = None) -> List[Dict[str, Any]]:
        """
        加载 GSM8K 数据集

        Args:
            data_path: 数据集路径

        Returns:
            List[Dict]: 问题列表
        """
        if data_path:
            self.data_path = data_path

        if not self.data_path:
            raise ValueError("请提供 GSM8K 数据集路径")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        return self.data

    def load_from_huggingface(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        从 HuggingFace 加载 GSM8K 数据集

        Args:
            split: 数据集分割 ("train", "test")

        Returns:
            List[Dict]: 问题列表
        """
        try:
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main")

            self.data = []
            for item in dataset[split]:
                self.data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "solution": self._extract_solution(item["answer"])
                })

            return self.data

        except ImportError:
            raise ImportError("需要安装 datasets 库: pip install datasets")

    def _extract_solution(self, answer: str) -> str:
        """
        从答案中提取解题过程

        Args:
            answer: 完整答案

        Returns:
            str: 解题过程
        """
        # GSM8K 答案格式为 "#### 数字"
        # 在 #### 之前的是解题过程
        if "####" in answer:
            parts = answer.split("####")
            return parts[0].strip()
        return answer

    def get_answer_value(self, answer: str) -> float:
        """
        从答案中提取数值

        Args:
            answer: 完整答案

        Returns:
            float: 答案数值
        """
        # 提取 #### 后面的数字
        match = re.search(r"####\s*([-\d.]+)", answer)
        if match:
            return float(match.group(1).replace(",", ""))
        return 0.0

    def format_for_training(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        将样本格式化为训练格式

        Args:
            sample: GSM8K 样本

        Returns:
            Dict: 格式化后的样本
        """
        return {
            "query": sample["question"],
            "expected_answer": self.get_answer_value(sample["answer"]),
            "solution": sample.get("solution", ""),
            "category": "math_reasoning"
        }

    def create_training_dataset(self, size: int = None,
                               shuffle: bool = True) -> List[Dict[str, Any]]:
        """
        创建训练数据集

        Args:
            size: 数据集大小
            shuffle: 是否打乱

        Returns:
            List[Dict]: 训练数据
        """
        import random

        # 格式化数据
        formatted_data = [self.format_for_training(item) for item in self.data]

        # 限制大小
        if size and size < len(formatted_data):
            formatted_data = formatted_data[:size]

        # 打乱
        if shuffle:
            random.shuffle(formatted_data)

        return formatted_data

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.data:
            return {}

        total = len(self.data)
        categories = {}

        for item in self.data:
            category = "math_reasoning"
            categories[category] = categories.get(category, 0) + 1

        return {
            "total_samples": total,
            "categories": categories
        }


# 创建示例数据（用于测试）
def create_sample_gsm8k_data() -> List[Dict[str, Any]]:
    """
    创建示例 GSM8K 数据（用于测试）

    Returns:
        List[Dict]: 示例数据
    """
    return [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": "Natalia sold 48 clips in April. In May, she sold half as many clips, so she sold 48 / 2 = <<48/2=24>>24 clips. Altogether, Natalia sold 48 + 24 = <<48+24=72>>72 clips. #### 72",
            "solution": "Natalia sold 48 clips in April. In May, she sold half as many clips, so she sold 48 / 2 = 24 clips. Altogether, Natalia sold 48 + 24 = 72 clips."
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": "Weng earns $12/hour for babysitting. 50 minutes is 50/60 = 50/60 = <<50/60=0.8333333333333333>>0.8333333333333333 hours. So she earned $12/hour * 0.8333333333333333 hours = $12 * 0.8333333333333333 = <<12*0.8333333333333333=10.0>>10.0. #### 10",
            "solution": "Weng earns $12/hour for babysitting. 50 minutes is 50/60 = 0.8333333333333333 hours. So she earned $12/hour * 0.8333333333333333 hours = $10.0."
        },
        {
            "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
            "answer": "Betty has $100 / 2 = $100/2 = <<100/2=50>>50. Her parents gave her $15. Her grandparents gave her $15 * 2 = $15*2 = <<15*2=30>>30. So, in total, Betty has $50 + $15 + $30 = $50+$15+$30 = <<50+15+30=95>>95. To buy the wallet, Betty needs $100 - $95 = $100-$95 = <<100-95=5>>5 more. #### 5",
            "solution": "Betty has $50. Her parents gave her $15. Her grandparents gave her $30. So, in total, Betty has $95. To buy the wallet, Betty needs $5 more."
        },
        {
            "question": "Julie read twice as many pages as Saturday. On Sunday, she read 20 pages. If the book has 80 pages, how many more pages does Julie need to read to finish the book?",
            "answer": "On Saturday, Julie read 20 / 2 = 20/2 = <<20/2=10>>10 pages. In total, she read 10 + 20 = 10+20 = <<10+20=30>>30 pages. The book has 80 pages, so she needs to read 80 - 30 = 80-30 = <<80-30=50>>50 more pages. #### 50",
            "solution": "On Saturday, Julie read 10 pages. In total, she read 30 pages. The book has 80 pages, so she needs to read 50 more pages."
        },
        {
            "question": "James buys a shirt that costs $20 and a pair of pants that cost $30. He gives the cashier $100. How much change does James receive?",
            "answer": "The total cost of the shirt and pants is $20 + $30 = $20+$30 = <<20+30=50>>50. James gives $100, so he receives $100 - $50 = $100-$50 = <<100-$50=50>>50 in change. #### 50",
            "solution": "The total cost of the shirt and pants is $50. James gives $100, so he receives $50 in change."
        }
    ]


if __name__ == "__main__":
    # 测试数据加载器
    print("测试 GSM8K 数据加载器...\n")

    # 使用示例数据
    sample_data = create_sample_gsm8k_data()

    # 保存示例数据
    with open("gsm8k_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    print(f"已创建示例数据: gsm8k_sample.json")

    # 测试加载
    loader = GSM8KLoader()
    loader.load("gsm8k_sample.json")

    print(f"\n数据集大小: {len(loader.data)}")
    print(f"统计信息: {loader.get_statistics()}")

    # 格式化第一个样本
    if loader.data:
        formatted = loader.format_for_training(loader.data[0])
        print(f"\n格式化样本:")
        print(f"  问题: {formatted['query']}")
        print(f"  期望答案: {formatted['expected_answer']}")
        print(f"  解题过程: {formatted['solution']}")
