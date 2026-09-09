"""
复杂任务示例

展示如何使用自适应求解器处理复杂任务
"""

import sys
from pathlib import Path
import json

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import AdaptiveSolver


def main():
    """运行复杂任务示例"""
    print("=" * 60)
    print("复杂任务示例")
    print("=" * 60)

    # 创建求解器
    solver = AdaptiveSolver(verbose=True)

    # 复杂任务列表
    complex_tasks = [
        "证明：任意两个连续整数的乘积是偶数",
        "分析斐波那契数列的性质，并推导通项公式",
        "设计一个二叉树的遍历算法，包括前序、中序和后序遍历",
        "为什么天空是蓝色的？请从物理角度解释",
        "用递归方法解决汉诺塔问题"
    ]

    print("\n处理复杂任务...\n")

    for task in complex_tasks:
        print(f"\n{'=' * 70}")
        print(f"复杂任务: {task}")
        print(f"{'=' * 70}")

        result = solver.solve(task)
        print(f"\n最终结果:")
        print(f"{'-' * 70}")
        print(result)

    # 显示最终统计
    print(f"\n{'=' * 60}")
    print("统计信息")
    print(f"{'=' * 60}")
    stats = solver.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
