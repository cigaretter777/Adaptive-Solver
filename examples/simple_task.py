"""
简单任务示例

展示如何使用自适应求解器处理简单任务
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import AdaptiveSolver


def main():
    """运行简单任务示例"""
    print("=" * 60)
    print("简单任务示例")
    print("=" * 60)

    # 创建求解器
    solver = AdaptiveSolver(verbose=True)

    # 简单任务列表
    simple_tasks = [
        "计算 123 * 456",
        "北京的首都是哪里？",
        "一年有多少天？",
        "1 + 1 = ?",
        "Python 是什么编程语言？"
    ]

    print("\n处理简单任务...\n")

    for task in simple_tasks:
        print(f"\n{'-' * 60}")
        result = solver.solve(task)
        print(f"\n结果: {result}")

    # 显示最终统计
    print(f"\n{'=' * 60}")
    print("统计信息")
    print(f"{'=' * 60}")
    print(solver.get_stats())


if __name__ == "__main__":
    main()
