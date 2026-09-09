"""
路由器演示

展示动态路由功能，帮助理解路由决策机制
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.router import analyze_complexity, route_query
from src.main import AdaptiveSolver


def demonstrate_complexity_analysis():
    """演示复杂度分析功能"""
    print("=" * 60)
    print("任务复杂度分析演示")
    print("=" * 60)

    test_queries = [
        "1 + 1 = ?",
        "计算圆的面积",
        "设计一个完整的电商网站",
        "证明勾股定理",
        "分析这段代码的性能瓶颈"
    ]

    print("\n分析任务复杂度:\n")

    for query in test_queries:
        result = analyze_complexity(query)
        print(f"{'-' * 60}")
        print(f"查询: {query}")
        print(f"复杂度: {result.level.value}")
        print(f"任务类型: {result.task_type.value}")
        print(f"Token 数: {result.token_count}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"判断理由: {', '.join(result.reasons)}")


def demonstrate_routing():
    """演示路由决策"""
    print("\n" + "=" * 60)
    print("路由决策演示")
    print("=" * 60)

    test_queries = [
        "1 + 1 = ?",
        "设计一个电商系统",
        "证明勾股定理",
        "Python 是什么？"
    ]

    print("\n路由决策结果:\n")

    for query in test_queries:
        result = route_query(query)
        print(f"{'-' * 60}")
        print(f"查询: {query}")
        print(f"路由: {result.decision.value}")
        print(f"模型: {result.model}")
        print(f"使用工作流: {'是' if result.use_workflow else '否'}")
        print(f"理由: {result.reasoning}")


def demonstrate_solver():
    """演示求解器的完整流程"""
    print("\n" + "=" * 60)
    print("完整求解流程演示")
    print("=" * 60)

    # 创建求解器（安静模式）
    solver = AdaptiveSolver(verbose=False)

    test_queries = [
        "计算 2^10",
        "为什么月亮会跟着人走？",
        "设计一个登录页面的前端代码"
    ]

    print("\n运行求解器...\n")

    for i, query in enumerate(test_queries, 1):
        print(f"[任务 {i}] {query}")
        result = solver.solve(query)
        print(f"长度: {len(result)} 字符")
        print(f"路由统计: {solver.get_stats()['route_stats']}")
        print()

    # 显示详细统计
    print(f"详细统计:")
    stats = solver.get_stats()
    for key, value in stats.items():
        if key != "router_stats":
            print(f"  {key}: {value}")


def main():
    """主函数"""
    print("自适应路由演示程序\n")

    # 1. 复杂度分析
    demonstrate_complexity_analysis()

    # 2. 路由决策
    demonstrate_routing()

    # 3. 完整求解流程
    demonstrate_solver()


if __name__ == "__main__":
    main()
