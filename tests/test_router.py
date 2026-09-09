"""
动态路由模块测试

测试复杂度分析和路由决策功能
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.router import (
    ComplexityAnalyzer,
    Router,
    ComplexityLevel,
    TaskType,
    RouteDecision,
    analyze_complexity,
    route_query
)


class TestComplexityAnalyzer:
    """复杂度分析器测试"""

    def test_simple_calculation(self):
        """测试简单计算任务"""
        query = "计算 123 * 456 的结果"
        result = analyze_complexity(query)

        assert result.level == ComplexityLevel.SIMPLE
        assert result.task_type == TaskType.CALCULATION
        assert result.confidence > 0.8

    def test_proof_task(self):
        """测试证明任务"""
        query = "证明：任意两个连续整数的乘积是偶数"
        result = analyze_complexity(query)

        assert result.level in [ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX]
        assert result.task_type == TaskType.PROOF

    def test_multi_step_task(self):
        """测试多步骤任务"""
        query = "首先分析问题，然后设计解决方案，最后实现代码"
        result = analyze_complexity(query)

        assert result.task_type == TaskType.MULTI_STEP
        assert result.level != ComplexityLevel.SIMPLE

    def test_code_generation(self):
        """测试代码生成任务"""
        query = "用 Python 实现一个二叉树的遍历算法"
        result = analyze_complexity(query)

        assert result.task_type == TaskType.CODE_GEN
        assert result.is_complex()

    def test_token_estimation(self):
        """测试 Token 估算"""
        analyzer = ComplexityAnalyzer()
        short = "你好"
        long = "这是一个非常长的句子，包含了很多中文和English words，用于测试Token估算功能的准确性。"

        short_count = analyzer._estimate_tokens(short)
        long_count = analyzer._estimate_tokens(long)

        assert long_count > short_count


class TestRouter:
    """路由器测试"""

    def test_simple_route_to_turbo(self):
        """测试简单任务路由到 Turbo"""
        query = "计算 123 * 456 的结果"
        router = Router()
        result = router.route(query)

        assert result.decision == RouteDecision.DIRECT_TURBO
        assert result.model == "qwen-turbo"
        assert not result.use_workflow

    def test_complex_route_to_max(self):
        """测试复杂任务路由到 Max"""
        query = "设计一个完整的用户认证系统，包含注册、登录、密码找回、邮箱验证等功能"
        router = Router()
        result = router.route(query)

        assert result.decision in [RouteDecision.DIRECT_MAX, RouteDecision.WORKFLOW]
        assert "qwen-max" in result.model

    def test_proof_route_to_workflow(self):
        """测试证明任务路由到工作流"""
        query = "证明：素数有无穷多个"
        router = Router(enable_workflow=True)
        result = router.route(query)

        assert result.decision == RouteDecision.WORKFLOW
        assert result.use_workflow
        assert "qwen-max" in result.model

    def test_force_route(self):
        """测试强制路由"""
        query = "一个简单的查询"
        router = Router()
        result = router.route(query, force_route="workflow")

        assert result.decision == RouteDecision.WORKFLOW
        assert result.use_workflow

    def test_stats_tracking(self):
        """测试统计追踪"""
        router = Router()

        # 执行多次路由
        router.route("计算 1+1")
        router.route("计算 2+2")
        router.route("证明一个定理")

        stats = router.get_stats()
        assert stats["total_count"] == 3
        assert stats["turbo_count"] >= 2

    def test_stats_reset(self):
        """测试统计重置"""
        router = Router()
        router.route("test")
        router.reset_stats()

        stats = router.get_stats()
        assert stats["total_count"] == 0


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_analyze_complexity(self):
        """测试便捷分析函数"""
        result = analyze_complexity("计算 1+1")

        assert isinstance(result, ComplexityLevel) or hasattr(result, 'level')
        assert result.level == ComplexityLevel.SIMPLE

    def test_route_query(self):
        """测试便捷路由函数"""
        result = route_query("计算 1+1")

        assert hasattr(result, 'decision')
        assert hasattr(result, 'model')


def run_tests():
    """运行所有测试"""
    print("运行动态路由模块测试...\n")

    # 复杂度分析器测试
    print("1. 测试复杂度分析器...")
    test_analyzer = TestComplexityAnalyzer()

    tests = [
        ("简单计算任务", test_analyzer.test_simple_calculation),
        ("证明任务", test_analyzer.test_proof_task),
        ("多步骤任务", test_analyzer.test_multi_step_task),
        ("代码生成任务", test_analyzer.test_code_generation),
        ("Token 估算", test_analyzer.test_token_estimation),
    ]

    for name, test in tests:
        try:
            test()
            print(f"  [OK] {name}")
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    # 路由器测试
    print("\n2. 测试路由器...")
    test_router = TestRouter()

    tests = [
        ("简单任务路由", test_router.test_simple_route_to_turbo),
        ("复杂任务路由", test_router.test_complex_route_to_max),
        ("证明任务路由", test_router.test_proof_route_to_workflow),
        ("强制路由", test_router.test_force_route),
        ("统计追踪", test_router.test_stats_tracking),
        ("统计重置", test_router.test_stats_reset),
    ]

    for name, test in tests:
        try:
            test()
            print(f"  [OK] {name}")
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    print("\n[OK] 测试完成!")


if __name__ == "__main__":
    run_tests()
