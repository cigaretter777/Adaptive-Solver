"""
自适应复杂任务求解器 - 主入口

整合动态路由和状态机编排，提供统一的求解接口
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.router import Router, RouteDecision
from src.graph import Workflow
from src.config import get_settings, validate_settings
from src.llm import create_llm_client


class AdaptiveSolver:
    """自适应复杂任务求解器"""

    def __init__(self, verbose: bool = True):
        """
        初始化求解器

        Args:
            verbose: 是否输出详细日志
        """
        self.settings = get_settings()
        self.verbose = verbose

        # 验证配置
        if not validate_settings():
            print("警告: 配置验证失败，请检查 .env 文件")

        # 初始化路由器
        self.router = Router(
            turbo_model=self.settings.QWEN_TURBO_MODEL,
            max_model=self.settings.QWEN_MAX_MODEL,
            token_threshold=self.settings.COMPLEXITY_TOKEN_THRESHOLD,
            enable_workflow=True
        )

        # 初始化工作流（如果需要）
        self.workflow = Workflow(
            llm_client=None,  # 暂时使用模拟
            enable_langgraph=self.settings.ENABLE_LANGGRAPH
        )

        # 路由统计
        self.stats = {
            "total_queries": 0,
            "turbo_route": 0,
            "max_route": 0,
            "workflow_route": 0
        }

        if self.verbose:
            self._print_info()

    def _print_info(self):
        """打印求解器信息"""
        print("=" * 60)
        print("自适应复杂任务求解器")
        print("=" * 60)
        print(f"配置:")
        print(f"  - Turbo 模型: {self.settings.QWEN_TURBO_MODEL}")
        print(f"  - Max 模型: {self.settings.QWEN_MAX_MODEL}")
        print(f"  - Token 阈值: {self.settings.COMPLEXITY_TOKEN_THRESHOLD}")
        print(f"  - 最大重试次数: {self.settings.MAX_RETRIES}")
        print(f"  - 环境: {self.settings.ENV}")
        print("=" * 60)

    def solve(self, query: str, history: Optional[List[Dict]] = None) -> str:
        """
        求解查询

        Args:
            query: 用户查询
            history: 对话历史（用于复杂度分析）

        Returns:
            str: 解答结果
        """
        self.stats["total_queries"] += 1

        if self.verbose:
            print(f"\n[查询 {self.stats['total_queries']}] {query}")
            print("-" * 60)

        # 1. 动态路由决策
        route_result = self.router.route(query, history)

        # 更新路由统计
        if route_result.decision == RouteDecision.DIRECT_TURBO:
            self.stats["turbo_route"] += 1
        elif route_result.decision == RouteDecision.DIRECT_MAX:
            self.stats["max_route"] += 1
        elif route_result.decision == RouteDecision.WORKFLOW:
            self.stats["workflow_route"] += 1

        if self.verbose:
            print(f"路由决策: {route_result.decision.value}")
            print(f"使用模型: {route_result.model}")
            print(f"使用工作流: {'是' if route_result.use_workflow else '否'}")
            print(f"决策理由: {route_result.reasoning}")
            print(f"复杂度等级: {route_result.analysis.level.value}")

        # 2. 根据路由结果执行
        if route_result.decision == RouteDecision.DIRECT_TURBO:
            # 简单任务：直接调用 Turbo 模型
            result = self._solve_with_turbo(query, route_result.analysis)
        elif route_result.decision == RouteDecision.DIRECT_MAX:
            # 中等复杂度：直接调用 Max 模型
            result = self._solve_with_max(query, route_result.analysis)
        else:  # WORKFLOW
            # 复杂任务：使用状态机工作流
            result = self._solve_with_workflow(query, route_result.analysis)

        if self.verbose:
            print(f"\n最终结果:")
            print("-" * 60)
            print(result)
            print(f"\n累计路由: {self._get_route_stats()}")

        return result

    def _solve_with_turbo(self, query: str, analysis: Any) -> str:
        """
        使用 Turbo 模型解决简单任务

        Args:
            query: 用户查询
            analysis: 复杂度分析结果

        Returns:
            str: 解决方案
        """
        # TODO: 实际连接 Qwen Turbo API
        return f"[Turbo 模型] 快速响应: {query}\n这是一个简单的查询，已经使用 Turbo 模型高效处理。"

    def _solve_with_max(self, query: str, analysis: Any) -> str:
        """
        使用 Max 模型解决中等复杂度任务

        Args:
            query: 用户查询
            analysis: 复杂度分析结果

        Returns:
            str: 解决方案
        """
        # TODO: 实际连接 Qwen Max API
        return f"[Max 模型] 深度思考: {query}\n这是一个需要深度思考的查询，已经使用 Max 模型进行分析。"

    def _solve_with_workflow(self, query: str, analysis: Any) -> str:
        """
        使用状态机工作流解决复杂任务

        Args:
            query: 用户查询
            analysis: 复杂度分析结果

        Returns:
            str: 解决方案
        """
        if self.verbose:
            print("\n进入状态机工作流...")
            print("-" * 40)

        # 运行工作流
        result = self.workflow.run(query, max_retries=self.settings.MAX_RETRIES, verbose=True)

        if result["status"] == "completed":
            return result["answer"]
        else:
            return f"抱歉，无法完成任务。状态: {result['status']}"

    def _get_route_stats(self) -> str:
        """获取路由统计"""
        total = self.stats["total_queries"]
        if total == 0:
            return "无查询记录"

        turbo_ratio = self.stats["turbo_route"] / total * 100
        max_ratio = self.stats["max_route"] / total * 100
        workflow_ratio = self.stats["workflow_route"] / total * 100

        return f"Turbo: {self.stats['turbo_route']} ({turbo_ratio:.1f}%), Max: {self.stats['max_route']} ({max_ratio:.1f}%), Workflow: {self.stats['workflow_route']} ({workflow_ratio:.1f}%)"

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats["route_stats"] = self._get_route_stats()
        stats["router_stats"] = self.router.get_stats()
        return stats

    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_queries": 0,
            "turbo_route": 0,
            "max_route": 0,
            "workflow_route": 0
        }
        self.router.reset_stats()

    def solve_with_stream(self, query: str, history: Optional[List[Dict]] = None):
        """
        流式求解（生成器）

        Args:
            query: 用户查询
            history: 对话历史

        Yields:
            Dict: 中间结果
        """
        # 1. 路由决策
        route_result = self.router.route(query, history)

        if route_result.decision == RouteDecision.WORKFLOW:
            # 流式执行工作流
            for step in self.workflow.run_stream(query):
                yield {
                    "step": step["step"],
                    "status": step["status"],
                    "partial_result": step.get("partial_result"),
                    "route_info": route_result.to_dict()
                }
        else:
            # 直接返回结果
            result = self.solve(query, history)
            yield {
                "step": 1,
                "status": "completed",
                "result": result,
                "route_info": route_result.to_dict()
            }


def main():
    """主函数 - 命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="自适应复杂任务求解器")
    parser.add_argument("--query", type=str, help="用户查询")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")

    args = parser.parse_args()

    # 如果没有指定任何参数，默认进入交互模式
    if not args.query and not args.interactive and not args.stats:
        args.interactive = True

    # 创建求解器
    solver = AdaptiveSolver(verbose=not args.quiet)

    if args.interactive:
        # 交互模式
        print("\n进入交互模式，输入 'quit' 或 'exit' 退出")
        while True:
            query = input("\n请输入您的查询: ").strip()
            if query.lower() in ["quit", "exit", "退出"]:
                break
            if not query:
                continue

            result = solver.solve(query)
            print(f"\n解答: {result}")

        # 显示统计
        if args.stats:
            print(f"\n{solver.get_stats()}")

    else:
        # 单次查询
        if args.query:
            result = solver.solve(args.query)
            print(result)

        # 显示统计
        if args.stats:
            print(f"\n{solver.get_stats()}")


if __name__ == "__main__":
    main()
