"""
动态路由器

根据任务复杂度分析结果，智能选择处理策略：
- 简单任务 → 直接调用 Qwen Turbo（低成本、快速响应）
- 复杂任务 → 调用 Qwen Max + 状态机工作流（高能力、深度思考）
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass

from .complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityAnalysis,
    ComplexityLevel,
    TaskType
)


class RouteDecision(Enum):
    """路由决策"""
    DIRECT_TURBO = "direct_turbo"      # 直接使用 Turbo
    DIRECT_MAX = "direct_max"          # 直接使用 Max
    WORKFLOW = "workflow"              # 进入状态机工作流
    REJECT = "reject"                  # 拒绝处理


@dataclass
class RouteResult:
    """路由结果"""
    decision: RouteDecision
    model: str
    use_workflow: bool
    reasoning: str
    analysis: ComplexityAnalysis
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "decision": self.decision.value,
            "model": self.model,
            "use_workflow": self.use_workflow,
            "reasoning": self.reasoning,
            "complexity_level": self.analysis.level.value,
            "task_type": self.analysis.task_type.value,
            "confidence": self.analysis.confidence,
            "token_count": self.analysis.token_count
        }


class Router:
    """
    动态路由器

    根据任务复杂度、任务类型、置信度等因素，智能选择处理策略
    """

    # 各模型配置
    MODEL_CONFIG = {
        "turbo": {
            "name": "qwen-turbo",
            "max_tokens": 1500,
            "temperature": 0.7,
            "description": "低成本模型，适合简单任务"
        },
        "max": {
            "name": "qwen-max-longcontext",
            "max_tokens": 8000,
            "temperature": 0.3,
            "description": "高性能模型，适合复杂任务"
        }
    }

    # 需要工作流处理的任务类型
    WORKFLOW_REQUIRED_TYPES = [
        TaskType.PROOF,
        TaskType.MULTI_STEP,
        TaskType.CODE_GEN
    ]

    def __init__(self,
                 turbo_model: str = "qwen-turbo",
                 max_model: str = "qwen-max-longcontext",
                 token_threshold: int = 500,
                 enable_workflow: bool = True):
        """
        初始化路由器

        Args:
            turbo_model: Turbo 模型名称
            max_model: Max 模型名称
            token_threshold: 复杂度 Token 阈值
            enable_workflow: 是否启用工作流
        """
        self.turbo_model = turbo_model
        self.max_model = max_model
        self.token_threshold = token_threshold
        self.enable_workflow = enable_workflow
        self.complexity_analyzer = ComplexityAnalyzer(token_threshold=token_threshold)

        # 路由统计
        self.stats = {
            "turbo_count": 0,
            "max_count": 0,
            "workflow_count": 0,
            "total_count": 0
        }

    def route(self,
              query: str,
              history: Optional[List[Dict]] = None,
              force_route: Optional[str] = None) -> RouteResult:
        """
        执行路由决策

        Args:
            query: 用户输入
            history: 对话历史
            force_route: 强制指定路由（用于测试）

        Returns:
            RouteResult: 路由结果
        """
        # 分析复杂度
        analysis = self.complexity_analyzer.analyze(query, history)

        # 如果强制指定路由
        if force_route:
            return self._force_route(query, force_route, analysis)

        # 正常路由决策
        decision, model, use_workflow, reasoning = self._make_decision(analysis, history)

        # 更新统计
        self._update_stats(decision, use_workflow)

        return RouteResult(
            decision=decision,
            model=model,
            use_workflow=use_workflow,
            reasoning=reasoning,
            analysis=analysis,
            metadata=self._build_metadata(analysis, decision)
        )

    def _make_decision(self,
                      analysis: ComplexityAnalysis,
                      history: Optional[List[Dict]]) -> tuple:
        """
        核心路由决策逻辑

        Returns:
            (decision, model, use_workflow, reasoning)
        """
        complexity_level = analysis.level
        task_type = analysis.task_type
        confidence = analysis.confidence

        # 1. 极高置信度的简单任务 → Turbo
        if (complexity_level == ComplexityLevel.SIMPLE and
            confidence > 0.85 and
            task_type in [TaskType.SIMPLE_QA, TaskType.CALCULATION]):
            return (
                RouteDecision.DIRECT_TURBO,
                self.turbo_model,
                False,
                f"简单{task_type.value}任务，置信度 {confidence:.2f}，使用 Turbo 快速响应"
            )

        # 2. 需要工作流的任务类型 → Max + Workflow
        if (self.enable_workflow and
            task_type in self.WORKFLOW_REQUIRED_TYPES):
            return (
                RouteDecision.WORKFLOW,
                self.max_model,
                True,
                f"{task_type.value}任务需要多步推理，使用工作流模式"
            )

        # 3. 复杂任务 → Max
        if complexity_level == ComplexityLevel.COMPLEX:
            return (
                RouteDecision.DIRECT_MAX,
                self.max_model,
                False,
                f"复杂任务（评分 {analysis.features.get('complexity_score', 0):.2f}），使用 Max 深度思考"
            )

        # 4. 中等复杂度任务 → 根据 Token 数量决定
        if complexity_level == ComplexityLevel.MEDIUM:
            if analysis.token_count < self.token_threshold:
                return (
                    RouteDecision.DIRECT_TURBO,
                    self.turbo_model,
                    False,
                    f"中等复杂度任务但输入较短，使用 Turbo"
                )
            else:
                return (
                    RouteDecision.DIRECT_MAX,
                    self.max_model,
                    False,
                    f"中等复杂度任务且输入较长，使用 Max"
                )

        # 5. 默认 → Turbo
        return (
            RouteDecision.DIRECT_TURBO,
            self.turbo_model,
            False,
            "默认路由到 Turbo"
        )

    def _force_route(self, query: str, force_route: str, analysis: ComplexityAnalysis) -> RouteResult:
        """强制路由（用于测试）"""
        force_route = force_route.lower()

        if force_route == "turbo":
            decision = RouteDecision.DIRECT_TURBO
            model = self.turbo_model
            use_workflow = False
            reasoning = "强制路由到 Turbo"
        elif force_route == "max":
            decision = RouteDecision.DIRECT_MAX
            model = self.max_model
            use_workflow = False
            reasoning = "强制路由到 Max"
        elif force_route == "workflow":
            decision = RouteDecision.WORKFLOW
            model = self.max_model
            use_workflow = True
            reasoning = "强制路由到工作流"
        else:
            # 回退到正常决策
            return self._make_decision(analysis, None)

        return RouteResult(
            decision=decision,
            model=model,
            use_workflow=use_workflow,
            reasoning=reasoning,
            analysis=analysis,
            metadata=self._build_metadata(analysis, decision)
        )

    def _update_stats(self, decision: RouteDecision, use_workflow: bool):
        """更新路由统计"""
        self.stats["total_count"] += 1

        if use_workflow:
            self.stats["workflow_count"] += 1
        elif decision == RouteDecision.DIRECT_TURBO:
            self.stats["turbo_count"] += 1
        elif decision == RouteDecision.DIRECT_MAX:
            self.stats["max_count"] += 1

    def _build_metadata(self, analysis: ComplexityAnalysis, decision: RouteDecision) -> Dict[str, Any]:
        """构建元数据"""
        return {
            "decision_timestamp": None,  # 可以添加时间戳
            "complexity_score": analysis.features.get("complexity_score", 0),
            "token_count": analysis.token_count,
            "task_type": analysis.task_type.value,
            "confidence": analysis.confidence,
            "reasons": analysis.reasons
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        total = self.stats["total_count"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "turbo_ratio": f"{self.stats['turbo_count'] / total * 100:.1f}%",
            "max_ratio": f"{self.stats['max_count'] / total * 100:.1f}%",
            "workflow_ratio": f"{self.stats['workflow_count'] / total * 100:.1f}%"
        }

    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "turbo_count": 0,
            "max_count": 0,
            "workflow_count": 0,
            "total_count": 0
        }

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取模型配置"""
        return self.MODEL_CONFIG.get(model_type, self.MODEL_CONFIG["turbo"])


# 便捷函数
def route_query(query: str,
                history: Optional[List[Dict]] = None,
                turbo_model: str = "qwen-turbo",
                max_model: str = "qwen-max-longcontext",
                token_threshold: int = 500) -> RouteResult:
    """
    便捷函数：快速路由查询

    Args:
        query: 用户输入
        history: 对话历史
        turbo_model: Turbo 模型名称
        max_model: Max 模型名称
        token_threshold: 复杂度阈值

    Returns:
        RouteResult: 路由结果
    """
    router = Router(
        turbo_model=turbo_model,
        max_model=max_model,
        token_threshold=token_threshold
    )
    return router.route(query, history)


if __name__ == "__main__":
    # 测试示例
    test_queries = [
        "计算 123 * 456 的结果",
        "证明：任意两个连续整数的乘积是偶数",
        "分析这个算法的时间复杂度，并给出优化建议",
        "Python 中如何实现一个二叉树的遍历？",
        "设计一个用户认证系统，包含注册、登录和密码找回功能"
    ]

    router = Router()

    print("=" * 60)
    print("动态路由器测试")
    print("=" * 60)

    for query in test_queries:
        result = router.route(query)
        print(f"\n{'─' * 60}")
        print(f"查询: {query}")
        print(f"{'─' * 60}")
        print(f"路由决策: {result.decision.value}")
        print(f"使用模型: {result.model}")
        print(f"使用工作流: {'是' if result.use_workflow else '否'}")
        print(f"决策理由: {result.reasoning}")
        print(f"复杂度等级: {result.analysis.level.value}")
        print(f"任务类型: {result.analysis.task_type.value}")
        print(f"Token 数量: {result.analysis.token_count}")
        print(f"置信度: {result.analysis.confidence:.2f}")

    # 显示统计
    print(f"\n{'=' * 60}")
    print("路由统计")
    print(f"{'=' * 60}")
    stats = router.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
