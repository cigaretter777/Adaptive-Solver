"""
动态路由模块

提供基于任务复杂度的智能路由功能，实现：
- 任务复杂度分析
- 智能路由决策
- 成本优化的模型选择
"""

from .complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityAnalysis,
    ComplexityLevel,
    TaskType,
    analyze_complexity
)

from .router import (
    Router,
    RouteDecision,
    RouteResult,
    route_query
)

__all__ = [
    # 复杂度分析
    "ComplexityAnalyzer",
    "ComplexityAnalysis",
    "ComplexityLevel",
    "TaskType",
    "analyze_complexity",
    # 路由决策
    "Router",
    "RouteDecision",
    "RouteResult",
    "route_query",
]
