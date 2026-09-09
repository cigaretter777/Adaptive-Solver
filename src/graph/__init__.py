"""
状态机编排模块

基于 LangGraph 构建可回退的 Agent 工作流，提供：
- 状态定义和管理
- 四个核心节点：Planner、Tool_Caller、Critic、Executor
- 完整的工作流定义
- Self-Correction 机制
"""

from .state import (
    AgentState,
    create_initial_state,
    update_message,
    update_status,
    increment_error,
    reset_errors,
    should_retry,
    is_completed,
    is_failed,
    has_plan,
    has_subtasks,
    get_current_subtask,
    advance_subtask,
    all_subtasks_completed
)

from .nodes.planner import (
    Planner,
    planner_node
)

from .nodes.tool_caller import (
    ToolCaller,
    ToolCall,
    tool_caller_node
)

from .nodes.critic import (
    Critic,
    critic_node
)

from .nodes.executor import (
    Executor,
    executor_node
)

from .workflow import (
    Workflow,
    WorkflowStatus,
    run_workflow
)

__all__ = [
    # 状态管理
    "AgentState",
    "create_initial_state",
    "update_message",
    "update_status",
    "increment_error",
    "reset_errors",
    "should_retry",
    "is_completed",
    "is_failed",
    "has_plan",
    "has_subtasks",
    "get_current_subtask",
    "advance_subtask",
    "all_subtasks_completed",
    # 节点
    "Planner",
    "planner_node",
    "ToolCaller",
    "ToolCall",
    "tool_caller_node",
    "Critic",
    "critic_node",
    "Executor",
    "executor_node",
    # 工作流
    "Workflow",
    "WorkflowStatus",
    "run_workflow",
]
