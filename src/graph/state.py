"""
状态机状态定义

定义 Agent 工作流的状态结构，使用 TypedDict 实现类型安全的状态管理。
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from dataclasses import dataclass, field


class AgentState(TypedDict):
    """
    Agent 状态定义

    状态流转：
    Initial -> Planner -> Tool_Caller -> Critic -> Executor
                          ^_______________|
    """

    # 消息历史
    messages: List[Dict[str, Any]]

    # 用户原始查询
    query: str

    # 任务规划
    plan: Optional[str]
    subtasks: List[str]
    current_subtask_index: int

    # 工具调用
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    # 执行状态
    status: str  # "planning", "executing", "reviewing", "completed", "failed"

    # 反思/批评
    criticism: Optional[str]
    suggestions: List[str]

    # 错误处理
    error_count: int
    max_retries: int
    last_error: Optional[str]

    # 最终答案
    answer: Optional[str]

    # 元数据
    metadata: Dict[str, Any]


def create_initial_state(query: str, max_retries: int = 3) -> AgentState:
    """
    创建初始状态

    Args:
        query: 用户查询
        max_retries: 最大重试次数

    Returns:
        AgentState: 初始状态
    """
    return {
        "messages": [{"role": "user", "content": query}],
        "query": query,
        "plan": None,
        "subtasks": [],
        "current_subtask_index": 0,
        "tool_calls": [],
        "tool_results": [],
        "status": "planning",
        "criticism": None,
        "suggestions": [],
        "error_count": 0,
        "max_retries": max_retries,
        "last_error": None,
        "answer": None,
        "metadata": {
            "start_time": None,
            "step_count": 0
        }
    }


def update_message(state: AgentState, role: str, content: str) -> AgentState:
    """
    添加消息到历史

    Args:
        state: 当前状态
        role: 消息角色 (user, assistant, system, tool)
        content: 消息内容

    Returns:
        AgentState: 更新后的状态
    """
    state["messages"].append({"role": role, "content": content})
    return state


def update_status(state: AgentState, status: str) -> AgentState:
    """
    更新执行状态

    Args:
        state: 当前状态
        status: 新状态

    Returns:
        AgentState: 更新后的状态
    """
    state["status"] = status
    return state


def increment_error(state: AgentState, error: str) -> AgentState:
    """
    增加错误计数

    Args:
        state: 当前状态
        error: 错误信息

    Returns:
        AgentState: 更新后的状态
    """
    state["error_count"] += 1
    state["last_error"] = error
    return state


def reset_errors(state: AgentState) -> AgentState:
    """
    重置错误计数

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    state["error_count"] = 0
    state["last_error"] = None
    return state


# 便捷的状态检查函数
def should_retry(state: AgentState) -> bool:
    """检查是否应该重试"""
    return state["error_count"] < state["max_retries"]


def is_completed(state: AgentState) -> bool:
    """检查是否已完成"""
    return state["status"] == "completed"


def is_failed(state: AgentState) -> bool:
    """检查是否失败"""
    return state["status"] == "failed"


def has_plan(state: AgentState) -> bool:
    """检查是否有计划"""
    return state["plan"] is not None and len(state["plan"]) > 0


def has_subtasks(state: AgentState) -> bool:
    """检查是否有子任务"""
    return len(state["subtasks"]) > 0


def get_current_subtask(state: AgentState) -> Optional[str]:
    """获取当前子任务"""
    if not has_subtasks(state):
        return None
    if state["current_subtask_index"] >= len(state["subtasks"]):
        return None
    return state["subtasks"][state["current_subtask_index"]]


def advance_subtask(state: AgentState) -> AgentState:
    """推进到下一个子任务"""
    state["current_subtask_index"] += 1
    return state


def all_subtasks_completed(state: AgentState) -> bool:
    """检查所有子任务是否完成"""
    return state["current_subtask_index"] >= len(state["subtasks"])


if __name__ == "__main__":
    # 测试状态管理
    print("测试状态管理功能...")

    # 创建初始状态
    state = create_initial_state("证明：素数有无穷多个")
    print(f"初始状态: {state['status']}")
    print(f"查询: {state['query']}")
    print(f"需要重试: {should_retry(state)}")
    print(f"已完成: {is_completed(state)}")

    # 测试消息更新
    state = update_message(state, "assistant", "让我来思考一下...")
    print(f"消息数量: {len(state['messages'])}")

    # 测试错误处理
    state = increment_error(state, "模拟错误")
    state = increment_error(state, "又一个错误")
    print(f"错误次数: {state['error_count']}")
    print(f"最后错误: {state['last_error']}")

    state = reset_errors(state)
    print(f"重置后错误次数: {state['error_count']}")

    # 测试子任务管理
    state["subtasks"] = ["分析问题", "寻找反证", "得出结论"]
    print(f"当前子任务: {get_current_subtask(state)}")

    state = advance_subtask(state)
    print(f"推进后子任务: {get_current_subtask(state)}")

    state = advance_subtask(state)
    state = advance_subtask(state)
    print(f"所有任务完成: {all_subtasks_completed(state)}")

    print("\n状态管理测试完成!")
