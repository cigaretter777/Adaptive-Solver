"""
状态机工作流定义

基于 LangGraph 构建可回退的 Agent 工作流，包含：
- Planner: 规划节点
- Tool_Caller: 工具调用节点
- Critic: 反思节点
- Executor: 执行节点

工作流设计:
    Start -> Planner -> Tool_Caller -> Critic -> (检查结果)
                                   |            |
                                   v            v
                              Tool_Caller <---(需要改进)
                                   |
                                   v
                              Critic <----(拒绝/回退)
                                   |
                                   v
                              Planner
                                   |
                                   v
                              Tool_Caller (重试)
                                   |
                                   v
                              Critic -> (通过) -> Executor -> End
"""

# 处理直接运行时的导入问题
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    # 切换到绝对导入
    from src.graph.state import AgentState, create_initial_state, should_retry, all_subtasks_completed, is_failed, is_completed
    from src.graph.nodes.planner import Planner
    from src.graph.nodes.tool_caller import ToolCaller
    from src.graph.nodes.critic import Critic
    from src.graph.nodes.executor import Executor
else:
    # 使用相对导入
    from .state import AgentState, create_initial_state, should_retry, all_subtasks_completed, is_failed, is_completed
    from .nodes.planner import Planner
    from .nodes.tool_caller import ToolCaller
    from .nodes.critic import Critic
    from .nodes.executor import Executor

from typing import Dict, Any
from enum import Enum

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("警告: LangGraph 未安装，将使用简化版工作流")


class WorkflowStatus(Enum):
    """工作流状态"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"


class Workflow:
    """
    状态机工作流

    整合所有节点，定义状态流转逻辑
    """

    def __init__(self, llm_client=None, enable_langgraph: bool = True):
        """
        初始化工作流

        Args:
            llm_client: LLM 客户端
            enable_langgraph: 是否使用 LangGraph
        """
        self.llm_client = llm_client
        self.enable_langgraph = enable_langgraph and LANGGRAPH_AVAILABLE

        # 初始化节点
        self.planner = Planner(llm_client)
        self.tool_caller = ToolCaller(llm_client)
        self.critic = Critic(llm_client)
        self.executor = Executor(llm_client)

        # 如果使用 LangGraph，编译状态图
        if self.enable_langgraph:
            self.graph = self._build_langgraph_workflow()
        else:
            self.graph = None

    def _build_langgraph_workflow(self) -> Any:
        """
        构建 LangGraph 状态图

        Returns:
            编译后的状态图
        """
        # 创建状态图
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("planner", self.planner)
        workflow.add_node("tool_caller", self.tool_caller)
        workflow.add_node("critic", self.critic)
        workflow.add_node("executor", self.executor)

        # 添加边（定义状态流转）
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "tool_caller")

        # 添加条件边（Critic 节点的决策）
        workflow.add_conditional_edges(
            "critic",
            self._critic_decision,
            {
                "approved": "executor",
                "needs_improvement": "tool_caller",
                "rejected": "planner"
            }
        )

        workflow.add_edge("executor", END)

        # 编译图
        return workflow.compile()

    def _critic_decision(self, state: AgentState) -> str:
        """
        Critic 节点的决策逻辑

        Args:
            state: 当前状态

        Returns:
            str: 下一个节点名称
        """
        status = state.get("status", "")
        error_count = state.get("error_count", 0)
        max_retries = state.get("max_retries", 3)

        # 检查是否超过最大重试次数
        if error_count >= max_retries:
            return "approved"  # 超过重试次数，直接继续

        # 根据状态决定下一步
        if status == "completed":
            return "approved"
        elif status == "failed":
            return "approved"  # 失败也继续到 executor，让 executor 处理
        elif status == "reviewing":
            return "approved"
        elif status == "executing":
            # 检查是否有批评
            if state.get("criticism"):
                # 如果有错误且可以重试，回退到 planner
                if "错误" in state["criticism"] and should_retry(state):
                    return "rejected"
                # 否则返回 tool_caller 改进
                return "needs_improvement"
            return "approved"
        else:
            return "approved"

    def run(self, query: str, max_retries: int = 3,
            verbose: bool = True) -> Dict[str, Any]:
        """
        运行工作流

        Args:
            query: 用户查询
            max_retries: 最大重试次数
            verbose: 是否输出详细日志

        Returns:
            Dict: 执行结果
        """
        if self.enable_langgraph:
            return self._run_langgraph(query, max_retries, verbose)
        else:
            return self._run_simple(query, max_retries, verbose)

    def _run_langgraph(self, query: str, max_retries: int,
                       verbose: bool) -> Dict[str, Any]:
        """
        使用 LangGraph 运行工作流

        Args:
            query: 用户查询
            max_retries: 最大重试次数
            verbose: 是否输出详细日志

        Returns:
            Dict: 执行结果
        """
        # 创建初始状态
        initial_state = create_initial_state(query, max_retries)

        if verbose:
            print(f"\n{'='*60}")
            print(f"开始执行工作流（LangGraph 模式）")
            print(f"查询: {query}")
            print(f"{'='*60}\n")

        # 运行状态图
        final_state = None
        step_count = 0

        for event in self.graph.stream(initial_state, stream_mode="values"):
            step_count += 1
            final_state = event
            status = event.get("status", "unknown")

            if verbose:
                print(f"[步骤 {step_count}] 当前状态: {status}")
                if event.get("plan"):
                    print(f"  计划: {event['plan'][:50]}...")
                if event.get("tool_calls"):
                    print(f"  工具调用: {len(event['tool_calls'])} 次")
                if event.get("criticism"):
                    print(f"  批评: {event['criticism'][:50]}...")

        return {
            "answer": final_state.get("answer", "未生成答案"),
            "status": final_state.get("status", "unknown"),
            "step_count": step_count,
            "tool_calls": final_state.get("tool_calls", []),
            "metadata": final_state.get("metadata", {})
        }

    def _run_simple(self, query: str, max_retries: int,
                    verbose: bool) -> Dict[str, Any]:
        """
        使用简化版工作流（不依赖 LangGraph）

        Args:
            query: 用户查询
            max_retries: 最大重试次数
            verbose: 是否输出详细日志

        Returns:
            Dict: 执行结果
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"开始执行工作流（简化模式）")
            print(f"查询: {query}")
            print(f"{'='*60}\n")

        # 创建初始状态
        state = create_initial_state(query, max_retries)
        step_count = 0
        max_steps = 20  # 防止无限循环

        while step_count < max_steps and not is_completed(state) and not is_failed(state):
            step_count += 1

            # 根据状态执行相应节点
            if state["status"] == "planning":
                state = self.planner(state)
                if verbose:
                    print(f"[步骤 {step_count}] 规划完成")

            elif state["status"] == "executing":
                if not all_subtasks_completed(state):
                    state = self.tool_caller(state)
                    if verbose:
                        print(f"[步骤 {step_count}] 工具调用: {state['tool_calls'][-1] if state['tool_calls'] else 'None'}")
                else:
                    # 所有子任务完成，进入审查
                    state = self.critic(state)
                    if verbose:
                        print(f"[步骤 {step_count}] 审查完成: {state.get('criticism', '通过')}")

            elif state["status"] == "reviewing":
                state = self.executor(state)
                if verbose:
                    print(f"[步骤 {step_count}] 执行完成")

            elif state["status"] == "failed":
                state = self.executor(state)
                if verbose:
                    print(f"[步骤 {step_count}] 处理失败")

            # 安全检查
            if step_count > max_steps:
                state["status"] = "failed"
                state["last_error"] = "超过最大步骤数"
                break

        return {
            "answer": state.get("answer", "未生成答案"),
            "status": state.get("status", "unknown"),
            "step_count": step_count,
            "tool_calls": state.get("tool_calls", []),
            "metadata": state.get("metadata", {})
        }

    def run_stream(self, query: str, max_retries: int = 3):
        """
        流式运行工作流（生成器）

        Args:
            query: 用户查询
            max_retries: 最大重试次数

        Yields:
            Dict: 每个步骤的中间结果
        """
        state = create_initial_state(query, max_retries)
        step_count = 0
        max_steps = 20

        while step_count < max_steps and not is_completed(state) and not is_failed(state):
            step_count += 1

            if state["status"] == "planning":
                state = self.planner(state)
            elif state["status"] == "executing":
                if not all_subtasks_completed(state):
                    state = self.tool_caller(state)
                else:
                    state = self.critic(state)
            elif state["status"] == "reviewing":
                state = self.executor(state)
            elif state["status"] == "failed":
                state = self.executor(state)

            yield {
                "step": step_count,
                "status": state["status"],
                "current_subtask": state["subtasks"][state["current_subtask_index"]]
                if state["current_subtask_index"] < len(state["subtasks"]) else None,
                "partial_result": state.get("tool_results", [])[-1] if state.get("tool_results") else None
            }


# 便捷函数
def run_workflow(query: str, llm_client=None, max_retries: int = 3,
                 verbose: bool = True) -> Dict[str, Any]:
    """
    便捷的工作流运行函数

    Args:
        query: 用户查询
        llm_client: LLM 客户端
        max_retries: 最大重试次数
        verbose: 是否输出详细日志

    Returns:
        Dict: 执行结果
    """
    workflow = Workflow(llm_client=llm_client)
    return workflow.run(query, max_retries=max_retries, verbose=verbose)


if __name__ == "__main__":
    # 测试工作流
    print("测试状态机工作流...\n")

    test_queries = [
        "计算斐波那契数列前10项的和",
        "证明：任意两个连续整数的乘积是偶数",
        "分析1, 2, 3, 4, 5这组数据"
    ]

    workflow = Workflow()

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"测试查询: {query}")
        print(f"{'='*70}")

        result = workflow.run(query, verbose=True)

        print(f"\n{'='*70}")
        print(f"最终结果:")
        print(f"{'='*70}")
        print(result["answer"])
        print(f"\n状态: {result['status']}")
        print(f"步骤数: {result['step_count']}")
        print(f"工具调用次数: {len(result['tool_calls'])}")
