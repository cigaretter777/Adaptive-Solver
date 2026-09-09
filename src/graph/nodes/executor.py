"""
执行节点 (Executor)

负责：
1. 整合所有工具执行结果
2. 生成最终的回答
3. 格式化输出
4. 确保回答与用户查询一致
"""

from typing import Dict, Any, List, Optional
import json

from ..state import (
    AgentState,
    update_message,
    update_status,
    is_completed,
    is_failed
)


class Executor:
    """
    执行节点

    整合结果，生成最终答案
    """

    def __init__(self, llm_client=None):
        """
        初始化执行器

        Args:
            llm_client: LLM 客户端（用于生成自然语言答案）
        """
        self.llm_client = llm_client

    def __call__(self, state: AgentState) -> AgentState:
        """
        生成最终答案

        Args:
            state: 当前状态

        Returns:
            AgentState: 更新后的状态
        """
        # 检查状态是否允许执行
        if is_failed(state):
            return self._handle_failure(state)

        # 收集所有结果
        results_summary = self._collect_results(state)

        # 生成最终答案
        answer = self._generate_answer(state, results_summary)

        # 更新状态
        state["answer"] = answer
        state["status"] = "completed"

        # 记录执行消息
        state = update_message(state, "assistant", answer)

        # 更新元数据
        state["metadata"]["step_count"] = len(state["tool_calls"])
        state["metadata"]["completed"] = True

        return state

    def _handle_failure(self, state: AgentState) -> AgentState:
        """
        处理失败情况

        Args:
            state: 当前状态

        Returns:
            AgentState: 更新后的状态
        """
        error_msg = f"抱歉，无法完成您的请求。{state.get('last_error', '')}"
        state["answer"] = error_msg
        state["status"] = "failed"
        state = update_message(state, "assistant", error_msg)
        return state

    def _collect_results(self, state: AgentState) -> Dict[str, Any]:
        """
        收集所有执行结果

        Args:
            state: 当前状态

        Returns:
            Dict: 结果汇总
        """
        summary = {
            "query": state["query"],
            "plan": state["plan"],
            "subtasks": state["subtasks"],
            "tool_results": [],
            "total_tool_calls": len(state["tool_calls"]),
            "error_count": state["error_count"]
        }

        for i, result in enumerate(state["tool_results"]):
            summary["tool_results"].append({
                "step": i + 1,
                "task": result.get("task"),
                "tool": result.get("tool"),
                "result": str(result.get("result", ""))
            })

        return summary

    def _generate_answer(self, state: AgentState, results_summary: Dict[str, Any]) -> str:
        """
        生成最终答案

        Args:
            state: 当前状态
            results_summary: 结果汇总

        Returns:
            str: 最终答案
        """
        query = state["query"]
        plan = state["plan"]
        tool_results = results_summary["tool_results"]

        # 根据任务类型生成不同格式的答案
        task_type = state["metadata"].get("task_type", "general")

        if task_type == "proof":
            return self._generate_proof_answer(query, plan, tool_results)
        elif task_type == "calculation":
            return self._generate_calculation_answer(query, tool_results)
        elif task_type == "code":
            return self._generate_code_answer(query, plan, tool_results)
        else:
            return self._generate_general_answer(query, plan, tool_results)

    def _generate_proof_answer(self, query: str, plan: str,
                                tool_results: List[Dict]) -> str:
        """
        生成证明类答案

        Args:
            query: 用户查询
            plan: 执行计划
            tool_results: 工具结果

        Returns:
            str: 证明答案
        """
        answer = f"## 证明\n\n"
        answer += f"**命题**: {query.replace('证明：', '').replace('证明', '')}\n\n"

        # 提取证明内容
        proof_steps = []
        for result in tool_results:
            result_text = result.get("result", "")
            if "逻辑检查" in result_text or "证明" in result_text:
                proof_steps.append(result_text)

        if proof_steps:
            answer += "**证明过程**:\n\n"
            for i, step in enumerate(proof_steps, 1):
                answer += f"{i}. {step}\n"
        else:
            answer += "**证明过程**:\n\n"
            for i, result in enumerate(tool_results, 1):
                answer += f"{i}. {result.get('task', '步骤')}: {result.get('result', '')}\n"

        answer += f"\n**结论**: 根据上述推理，命题得证。"

        return answer

    def _generate_calculation_answer(self, query: str,
                                      tool_results: List[Dict]) -> str:
        """
        生成计算类答案

        Args:
            query: 用户查询
            tool_results: 工具结果

        Returns:
            str: 计算答案
        """
        answer = f"## 计算结果\n\n"

        # 查找最终计算结果
        final_result = None
        for result in tool_results:
            result_text = result.get("result", "")
            if "计算结果" in result_text or "和为" in result_text:
                final_result = result_text
                break

        if final_result:
            answer += f"**答案**: {final_result}\n\n"
        else:
            answer += f"**答案**: "
            for result in tool_results:
                answer += f"{result.get('result', '')} "
            answer += "\n\n"

        # 显示计算过程
        answer += "**计算过程**:\n\n"
        for i, result in enumerate(tool_results, 1):
            answer += f"{i}. {result.get('task', '')}: {result.get('result', '')}\n"

        return answer

    def _generate_code_answer(self, query: str, plan: str,
                               tool_results: List[Dict]) -> str:
        """
        生成代码类答案

        Args:
            query: 用户查询
            plan: 执行计划
            tool_results: 工具结果

        Returns:
            str: 代码答案
        """
        answer = f"## 代码实现\n\n"
        answer += f"**需求**: {query}\n\n"

        answer += "**实现方案**:\n\n"
        for i, step in enumerate(tool_results, 1):
            answer += f"{i}. {step.get('task', '')}: {step.get('result', '')}\n"

        answer += f"\n**注意**: 完整的代码实现需要根据具体需求进一步细化。"

        return answer

    def _generate_general_answer(self, query: str, plan: str,
                                  tool_results: List[Dict]) -> str:
        """
        生成通用答案

        Args:
            query: 用户查询
            plan: 执行计划
            tool_results: 工具结果

        Returns:
            str: 通用答案
        """
        answer = f"## 回答\n\n"
        answer += f"**问题**: {query}\n\n"

        if plan:
            answer += f"**分析**: {plan}\n\n"

        answer += "**执行结果**:\n\n"
        for i, result in enumerate(tool_results, 1):
            answer += f"{i}. {result.get('task', '')}\n"
            answer += f"   结果: {result.get('result', '')}\n\n"

        # 尝试提取最终结论
        if tool_results:
            last_result = tool_results[-1].get("result", "")
            answer += f"**结论**: {last_result}"

        return answer

    def format_answer(self, answer: str, format_type: str = "markdown") -> str:
        """
        格式化答案

        Args:
            answer: 原始答案
            format_type: 格式类型 (markdown, plain, json)

        Returns:
            str: 格式化后的答案
        """
        if format_type == "json":
            return json.dumps({"answer": answer}, ensure_ascii=False, indent=2)
        elif format_type == "plain":
            # 移除 Markdown 格式
            import re
            plain = re.sub(r'[#*`_\[\]]', '', answer)
            return plain
        else:
            return answer


# 便捷函数
def executor_node(state: AgentState) -> AgentState:
    """
    便捷的执行节点函数

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    executor = Executor()
    return executor(state)


if __name__ == "__main__":
    # 测试执行节点
    from ..state import create_initial_state
    from ..nodes.planner import planner_node
    from ..nodes.tool_caller import tool_caller_node
    from ..nodes.critic import critic_node

    print("测试执行节点...\n")

    # 创建测试状态
    state = create_initial_state("计算斐波那契数列前10项的和")

    # 执行完整流程
    state = planner_node(state)

    while not all_subtasks_completed(state):
        state = tool_caller_node(state)

    state = critic_node(state)
    state = executor_node(state)

    print("="*60)
    print("最终答案:")
    print("="*60)
    print(state["answer"])
    print("\n" + "="*60)
    print(f"状态: {state['status']}")
    print(f"工具调用次数: {state['metadata']['step_count']}")
