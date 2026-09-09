"""
反思节点 (Critic)

负责：
1. 检查工具执行结果的质量
2. 识别逻辑错误或不一致
3. 提供改进建议
4. 决定是否需要回退重试
"""

from typing import Dict, Any, List, Optional, Tuple
import re

from ..state import (
    AgentState,
    update_message,
    update_status,
    all_subtasks_completed,
    should_retry
)


class Critic:
    """
    反思节点

    检查结果质量，决定是否需要回退
    """

    # 需要回退的错误模式
    ERROR_PATTERNS = [
        r"失败", r"错误", r"异常",
        r"fail", r"error", r"exception", r"wrong",
        r"无法", r"不能", r"不支持",
        r"不存在", r"无效"
    ]

    # 需要改进的警告模式
    WARNING_PATTERNS = [
        r"可能", r"也许", r"不确定", r"需要验证",
        r"might", r"maybe", r"uncertain", r"verify"
    ]

    # 成功的指示模式
    SUCCESS_PATTERNS = [
        r"完成", r"成功", r"正确", r"通过",
        r"complete", r"success", r"correct", r"pass",
        r"证明", r"推导", r"结论"
    ]

    def __init__(self, llm_client=None, strict_mode: bool = False):
        """
        初始化反思器

        Args:
            llm_client: LLM 客户端
            strict_mode: 严格模式（True 时更容易判定为错误）
        """
        self.llm_client = llm_client
        self.strict_mode = strict_mode

    def __call__(self, state: AgentState) -> AgentState:
        """
        执行反思检查

        Args:
            state: 当前状态

        Returns:
            AgentState: 更新后的状态
        """
        # 检查是否所有子任务已完成
        if all_subtasks_completed(state):
            # 进行最终检查
            approval, criticism, suggestions = self._review_final_result(state)
        else:
            # 检查中间结果
            approval, criticism, suggestions = self._review_intermediate_result(state)

        # 更新状态
        state["criticism"] = criticism
        state["suggestions"] = suggestions

        # 记录反思消息
        critic_message = self._format_critic_message(approval, criticism, suggestions)
        state = update_message(state, "critic", critic_message)

        # 根据检查结果决定下一步
        if approval == "approved":
            # 检查通过，继续执行或完成
            if all_subtasks_completed(state):
                state["status"] = "reviewing"
            else:
                state["status"] = "executing"
        elif approval == "needs_improvement":
            # 需要改进，给出建议但继续
            state["status"] = "executing"
        else:  # rejected
            # 检查失败，需要回退
            if should_retry(state):
                state["status"] = "planning"  # 回退到规划阶段
            else:
                state["status"] = "failed"  # 超过重试次数

        return state

    def _review_intermediate_result(self, state: AgentState) -> Tuple[str, Optional[str], List[str]]:
        """
        审查中间结果

        Args:
            state: 当前状态

        Returns:
            Tuple[str, Optional[str], List[str]]: (审批结果, 批评, 建议)
        """
        # 检查最新的工具结果
        if not state["tool_results"]:
            return "needs_improvement", "没有可用的工具执行结果", []

        latest_result = state["tool_results"][-1]
        result_text = str(latest_result.get("result", ""))

        # 检查错误模式
        if self._contains_error(result_text):
            return "rejected", "工具执行结果包含错误，需要重新执行", [
                "检查工具输入是否正确",
                "尝试使用其他工具",
                "重新分析任务需求"
            ]

        # 检查警告模式
        if self._contains_warning(result_text):
            return "needs_improvement", "结果存在不确定性，建议验证", [
                "验证计算结果",
                "检查逻辑推理",
                "确认答案的正确性"
            ]

        # 结果看起来正常
        return "approved", None, []

    def _review_final_result(self, state: AgentState) -> Tuple[str, Optional[str], List[str]]:
        """
        审查最终结果

        Args:
            state: 当前状态

        Returns:
            Tuple[str, Optional[str], List[str]]: (审批结果, 批评, 建议)
        """
        # 收集所有工具结果
        all_results = [str(r.get("result", "")) for r in state["tool_results"]]
        combined_results = " ".join(all_results)

        # 检查是否有错误
        if any(self._contains_error(r) for r in all_results):
            return "rejected", "执行过程中存在错误，最终结果不可靠", [
                "重新执行失败的步骤",
                "检查工具配置",
                "验证输入数据"
            ]

        # 检查结果完整性
        if not self._check_completeness(state):
            return "needs_improvement", "结果可能不完整", [
                "确认所有子任务都已完成",
                "检查是否遗漏了关键信息",
                "补充缺失的分析"
            ]

        # 检查结果一致性
        if not self._check_consistency(state):
            return "needs_improvement", "结果可能存在逻辑不一致", [
                "检查各步骤之间的逻辑关系",
                "验证推理链条",
                "确保结论与前提一致"
            ]

        # 所有检查通过
        return "approved", "所有检查通过，结果可靠", []

    def _contains_error(self, text: str) -> bool:
        """
        检查文本是否包含错误

        Args:
            text: 要检查的文本

        Returns:
            bool: 是否包含错误
        """
        text_lower = text.lower()
        for pattern in self.ERROR_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def _contains_warning(self, text: str) -> bool:
        """
        检查文本是否包含警告

        Args:
            text: 要检查的文本

        Returns:
            bool: 是否包含警告
        """
        text_lower = text.lower()
        for pattern in self.WARNING_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def _contains_success(self, text: str) -> bool:
        """
        检查文本是否包含成功指示

        Args:
            text: 要检查的文本

        Returns:
            bool: 是否包含成功指示
        """
        text_lower = text.lower()
        for pattern in self.SUCCESS_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def _check_completeness(self, state: AgentState) -> bool:
        """
        检查结果完整性

        Args:
            state: 当前状态

        Returns:
            bool: 结果是否完整
        """
        # 确保所有子任务都有结果
        if len(state["tool_results"]) < len(state["subtasks"]):
            return False

        # 确保没有空的结果
        for result in state["tool_results"]:
            if not result.get("result"):
                return False

        return True

    def _check_consistency(self, state: AgentState) -> bool:
        """
        检查结果一致性

        Args:
            state: 当前状态

        Returns:
            bool: 结果是否一致
        """
        # 简单的检查：不应该同时有成功和失败的结果
        has_success = any(
            self._contains_success(str(r.get("result", "")))
            for r in state["tool_results"]
        )
        has_error = any(
            self._contains_error(str(r.get("result", "")))
            for r in state["tool_results"]
        )

        return not (has_success and has_error)

    def _format_critic_message(self, approval: str, criticism: Optional[str],
                               suggestions: List[str]) -> str:
        """
        格式化反思消息

        Args:
            approval: 审批结果
            criticism: 批评内容
            suggestions: 建议列表

        Returns:
            str: 格式化的消息
        """
        message = "🔍 反思检查\n\n"

        # 审批结果
        approval_emoji = {
            "approved": "✅",
            "needs_improvement": "⚠️",
            "rejected": "❌"
        }
        message += f"**审批结果**: {approval_emoji.get(approval, '?')} {approval}\n"

        # 批评
        if criticism:
            message += f"\n**问题**: {criticism}\n"

        # 建议
        if suggestions:
            message += "\n**建议**:\n"
            for i, suggestion in enumerate(suggestions, 1):
                message += f"{i}. {suggestion}\n"

        return message


# 便捷函数
def critic_node(state: AgentState) -> AgentState:
    """
    便捷的反思节点函数

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    critic = Critic()
    return critic(state)


if __name__ == "__main__":
    # 测试反思节点
    from ..state import create_initial_state
    from ..nodes.planner import planner_node
    from ..nodes.tool_caller import tool_caller_node

    print("测试反思节点...\n")

    # 测试场景1：正常情况
    print("="*60)
    print("场景1: 正常完成")
    print("="*60)

    state = create_initial_state("计算斐波那契数列前10项的和")
    state = planner_node(state)

    # 执行所有子任务
    while not all_subtasks_completed(state):
        state = tool_caller_node(state)

    # 反思检查
    state = critic_node(state)
    print(f"审批结果: {state['criticism'] if state['criticism'] else '通过'}")
    print(f"状态: {state['status']}\n")

    # 测试场景2：错误情况
    print("="*60)
    print("场景2: 包含错误")
    print("="*60)

    state = create_initial_state("一个模拟错误的任务")
    state = planner_node(state)

    # 模拟错误结果
    state["tool_results"].append({
        "task": "测试任务",
        "result": "执行失败：无法找到数据",
        "tool": "test_tool"
    })

    state = critic_node(state)
    print(f"审批结果: {state['criticism']}")
    print(f"状态: {state['status']}")
    print(f"建议: {state['suggestions']}")
