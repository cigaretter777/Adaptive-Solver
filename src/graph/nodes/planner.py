"""
规划节点 (Planner)

负责：
1. 分析用户查询，理解任务目标
2. 将复杂任务分解为可执行的子任务
3. 制定执行计划
"""

from typing import Dict, Any, List, Optional
import re
import json

from ..state import (
    AgentState,
    update_message,
    update_status,
    has_plan,
    reset_errors
)


class Planner:
    """
    规划节点

    将复杂任务分解为子任务，制定执行计划
    """

    # 规划提示模板
    PLANNING_PROMPT = """你是一个任务规划专家。请分析用户的需求，将其分解为清晰的执行步骤。

用户查询: {query}

请按照以下格式输出计划：

## 总体目标
{goal}

## 执行步骤
1. [步骤1描述]
2. [步骤2描述]
3. [步骤3描述]
...

## 需要的工具
- [工具1]: [用途说明]
- [工具2]: [用途说明]

## 注意事项
- [注意事项1]
- [注意事项2]
"""

    # 任务类型识别模式
    TASK_PATTERNS = {
        "proof": [
            r"证明.*", r"证.*定理.*", r"推导.*"
        ],
        "calculation": [
            r"计算.*", r"\d+[\+\-\*\/]\d+.*", r"求.*"
        ],
        "analysis": [
            r"分析.*", r"评估.*", r"比较.*"
        ],
        "code": [
            r"实现.*", r"写.*代码.*", r"设计.*函数.*"
        ],
        "explain": [
            r"解释.*", r"说明.*", r"为什么.*"
        ]
    }

    def __init__(self, llm_client=None):
        """
        初始化规划器

        Args:
            llm_client: LLM 客户端（如果为 None，则使用基于规则的方法）
        """
        self.llm_client = llm_client

    def __call__(self, state: AgentState) -> AgentState:
        """
        执行规划

        Args:
            state: 当前状态

        Returns:
            AgentState: 更新后的状态
        """
        query = state["query"]

        # 如果已有计划且没有错误，跳过规划但确保状态正确
        if has_plan(state) and state["error_count"] == 0:
            state["status"] = "executing"
            return state

        # 重置错误（因为重新规划意味着新的尝试）
        state = reset_errors(state)

        # 分析任务类型
        task_type = self._classify_task_type(query)

        # 生成计划
        if self.llm_client:
            plan = self._generate_plan_with_llm(query, task_type)
        else:
            plan = self._generate_plan_with_rules(query, task_type)

        # 更新状态
        state["plan"] = plan["goal"]
        state["subtasks"] = plan["steps"]
        state["current_subtask_index"] = 0
        state["status"] = "executing"

        # 记录规划消息
        planning_message = self._format_planning_message(plan)
        state = update_message(state, "system", planning_message)

        # 存储元数据
        state["metadata"]["task_type"] = task_type
        state["metadata"]["plan_generated"] = True

        return state

    def _classify_task_type(self, query: str) -> str:
        """
        分类任务类型

        Args:
            query: 用户查询

        Returns:
            str: 任务类型
        """
        query_lower = query.lower()

        for task_type, patterns in self.TASK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return task_type

        return "general"

    def _generate_plan_with_llm(self, query: str, task_type: str) -> Dict[str, Any]:
        """
        使用 LLM 生成计划

        Args:
            query: 用户查询
            task_type: 任务类型

        Returns:
            Dict: 包含目标、步骤、工具的计划
        """
        prompt = self.PLANNING_PROMPT.format(query=query)

        # TODO: 调用 LLM API 生成计划
        # 这里是示例实现
        return self._generate_plan_with_rules(query, task_type)

    def _generate_plan_with_rules(self, query: str, task_type: str) -> Dict[str, Any]:
        """
        使用基于规则的方法生成计划

        Args:
            query: 用户查询
            task_type: 任务类型

        Returns:
            Dict: 包含目标、步骤、工具的计划
        """
        # 根据任务类型生成不同的计划模板
        plans = {
            "proof": {
                "goal": f"证明：{query.replace('证明：', '').replace('证明', '')}",
                "steps": [
                    "明确要证明的命题或定理",
                    "选择适当的证明方法（直接证明、反证法、归纳法等）",
                    "执行证明过程",
                    "验证证明的完整性和正确性",
                    "整理并呈现最终证明"
                ],
                "tools": ["逻辑推理", "数学公式验证"],
                "notes": ["注意每一步的逻辑严密性", "确保推理链条完整"]
            },
            "calculation": {
                "goal": f"计算：{query}",
                "steps": [
                    "识别计算类型和涉及的数据",
                    "应用相应的计算规则或公式",
                    "逐步执行计算",
                    "验证计算结果",
                    "给出最终答案"
                ],
                "tools": ["计算器", "公式库"],
                "notes": ["注意计算的精确性", "检查中间步骤"]
            },
            "analysis": {
                "goal": f"分析：{query.replace('分析：', '').replace('分析', '')}",
                "steps": [
                    "明确分析的对象和目标",
                    "收集相关信息和数据",
                    "识别关键因素和关系",
                    "进行系统性分析",
                    "得出结论并提出建议"
                ],
                "tools": ["数据分析", "逻辑推理"],
                "notes": ["保持客观的分析态度", "考虑多方面因素"]
            },
            "code": {
                "goal": f"实现：{query.replace('实现：', '').replace('实现', '')}",
                "steps": [
                    "理解需求，明确功能要求",
                    "设计算法和数据结构",
                    "编写核心代码",
                    "测试和调试",
                    "优化和完善"
                ],
                "tools": ["代码执行", "测试框架"],
                "notes": ["注意代码的可读性和可维护性", "考虑边界条件"]
            },
            "explain": {
                "goal": f"解释：{query.replace('解释：', '').replace('解释', '')}",
                "steps": [
                    "理解要解释的概念或问题",
                    "组织解释的逻辑结构",
                    "提供清晰的说明和示例",
                    "总结关键要点",
                    "回答可能的后续问题"
                ],
                "tools": ["知识检索", "类比推理"],
                "notes": ["使用简单易懂的语言", "提供具体例子"]
            },
            "general": {
                "goal": f"解决：{query}",
                "steps": [
                    "理解问题的核心",
                    "分析相关因素",
                    "制定解决方案",
                    "执行方案",
                    "验证结果"
                ],
                "tools": ["通用推理", "知识库"],
                "notes": ["确保理解准确", "考虑各种可能性"]
            }
        }

        return plans.get(task_type, plans["general"])

    def _format_planning_message(self, plan: Dict[str, Any]) -> str:
        """
        格式化规划消息

        Args:
            plan: 计划字典

        Returns:
            str: 格式化的消息
        """
        message = f"📋 执行计划\n\n"
        message += f"**目标**: {plan['goal']}\n\n"
        message += f"**执行步骤**:\n"
        for i, step in enumerate(plan['steps'], 1):
            message += f"{i}. {step}\n"
        return message


# 便捷函数
def planner_node(state: AgentState) -> AgentState:
    """
    便捷的规划节点函数

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    planner = Planner()
    return planner(state)


if __name__ == "__main__":
    # 测试规划节点
    from ..state import create_initial_state

    print("测试规划节点...\n")

    test_queries = [
        "证明：任意两个连续整数的乘积是偶数",
        "计算斐波那契数列前10项的和",
        "分析这个算法的时间复杂度",
        "实现一个二叉树的遍历算法"
    ]

    planner = Planner()

    for query in test_queries:
        print(f"{'='*60}")
        print(f"查询: {query}")
        print(f"{'='*60}")

        state = create_initial_state(query)
        state = planner(state)

        print(f"任务类型: {state['metadata']['task_type']}")
        print(f"目标: {state['plan']}")
        print(f"子任务数: {len(state['subtasks'])}")
        print(f"执行步骤:")
        for i, step in enumerate(state['subtasks'], 1):
            print(f"  {i}. {step}")
        print()
