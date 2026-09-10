"""Prompt construction from public state and live tool schemas only."""

import orjson

from adaptive_math.agent.model_client import ChatMessage
from adaptive_math.core.types import Budget, MathTask
from adaptive_math.tools.registry import ToolRegistry

PROMPT_VERSION = "agent-v1"


def render_initial_messages(task: MathTask, budget: Budget, registry: ToolRegistry) -> tuple[ChatMessage, ...]:
    tools = orjson.dumps(registry.descriptions(), option=orjson.OPT_SORT_KEYS).decode()
    system = (
        "You are a math-solving agent. Return exactly one action per turn: "
        "<tool_call>{\"name\":...,\"arguments\":{...}}</tool_call> or "
        "<final>{\"answer\":...}</final>. An optional <think>...</think> may precede it. "
        "Do not put prose outside these tags. "
        f"Tool schemas: {tools}"
    )
    user = (
        f"Problem: {task.problem}\nAnswer type: {task.answer_type.value}\n"
        f"Budget: steps={budget.max_steps}, tool_calls={budget.max_tool_calls}, "
        f"python_seconds={budget.max_python_seconds}."
    )
    return (ChatMessage(role="system", content=system), ChatMessage(role="user", content=user))


def render_observation(content: str) -> ChatMessage:
    return ChatMessage(role="tool", content=content)
