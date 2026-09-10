"""Typed tool boundary for the math agent runtime."""

from adaptive_math.tools.base import ToolContext, ToolErrorCode, ToolExecutionError, ToolResult
from adaptive_math.tools.registry import ToolRegistry

__all__ = ["ToolContext", "ToolErrorCode", "ToolExecutionError", "ToolRegistry", "ToolResult"]
