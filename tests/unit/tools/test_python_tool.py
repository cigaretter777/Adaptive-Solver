import asyncio

from adaptive_math.tools.base import ToolContext, ToolErrorCode, ToolResult
from adaptive_math.tools.python_tool import PythonArguments, PythonTool


class RecordingSandbox:
    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.code: str | None = None

    async def run_code(self, code: str) -> ToolResult:
        self.code = code
        return self.result


def _context() -> ToolContext:
    return ToolContext(trace_id="trace", task_id="task", remaining_observation_chars=10)


def test_python_tool_delegates_only_code_to_sandbox_and_bounds_observation() -> None:
    sandbox = RecordingSandbox(ToolResult(ok=True, output="12345678901", latency_ms=7))
    tool = PythonTool(sandbox)

    result = asyncio.run(tool.execute(PythonArguments(code="print(6 * 7)"), _context()))

    assert sandbox.code == "print(6 * 7)"
    assert result.ok
    assert result.output == "1234567890"
    assert result.truncated


def test_python_tool_propagates_structured_sandbox_failure() -> None:
    tool = PythonTool(
        RecordingSandbox(
            ToolResult(ok=False, output="", error_code=ToolErrorCode.UNAVAILABLE, latency_ms=3)
        )
    )

    result = asyncio.run(tool.execute(PythonArguments(code="print(1)"), _context()))

    assert not result.ok
    assert result.error_code is ToolErrorCode.UNAVAILABLE
