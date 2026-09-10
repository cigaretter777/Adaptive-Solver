import asyncio

import pytest

from adaptive_math.tools.base import ToolContext
from adaptive_math.tools.sympy_tool import SympyArguments, SympyTool


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "x.__class__", "[x for x in range(3)]", "sin(system(x))", "x^1000000", "(" * 65 + "x" + ")" * 65],
)
def test_sympy_tool_rejects_unsafe_or_excessive_input(expression: str) -> None:
    result = asyncio.run(
        SympyTool().execute(
            SympyArguments(operation="simplify", expression=expression),
            ToolContext(trace_id="trace", task_id="task", remaining_observation_chars=1000),
        )
    )

    assert not result.ok
    assert result.error_code is not None
