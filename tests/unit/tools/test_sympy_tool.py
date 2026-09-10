import asyncio

from adaptive_math.tools.base import ToolContext
from adaptive_math.tools.sympy_tool import SympyArguments, SympyTool


def context() -> ToolContext:
    return ToolContext(trace_id="trace", task_id="task", remaining_observation_chars=1000)


def test_sympy_tool_factors_and_sorts_solutions_deterministically() -> None:
    tool = SympyTool()

    factor = asyncio.run(tool.execute(SympyArguments(operation="factor", expression="x^2 - 1"), context()))
    solve = asyncio.run(tool.execute(SympyArguments(operation="solve", expression="x^2 - 4", variables=["x"]), context()))

    assert factor.ok and factor.output == "(x - 1)*(x + 1)"
    assert solve.ok and solve.output == "[-2, 2]"


def test_sympy_tool_supports_calculus() -> None:
    tool = SympyTool()

    derivative = asyncio.run(tool.execute(SympyArguments(operation="diff", expression="sin(x)", variables=["x"]), context()))
    integral = asyncio.run(tool.execute(SympyArguments(operation="integrate", expression="x", variables=["x"], lower="0", upper="2"), context()))

    assert derivative.output == "cos(x)"
    assert integral.output == "2"
