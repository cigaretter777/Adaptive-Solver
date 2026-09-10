import asyncio
import json

import httpx
import pytest

from adaptive_math.tools.base import ToolErrorCode
from adaptive_math.tools.sandboxfusion import SandboxFusionClient


def test_run_code_posts_only_code_and_language(respx_mock: object) -> None:
    route = respx_mock.post("http://sandbox.test/run_code").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "success",
                "execution_time": 0.12,
                "return_code": 0,
                "stdout": "42\n",
                "stderr": "",
            },
        )
    )
    client = SandboxFusionClient(base_url="http://sandbox.test")

    result = asyncio.run(client.run_code("print(42)"))

    assert route.called
    assert json.loads(route.calls[0].request.content) == {"code": "print(42)", "language": "python"}
    assert result.ok and result.output == "stdout:\n42\n"
    asyncio.run(client.aclose())


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(200, json={"status": "error", "execution_time": 0.1, "return_code": 1, "stdout": "", "stderr": "SyntaxError"}), ToolErrorCode.EXECUTION_ERROR),
        (httpx.Response(500), ToolErrorCode.UNAVAILABLE),
        (httpx.Response(200, json={"unexpected": "shape"}), ToolErrorCode.EXECUTION_ERROR),
    ],
)
def test_run_code_maps_remote_failures(
    respx_mock: object, response: httpx.Response, expected: ToolErrorCode
) -> None:
    respx_mock.post("http://sandbox.test/run_code").mock(return_value=response)
    client = SandboxFusionClient(base_url="http://sandbox.test")

    result = asyncio.run(client.run_code("raise ValueError()"))

    assert not result.ok
    assert result.error_code is expected
    asyncio.run(client.aclose())
