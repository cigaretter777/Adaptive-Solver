"""Small, fail-closed client for the external Python execution service."""

import os
import time
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from adaptive_math.tools.base import ToolErrorCode, ToolResult


class SandboxRunRequest(BaseModel):
    """The intentionally narrow request accepted by SandboxFusion."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=20_000)
    language: Literal["python"] = "python"


class SandboxRunResult(BaseModel):
    """Execution payload nested under SandboxFusion's ``run_result`` key."""

    model_config = ConfigDict(extra="forbid")

    status: str
    execution_time: float = Field(ge=0)
    return_code: int
    stdout: str
    stderr: str


class SandboxRunResponse(BaseModel):
    """The response envelope returned by SandboxFusion's ``/run_code`` API."""

    model_config = ConfigDict(extra="ignore")

    status: str
    run_result: SandboxRunResult | None = None


class SandboxFusionClient:
    """Execute code remotely; errors never trigger a local fallback."""

    def __init__(self, base_url: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        resolved_url = base_url or os.environ.get("ADAPTIVE_MATH_SANDBOX_URL", "http://127.0.0.1:8080")
        self._client = client or httpx.AsyncClient(
            base_url=resolved_url.rstrip("/"), timeout=httpx.Timeout(6.0, connect=1.0)
        )
        self._owns_client = client is None

    async def run_code(self, code: str) -> ToolResult:
        request = SandboxRunRequest(code=code)
        started = time.perf_counter()
        try:
            response = await self._client.post("/run_code", json=request.model_dump())
        except httpx.TimeoutException:
            return ToolResult(
                ok=False,
                output="",
                error_code=ToolErrorCode.TIMEOUT,
                latency_ms=_elapsed_ms(started),
            )
        except httpx.HTTPError:
            return ToolResult(
                ok=False,
                output="",
                error_code=ToolErrorCode.UNAVAILABLE,
                latency_ms=_elapsed_ms(started),
            )

        if response.status_code >= 500:
            return ToolResult(
                ok=False,
                output="",
                error_code=ToolErrorCode.UNAVAILABLE,
                latency_ms=_elapsed_ms(started),
            )
        try:
            payload = SandboxRunResponse.model_validate(response.json())
        except (ValidationError, ValueError):
            return ToolResult(
                ok=False,
                output="",
                error_code=ToolErrorCode.EXECUTION_ERROR,
                latency_ms=_elapsed_ms(started),
            )

        if payload.run_result is None:
            return ToolResult(
                ok=False,
                output="",
                error_code=ToolErrorCode.EXECUTION_ERROR,
                latency_ms=_elapsed_ms(started),
            )
        run_result = payload.run_result
        output = _format_output(run_result.stdout, run_result.stderr)
        if payload.status == "Success" and run_result.status == "Finished" and run_result.return_code == 0:
            return ToolResult(
                ok=True,
                output=output,
                latency_ms=_elapsed_ms(started),
                metadata={"execution_time": run_result.execution_time},
            )
        return ToolResult(
            ok=False,
            output=output,
            error_code=ToolErrorCode.EXECUTION_ERROR,
            latency_ms=_elapsed_ms(started),
            metadata={"execution_time": run_result.execution_time},
        )

    async def health(self) -> bool:
        try:
            response = await self._client.get("/health")
        except httpx.HTTPError:
            return False
        return response.is_success

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _format_output(stdout: str, stderr: str) -> str:
    sections: list[str] = []
    if stdout:
        sections.append(f"stdout:\n{stdout}")
    if stderr:
        sections.append(f"stderr:\n{stderr}")
    return "\n".join(sections)


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))
