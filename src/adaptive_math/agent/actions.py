"""Typed, non-executable actions emitted by one assistant turn."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.core.types import JSONValue


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    arguments: dict[str, JSONValue]


class ToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool_call"] = "tool_call"
    call: ToolCall


class FinalAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["final"] = "final"
    answer: str = Field(min_length=1, max_length=8192)


AgentAction = Annotated[ToolAction | FinalAction, Field(discriminator="kind")]
