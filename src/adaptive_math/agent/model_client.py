"""Provider-independent async model boundary used by runtime and tests."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int = Field(default=1024, gt=0, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None


class ModelTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    prompt_tokens: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    finish_reason: str
    model_id: str


class ModelClient(Protocol):
    async def generate(
        self, messages: tuple[ChatMessage, ...], config: GenerationConfig
    ) -> ModelTurn: ...
