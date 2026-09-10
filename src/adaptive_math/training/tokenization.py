"""Incremental chat-template tokenization with exact assistant-only masks."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.agent.model_client import ChatMessage

IGNORE_INDEX = -100


class MessageSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class TokenizedTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_mask: tuple[int, ...]
    message_spans: tuple[MessageSpan, ...]
    tokenizer_revision: str


def tokenize_trajectory(
    messages: tuple[ChatMessage, ...], tokenizer: Any, *, max_length: int
) -> TokenizedTrajectory:
    """Use exact chat-template prefix deltas; never infer spans from decoded text."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    revision = getattr(tokenizer, "revision", None)
    if not isinstance(revision, str) or not revision:
        raise ValueError("tokenizer revision is required")
    prefix: list[int] = []
    labels: list[int] = []
    mask: list[int] = []
    spans: list[MessageSpan] = []
    rendered: list[dict[str, str]] = []
    for message in messages:
        rendered.append(message.model_dump())
        current = _to_tokens(
            tokenizer.apply_chat_template(
                rendered, tokenize=True, add_generation_prompt=False, return_tensors=None
            )
        )
        if current[: len(prefix)] != prefix:
            raise ValueError("chat-template prefix does not align exactly")
        start = len(prefix)
        delta = current[start:]
        trainable = message.role == "assistant"
        labels.extend(delta if trainable else [IGNORE_INDEX] * len(delta))
        mask.extend([1 if trainable else 0] * len(delta))
        spans.append(MessageSpan(role=message.role, start=start, end=len(current)))
        prefix = current
    if len(prefix) > max_length:
        raise ValueError("trajectory exceeds max_length and must not be truncated")
    return TokenizedTrajectory(
        input_ids=tuple(prefix),
        attention_mask=tuple(1 for _ in prefix),
        labels=tuple(labels),
        assistant_mask=tuple(mask),
        message_spans=tuple(spans),
        tokenizer_revision=revision,
    )


def _to_tokens(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(token, int) for token in value):
        raise ValueError("chat template must return one sequence of integer token IDs")
    return value
