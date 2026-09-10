"""Bounded parser for one optional think block and one top-level action."""

import hashlib
import re
from enum import StrEnum

import orjson
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from adaptive_math.agent.actions import AgentAction
from adaptive_math.core.types import JSONValue

MAX_RAW_CHARS = 32768
MAX_REASONING_CHARS = 24576
_ACTION_ADAPTER: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)
_ENVELOPE = re.compile(
    r"(?:<think>(?P<think>(?:(?!<think>|</think>).)*)</think>)?"
    r"(?:<tool_call>(?P<tool>.*?)</tool_call>|<final>(?P<final>.*?)</final>)",
    re.DOTALL,
)


class ParseErrorCode(StrEnum):
    EMPTY = "empty"
    TOO_LONG = "too_long"
    INVALID_ENVELOPE = "invalid_envelope"
    MULTIPLE_ACTIONS = "multiple_actions"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"


class ParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str | None
    action: AgentAction | None
    error: ParseErrorCode | None
    raw_hash: str
    details: dict[str, JSONValue] = Field(default_factory=dict)


def parse_action(raw: str) -> ParseResult:
    """Parse a protocol turn without interpreting or executing its payload."""
    raw_hash = hashlib.sha256(raw.encode()).hexdigest()
    text = raw.strip()
    if not text:
        return _failure(raw_hash, ParseErrorCode.EMPTY)
    if len(text) > MAX_RAW_CHARS:
        return _failure(raw_hash, ParseErrorCode.TOO_LONG)
    action_count = text.count("<tool_call>") + text.count("<final>")
    if action_count > 1:
        return _failure(raw_hash, ParseErrorCode.MULTIPLE_ACTIONS)
    match = _ENVELOPE.fullmatch(text)
    if match is None:
        return _failure(raw_hash, ParseErrorCode.INVALID_ENVELOPE)
    reasoning = match.group("think")
    if reasoning is not None and len(reasoning) > MAX_REASONING_CHARS:
        return _failure(raw_hash, ParseErrorCode.INVALID_SCHEMA)
    payload_text = match.group("tool") or match.group("final")
    try:
        payload = orjson.loads(payload_text)
    except orjson.JSONDecodeError:
        return _failure(raw_hash, ParseErrorCode.INVALID_JSON)
    if not isinstance(payload, dict):
        return _failure(raw_hash, ParseErrorCode.INVALID_SCHEMA)
    candidate: dict[str, object]
    if match.group("tool") is not None:
        candidate = {"kind": "tool_call", "call": payload}
    else:
        candidate = {"kind": "final", **payload}
    try:
        action = _ACTION_ADAPTER.validate_python(candidate)
    except ValueError:
        return _failure(raw_hash, ParseErrorCode.INVALID_SCHEMA)
    return ParseResult(reasoning=reasoning, action=action, error=None, raw_hash=raw_hash)


def _failure(raw_hash: str, error: ParseErrorCode) -> ParseResult:
    return ParseResult(reasoning=None, action=None, error=error, raw_hash=raw_hash)
