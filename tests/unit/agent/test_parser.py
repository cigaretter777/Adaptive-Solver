import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from adaptive_math.agent.actions import FinalAction, ToolAction
from adaptive_math.agent.parser import ParseErrorCode, parse_action


@pytest.mark.parametrize(
    ("raw", "action_type", "reasoning"),
    [
        ('<final>{"answer":"42"}</final>', FinalAction, None),
        (
            '<think>Use symbolic simplification.</think><tool_call>{"name":"sympy","arguments":{"expression":"x**2 - 1"}}</tool_call>',
            ToolAction,
            "Use symbolic simplification.",
        ),
        (
            '<tool_call>{"name":"python","arguments":{"code":"print(1 + 2)\\nprint(3)"}}</tool_call>',
            ToolAction,
            None,
        ),
    ],
)
def test_parse_action_accepts_exactly_one_valid_action(
    raw: str, action_type: type[FinalAction] | type[ToolAction], reasoning: str | None
) -> None:
    result = parse_action(raw)

    assert isinstance(result.action, action_type)
    assert result.reasoning == reasoning
    assert result.error is None
    assert result.raw_hash == hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("", ParseErrorCode.EMPTY),
        ("explain <final>{\"answer\":\"1\"}</final>", ParseErrorCode.INVALID_ENVELOPE),
        ("<think>x</think><think>y</think><final>{\"answer\":\"1\"}</final>", ParseErrorCode.INVALID_ENVELOPE),
        ("<final>{\"answer\":\"1\"}</final><think>x</think>", ParseErrorCode.INVALID_ENVELOPE),
        ("<final>{\"answer\":\"1\"}</final><final>{\"answer\":\"2\"}</final>", ParseErrorCode.MULTIPLE_ACTIONS),
        ("<tool_call>{\"name\":\"python\",\"arguments\":[]}</tool_call>", ParseErrorCode.INVALID_SCHEMA),
        ("<final>{not json}</final>", ParseErrorCode.INVALID_JSON),
        ("<unknown>{}</unknown>", ParseErrorCode.INVALID_ENVELOPE),
        ("<final>{\"answer\":\"1\"}</final> trailing", ParseErrorCode.INVALID_ENVELOPE),
        ("<final>{\"answer\":\"1\"}", ParseErrorCode.INVALID_ENVELOPE),
        ("x" * 32769, ParseErrorCode.TOO_LONG),
    ],
)
def test_parse_action_rejects_invalid_envelopes(raw: str, error: ParseErrorCode) -> None:
    result = parse_action(raw)

    assert result.action is None
    assert result.error == error


@settings(max_examples=100)
@given(st.text(max_size=40000))
def test_parse_action_never_raises_for_arbitrary_text(raw: str) -> None:
    result = parse_action(raw)

    assert result.raw_hash == hashlib.sha256(raw.encode()).hexdigest()
