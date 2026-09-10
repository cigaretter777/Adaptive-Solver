import pytest

from adaptive_math.agent.model_client import ChatMessage
from adaptive_math.training.tokenization import tokenize_trajectory


class FakeTokenizer:
    """Every message contributes a distinct, deterministic two-token span."""

    revision = "fake-r1"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> list[int]:
        tokens: list[int] = []
        for index, _ in enumerate(messages):
            tokens.extend([index * 10 + 1, index * 10 + 2])
        return tokens


def test_only_assistant_spans_contribute_to_loss() -> None:
    messages = (
        ChatMessage(role="system", content="rules"),
        ChatMessage(role="user", content="problem"),
        ChatMessage(role="assistant", content='<tool_call>{"name":"sympy","arguments":{}}</tool_call>'),
        ChatMessage(role="tool", content="2"),
        ChatMessage(role="assistant", content='<final>{"answer":"2"}</final>'),
    )

    encoded = tokenize_trajectory(messages, FakeTokenizer(), max_length=32)

    assert encoded.input_ids == (1, 2, 11, 12, 21, 22, 31, 32, 41, 42)
    assert encoded.labels == (-100, -100, -100, -100, 21, 22, -100, -100, 41, 42)
    assert encoded.assistant_mask == (0, 0, 0, 0, 1, 1, 0, 0, 1, 1)
    assert encoded.tokenizer_revision == "fake-r1"


def test_tokenization_rejects_prefix_mismatch_and_overlength() -> None:
    class BrokenTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> list[int]:
            return [len(messages)]

    messages = (ChatMessage(role="system", content="rules"), ChatMessage(role="user", content="problem"))
    with pytest.raises(ValueError, match="prefix"):
        tokenize_trajectory(messages, BrokenTokenizer(), max_length=32)
    with pytest.raises(ValueError, match="max_length"):
        tokenize_trajectory(messages, FakeTokenizer(), max_length=2)
