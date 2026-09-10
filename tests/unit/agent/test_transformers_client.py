import asyncio

from adaptive_math.agent.model_client import ChatMessage, GenerationConfig
from adaptive_math.agent.transformers_client import TransformersModelClient


class FakeTokenizer:
    eos_token_id = 0
    chat_template = "fake-template"

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> list[int]:
        self.messages = messages
        return [10, 11]

    def decode(self, tokens: list[int], **kwargs: object) -> str:
        assert tokens == [20, 21]
        return '<final>{"answer":"2"}</final><|end|>'


class FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def generate(self, input_ids: list[int], **kwargs: object) -> list[list[int]]:
        self.kwargs = kwargs
        assert input_ids == [10, 11]
        return [[10, 11, 20, 21]]


def test_transformers_client_uses_chat_template_and_stops_on_configured_marker() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    client = TransformersModelClient(tokenizer, model, model_id="fake", stop_strings=("<|end|>",))

    turn = asyncio.run(
        client.generate(
            (ChatMessage(role="system", content="rules"), ChatMessage(role="user", content="problem")),
            GenerationConfig(max_new_tokens=8, seed=7),
        )
    )

    assert tokenizer.messages == [{"role": "system", "content": "rules"}, {"role": "user", "content": "problem"}]
    assert turn.text == '<final>{"answer":"2"}</final>'
    assert turn.prompt_tokens == 2 and turn.generated_tokens == 2
    assert model.kwargs is not None and model.kwargs["max_new_tokens"] == 8
