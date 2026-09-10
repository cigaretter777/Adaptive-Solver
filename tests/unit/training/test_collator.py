from adaptive_math.training.collator import CausalLMCollator
from adaptive_math.training.tokenization import TokenizedTrajectory


def test_collator_pads_inputs_and_ignores_padding_labels() -> None:
    collator = CausalLMCollator(pad_token_id=0, max_length=8)
    batch = collator(
        [
            TokenizedTrajectory(input_ids=(1, 2), attention_mask=(1, 1), labels=(-100, 2), assistant_mask=(0, 1), message_spans=(), tokenizer_revision="r"),
            TokenizedTrajectory(input_ids=(3,), attention_mask=(1,), labels=(3,), assistant_mask=(1,), message_spans=(), tokenizer_revision="r"),
        ]
    )

    assert batch["input_ids"] == [[1, 2], [3, 0]]
    assert batch["labels"] == [[-100, 2], [3, -100]]
    assert batch["attention_mask"] == [[1, 1], [1, 0]]
