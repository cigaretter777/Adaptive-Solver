"""Deterministic padding for already-validated causal-LM trajectories."""

from adaptive_math.training.tokenization import IGNORE_INDEX, TokenizedTrajectory


class CausalLMCollator:
    def __init__(self, *, pad_token_id: int, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._pad_token_id = pad_token_id
        self._max_length = max_length

    def __call__(self, examples: list[TokenizedTrajectory]) -> dict[str, list[list[int]]]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        revisions = {example.tokenizer_revision for example in examples}
        if len(revisions) != 1:
            raise ValueError("tokenizer revision mismatch")
        longest = max(len(example.input_ids) for example in examples)
        if longest > self._max_length:
            raise ValueError("example exceeds max_length and must not be truncated")
        return {
            "input_ids": [_pad(example.input_ids, longest, self._pad_token_id) for example in examples],
            "attention_mask": [_pad(example.attention_mask, longest, 0) for example in examples],
            "labels": [_pad(example.labels, longest, IGNORE_INDEX) for example in examples],
            "assistant_mask": [_pad(example.assistant_mask, longest, 0) for example in examples],
        }


def _pad(values: tuple[int, ...], length: int, pad_value: int) -> list[int]:
    return [*values, *([pad_value] * (length - len(values)))]
