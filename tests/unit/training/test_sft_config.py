import pytest

from adaptive_math.training.config import SFTConfig


def test_sft_config_requires_immutable_revisions_and_valid_lora_settings() -> None:
    config = SFTConfig(
        model_id="Qwen/Qwen3-4B", model_revision="a" * 40, tokenizer_revision="b" * 40,
        data_manifest_sha256="c" * 64, output_dir="artifacts/sft", precision="bf16", max_length=8192,
        epochs=1, learning_rate=1e-5, warmup_ratio=0.03, weight_decay=0.0, lora_rank=64,
        lora_alpha=128, lora_dropout=0.05, lora_targets=("q_proj",), gradient_checkpointing=True,
        seed=20260910, logging_steps=10,
    )

    assert config.precision == "bf16"
    with pytest.raises(ValueError, match="revision"):
        config.model_copy(update={"model_revision": "main"}).validated()
