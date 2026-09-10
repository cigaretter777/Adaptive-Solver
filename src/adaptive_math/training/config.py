"""Immutable configuration contracts used before cloud training starts."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


class SFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    model_revision: str
    tokenizer_revision: str
    data_manifest_sha256: str
    output_dir: str
    precision: Literal["bf16"]
    max_length: int = Field(gt=0, le=8192)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    warmup_ratio: float = Field(ge=0, le=1)
    weight_decay: float = Field(ge=0)
    lora_rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(ge=0, lt=1)
    lora_targets: tuple[str, ...] = Field(min_length=1)
    gradient_checkpointing: bool
    seed: int
    logging_steps: int = Field(gt=0)
    # Additive fields (master contract allows adding, never renaming):
    # effective batch = per_device_batch_size * gradient_accumulation_steps
    # * world_size, recorded in every resolved_config.yaml.
    per_device_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)

    @field_validator("model_revision", "tokenizer_revision")
    @classmethod
    def _immutable_revision(cls, value: str) -> str:
        if _SHA40.fullmatch(value) is None:
            raise ValueError("revision must be a lowercase 40-character commit SHA")
        return value

    @field_validator("data_manifest_sha256")
    @classmethod
    def _manifest_hash(cls, value: str) -> str:
        if _SHA64.fullmatch(value) is None:
            raise ValueError("data manifest hash must be a lowercase 64-character SHA-256")
        return value

    def validated(self) -> "SFTConfig":
        return SFTConfig.model_validate(self.model_dump())
