"""Source registry and raw record loaders.

The registry pins every dataset to an immutable revision, license, intended
use and citation. Loaders return raw records plus the resolved upstream
revision so the build manifest stays auditable.
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.core.types import AnswerType, JSONValue

LoaderName = Literal[
    "openr1", "numinamath_tir", "dapo", "math", "math500", "aime", "omnimath", "synthetic"
]


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    uri: str
    revision: str
    license: str
    intended_use: Literal["train", "eval"]
    citation: str
    loader: LoaderName
    loader_params: dict[str, JSONValue] = Field(default_factory=dict)
    answer_type: AnswerType


class SourceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    sources: dict[str, SourceSpec]

    def eval_source_names(self) -> set[str]:
        return {name for name, spec in self.sources.items() if spec.intended_use == "eval"}


def load_source_records(
    spec: SourceSpec, sample: int | None = None
) -> tuple[list[dict[str, Any]], str | None]:
    """Load raw records for one source; returns (records, resolved_revision)."""
    if spec.loader == "synthetic":
        return _load_synthetic(spec, sample), None
    dataset_id = str(spec.loader_params["dataset"])
    split = str(spec.loader_params.get("split", "train"))
    records, resolved = _load_hf(dataset_id, split, sample, spec.revision)
    return _adapt_records(spec, records), resolved


def _load_synthetic(spec: SourceSpec, sample: int | None) -> list[dict[str, Any]]:
    raw_path = spec.loader_params.get("path")
    path = Path(str(raw_path)) if raw_path else Path(spec.uri.removeprefix("file://"))
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if sample is not None:
        records = records[:sample]
    return records


def _load_hf(
    dataset_id: str, split: str, sample: int | None, revision: str
) -> tuple[list[dict[str, Any]], str | None]:
    from datasets import load_dataset

    split_spec = f"{split}[:{sample}]" if sample is not None else split
    dataset = load_dataset(path=dataset_id, split=split_spec, revision=revision)
    resolved: str | None = getattr(getattr(dataset, "info", None), "sha", None)
    return [_to_json(dict(row)) for row in dataset], resolved


def _adapt_records(spec: SourceSpec, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize source-specific layouts before generic canonicalization."""
    if spec.loader != "dapo":
        return records
    adapted: list[dict[str, Any]] = []
    for record in records:
        prompt = record.get("prompt")
        reward_model = record.get("reward_model")
        if not isinstance(prompt, list) or not isinstance(reward_model, dict):
            adapted.append({})
            continue
        messages = [message for message in prompt if isinstance(message, dict)]
        user_messages = [message for message in messages if message.get("role") == "user"]
        content = (user_messages or messages)[-1].get("content") if messages else None
        answer = reward_model.get("ground_truth")
        extra_info = record.get("extra_info")
        source_index = extra_info.get("index") if isinstance(extra_info, dict) else None
        adapted.append(
            {
                "problem": content,
                "answer": answer,
                "source_index": source_index,
            }
        )
    return adapted


def _to_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}
    try:
        return value.item()  # numpy scalars
    except AttributeError:
        return str(value)
