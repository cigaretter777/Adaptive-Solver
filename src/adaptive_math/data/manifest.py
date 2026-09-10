"""Build manifest: the auditable record of one dataset build."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    uri: str
    revision: str
    resolved_revision: str | None
    license: str
    intended_use: str
    citation: str
    answer_type: str
    loaded_count: int
    kept_count: int
    quarantined_count: int


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    file: str
    count: int
    task_ids_hash: str
    file_hash: str


class DedupManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exact_removed: int
    near_removed: int


class DataManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0")
    pipeline_version: str
    seed: int
    sources: dict[str, SourceManifest]
    splits: dict[str, SplitManifest]
    dedup: DedupManifest
    quarantine_total: int


def write_manifest(manifest: DataManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))


def load_manifest(path: Path) -> DataManifest:
    return DataManifest.model_validate(json.loads(path.read_text()))
