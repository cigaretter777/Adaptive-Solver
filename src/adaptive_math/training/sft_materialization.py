"""Materialize the private, verifier-labeled task pool consumed by teacher rollout."""

import json
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml
from pydantic import BaseModel, ConfigDict

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.core.types import AnswerType, Budget, LabeledMathTask, MathTask, ReferenceAnswer


class TaskPoolManifest(BaseModel):
    """Content-addressed provenance for one private teacher-rollout input file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "sft-task-pool-v1"
    source_parquet: str
    source_parquet_sha256: str
    output_jsonl: str
    output_sha256: str
    task_count: int


_REQUIRED_COLUMNS = frozenset(
    {
        "task_id",
        "problem",
        "answer_type",
        "dataset",
        "split",
        "source_hash",
        "pipeline_version",
        "metadata",
        "reference_value",
        "reference_acceptable_forms",
    }
)


def export_labeled_task_pool(
    source_parquet: Path,
    output_jsonl: Path,
    manifest_path: Path,
    *,
    limit: int | None = None,
) -> TaskPoolManifest:
    """Export a deterministic, private JSONL pool for ``OfflineMathEnv`` only."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    table = pq.read_table(source_parquet)
    missing = sorted(_REQUIRED_COLUMNS.difference(table.column_names))
    if missing:
        raise ValueError(f"source parquet missing required columns: {', '.join(missing)}")

    rows = sorted(table.select(sorted(_REQUIRED_COLUMNS)).to_pylist(), key=lambda row: str(row["task_id"]))
    if limit is not None:
        rows = rows[:limit]
    records = [_labeled_task_from_row(row).model_dump_json() for row in rows]
    payload = "".join(record + "\n" for record in records).encode()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_bytes(payload)
    manifest = TaskPoolManifest(
        source_parquet=str(source_parquet),
        source_parquet_sha256=sha256_hex(source_parquet.read_bytes()),
        output_jsonl=str(output_jsonl),
        output_sha256=sha256_hex(payload),
        task_count=len(records),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def read_task_pool_manifest(path: Path) -> TaskPoolManifest:
    return TaskPoolManifest.model_validate_json(path.read_text())


def load_budget_file(path: Path) -> Budget:
    """Load the shared Agent budget YAML used by rollout and SFT reconstruction."""
    try:
        raw = yaml.safe_load(path.read_text())
        return Budget.model_validate(raw)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"cannot load budget file {path}: {exc}") from exc


def _labeled_task_from_row(row: dict[str, object]) -> LabeledMathTask:
    metadata = row["metadata"]
    if not isinstance(metadata, str):
        raise TypeError("metadata must be a JSON string")
    acceptable_forms = row["reference_acceptable_forms"]
    if not isinstance(acceptable_forms, list) or not all(
        isinstance(value, str) for value in acceptable_forms
    ):
        raise ValueError("reference_acceptable_forms must be a list of strings")
    try:
        answer_type = AnswerType(str(row["answer_type"]))
        return LabeledMathTask(
            task=MathTask(
                task_id=str(row["task_id"]),
                problem=str(row["problem"]),
                answer_type=answer_type,
                dataset=str(row["dataset"]),
                split=str(row["split"]),
                source_hash=str(row["source_hash"]),
                pipeline_version=str(row["pipeline_version"]),
                metadata=json.loads(metadata),
            ),
            reference=ReferenceAnswer(
                value=str(row["reference_value"]),
                answer_type=answer_type,
                acceptable_forms=tuple(acceptable_forms),
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid labeled task row {row.get('task_id')!r}: {exc}") from exc
