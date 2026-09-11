"""Parquet persistence and structural audit for verified SFT records."""

from collections import Counter
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.training.sft_records import SFTTrajectoryRecord


class SFTAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_records: int
    behavior_counts: dict[str, int]


class SFTDataManifest(BaseModel):
    """Immutable provenance for a verified SFT Parquet artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "sft-data-v1"
    parquet: str
    parquet_sha256: str
    total_records: int
    behavior_counts: dict[str, int]
    source_task_pool_manifest_sha256: str
    source_trace_sha256: str


def write_records(records: list[SFTTrajectoryRecord], path: Path) -> None:
    """Write only validated public demonstrations to a compact Parquet file."""
    audit_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"record_json": [record.model_dump_json() for record in records]})
    pq.write_table(table, path, compression="zstd")


def read_records(path: Path) -> list[SFTTrajectoryRecord]:
    table = pq.read_table(path, columns=["record_json"])
    column = table.column("record_json")
    return [SFTTrajectoryRecord.model_validate_json(value.as_py()) for value in column]


def write_sft_manifest(
    records_path: Path,
    manifest_path: Path,
    *,
    source_task_pool_manifest_sha256: str,
    source_trace_sha256: str,
) -> SFTDataManifest:
    """Audit the persisted data and write the manifest consumed by SFT training."""
    for name, value in {
        "source_task_pool_manifest_sha256": source_task_pool_manifest_sha256,
        "source_trace_sha256": source_trace_sha256,
    }.items():
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{name} must be a 64-character lowercase SHA-256")
    audit = audit_records(read_records(records_path))
    manifest = SFTDataManifest(
        parquet=str(records_path),
        parquet_sha256=sha256_hex(records_path.read_bytes()),
        total_records=audit.total_records,
        behavior_counts=audit.behavior_counts,
        source_task_pool_manifest_sha256=source_task_pool_manifest_sha256,
        source_trace_sha256=source_trace_sha256,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def read_sft_manifest(path: Path) -> SFTDataManifest:
    return SFTDataManifest.model_validate_json(path.read_text())


def audit_records(records: list[SFTTrajectoryRecord]) -> SFTAudit:
    """Reject duplicates before reporting deterministic behavior counts."""
    keys = [(record.behavior.value, record.task_id) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate task_id within an SFT behavior bucket")
    counts = Counter(record.behavior.value for record in records)
    return SFTAudit(total_records=len(records), behavior_counts=dict(sorted(counts.items())))
