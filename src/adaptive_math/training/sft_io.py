"""Parquet persistence and structural audit for verified SFT records."""

from collections import Counter
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict

from adaptive_math.training.sft_records import SFTTrajectoryRecord


class SFTAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_records: int
    behavior_counts: dict[str, int]


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


def audit_records(records: list[SFTTrajectoryRecord]) -> SFTAudit:
    """Reject duplicates before reporting deterministic behavior counts."""
    keys = [(record.behavior.value, record.task_id) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate task_id within an SFT behavior bucket")
    counts = Counter(record.behavior.value for record in records)
    return SFTAudit(total_records=len(records), behavior_counts=dict(sorted(counts.items())))
