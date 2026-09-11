from pathlib import Path

import pytest

from adaptive_math.agent.model_client import ChatMessage
from adaptive_math.training.sft_io import (
    audit_records,
    read_records,
    read_sft_manifest,
    write_records,
    write_sft_manifest,
)
from adaptive_math.training.sft_records import SFTBehavior, SFTTrajectoryRecord
from adaptive_math.verifier.service import VerifierResult, VerifierStatus


def _record(task_id: str = "task") -> SFTTrajectoryRecord:
    return SFTTrajectoryRecord(
        task_id=task_id,
        behavior=SFTBehavior.DIRECT,
        messages=(
            ChatMessage(role="system", content="rules"),
            ChatMessage(role="user", content="1+1"),
            ChatMessage(role="assistant", content='<final>{"answer":"2"}</final>'),
        ),
        final_answer="2",
        verifier_result=VerifierResult(status=VerifierStatus.CORRECT, reward=1.0, normalized_prediction="2", normalized_reference="2"),
        source_trace_ids=("trace",), tool_execution_ids=(), tokenizer_revision="unresolved", transform_version="sft-v1",
    )


def test_sft_parquet_round_trip_and_audit(tmp_path: Path) -> None:
    path = tmp_path / "sft.parquet"
    write_records([_record()], path)

    loaded = read_records(path)
    audit = audit_records(loaded)

    assert loaded == [_record()]
    assert audit.total_records == 1
    assert audit.behavior_counts == {"direct": 1}


def test_sft_audit_rejects_duplicate_behavior_task_pair() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        audit_records([_record(), _record()])


def test_sft_manifest_content_addresses_verified_parquet(tmp_path: Path) -> None:
    records_path = tmp_path / "sft.parquet"
    manifest_path = tmp_path / "sft.manifest.json"
    write_records([_record()], records_path)

    manifest = write_sft_manifest(
        records_path,
        manifest_path,
        source_task_pool_manifest_sha256="a" * 64,
        source_trace_sha256="b" * 64,
    )

    assert manifest.total_records == 1
    assert manifest.behavior_counts == {"direct": 1}
    assert len(manifest.parquet_sha256) == 64
    assert read_sft_manifest(manifest_path) == manifest
