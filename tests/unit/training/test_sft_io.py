from pathlib import Path

import pytest

from adaptive_math.agent.model_client import ChatMessage
from adaptive_math.training.sft_io import audit_records, read_records, write_records
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
