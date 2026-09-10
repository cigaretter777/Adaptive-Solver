import pytest

from adaptive_math.agent.model_client import ChatMessage
from adaptive_math.training.sft_records import SFTBehavior, SFTTrajectoryRecord
from adaptive_math.verifier.service import VerifierResult, VerifierStatus


def _result() -> VerifierResult:
    return VerifierResult(
        status=VerifierStatus.CORRECT,
        reward=1.0,
        normalized_prediction="2",
        normalized_reference="2",
    )


def test_sft_record_requires_protocol_valid_terminal_assistant_action() -> None:
    record = SFTTrajectoryRecord(
        task_id="unit:1",
        behavior=SFTBehavior.DIRECT,
        messages=(
            ChatMessage(role="system", content="rules"),
            ChatMessage(role="user", content="1+1"),
            ChatMessage(role="assistant", content='<final>{"answer":"2"}</final>'),
        ),
        final_answer="2",
        verifier_result=_result(),
        source_trace_ids=("trace",),
        tool_execution_ids=(),
        tokenizer_revision="unresolved",
        transform_version="sft-v1",
    )

    assert record.behavior is SFTBehavior.DIRECT


def test_sft_record_rejects_tool_role_without_a_preceding_tool_action() -> None:
    with pytest.raises(ValueError, match="tool"):
        SFTTrajectoryRecord(
            task_id="unit:1",
            behavior=SFTBehavior.SYMPY,
            messages=(
                ChatMessage(role="system", content="rules"),
                ChatMessage(role="user", content="1+1"),
                ChatMessage(role="tool", content="2"),
                ChatMessage(role="assistant", content='<final>{"answer":"2"}</final>'),
            ),
            final_answer="2",
            verifier_result=_result(),
            source_trace_ids=("trace",),
            tool_execution_ids=("call",),
            tokenizer_revision="unresolved",
            transform_version="sft-v1",
        )
