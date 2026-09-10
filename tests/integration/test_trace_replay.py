from pathlib import Path

import pytest

from adaptive_math.agent.replay import TraceEnvelope, TraceReplayError, replay, verify_hash
from adaptive_math.agent.state import EventKind, TerminationReason, TraceEvent, Usage
from adaptive_math.agent.trace import Trajectory


def _trajectory() -> Trajectory:
    return Trajectory(
        trace_id="trace-1",
        task_id="task-1",
        events=(
            TraceEvent(sequence=0, kind=EventKind.MODEL_OUTPUT, monotonic_ms=0, payload={"raw": "<final>{}</final>", "generated_tokens": 3}),
            TraceEvent(sequence=1, kind=EventKind.FINAL, monotonic_ms=1, payload={"answer": "2"}),
        ),
        final_answer="2",
        termination_reason=TerminationReason.FINAL,
        usage=Usage(steps=1, generated_tokens=3),
        runtime_version="runtime-v1+agent-v1",
    )


def test_trace_hash_and_replay_are_deterministic() -> None:
    envelope = TraceEnvelope.create(_trajectory(), prompt_version="agent-v1", created_at="2026-09-10T00:00:00Z")

    assert verify_hash(envelope)
    assert replay(envelope) == _trajectory()


def test_replay_rejects_hash_and_usage_tampering() -> None:
    envelope = TraceEnvelope.create(_trajectory(), prompt_version="agent-v1", created_at="2026-09-10T00:00:00Z")
    changed = envelope.model_copy(update={"content_hash": "0" * 64})
    bad_usage = envelope.model_copy(update={"trajectory": _trajectory().model_copy(update={"usage": Usage()})})

    assert not verify_hash(changed)
    with pytest.raises(TraceReplayError, match="hash"):
        replay(changed)
    with pytest.raises(TraceReplayError, match="usage"):
        replay(bad_usage, verify_content_hash=False)


def test_golden_trace_fixture_replays() -> None:
    path = Path("tests/fixtures/golden_traces/direct_correct.json")
    envelope = TraceEnvelope.model_validate_json(path.read_bytes())

    assert replay(envelope).final_answer == "2"
