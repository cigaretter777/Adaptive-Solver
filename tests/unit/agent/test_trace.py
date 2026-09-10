from adaptive_math.agent.state import EventKind, TerminationReason, TraceEvent, Usage
from adaptive_math.agent.trace import Trajectory


def test_trajectory_json_round_trip_preserves_deterministic_event_stream() -> None:
    trajectory = Trajectory(
        trace_id="trace-1",
        task_id="unit:task",
        events=(TraceEvent(sequence=0, kind=EventKind.FINAL, monotonic_ms=5, payload={"answer": "2"}),),
        final_answer="2",
        termination_reason=TerminationReason.FINAL,
        usage=Usage(steps=1),
        runtime_version="runtime-v1",
    )

    assert Trajectory.model_validate_json(trajectory.model_dump_json()) == trajectory
