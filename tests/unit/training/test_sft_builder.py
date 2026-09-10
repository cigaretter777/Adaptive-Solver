from adaptive_math.agent.state import EventKind, TerminationReason, TraceEvent, Usage
from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.types import AnswerType, Budget, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.training.sft_builder import build_sft_record


def test_builder_reconstructs_a_verified_direct_record_from_runtime_trace() -> None:
    task = MathTask(task_id="unit:1", problem="1+1", answer_type=AnswerType.INTEGER, dataset="unit", split="train", source_hash="a" * 64, pipeline_version="v1")
    trajectory = Trajectory(
        trace_id="trace",
        task_id="unit:1",
        events=(
            TraceEvent(sequence=0, kind=EventKind.MODEL_OUTPUT, monotonic_ms=0, payload={"raw": '<final>{"answer":"2"}</final>', "generated_tokens": 2}),
            TraceEvent(sequence=1, kind=EventKind.FINAL, monotonic_ms=1, payload={"answer": "2"}),
        ),
        final_answer="2", termination_reason=TerminationReason.FINAL,
        usage=Usage(steps=1, generated_tokens=2), runtime_version="runtime-v1+agent-v1",
    )

    record = build_sft_record(LabeledMathTask(task=task, reference=ReferenceAnswer(value="2", answer_type=AnswerType.INTEGER)), trajectory, Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100), ToolRegistry([]))

    assert record.final_answer == "2"
    assert record.verifier_result.status.value == "correct"
