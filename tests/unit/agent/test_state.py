import pytest
from pydantic import ValidationError

from adaptive_math.agent.state import AgentState, EventKind, TerminationReason, Usage
from adaptive_math.core.types import AnswerType, Budget, MathTask


def make_state() -> AgentState:
    return AgentState(
        task=MathTask(
            task_id="unit:task",
            problem="What is 1+1?",
            answer_type=AnswerType.INTEGER,
            dataset="unit",
            split="train",
            source_hash="a" * 64,
            pipeline_version="unit-v1",
        ),
        budget=Budget(max_steps=2, max_tool_calls=1, max_python_seconds=1.0, max_observation_chars=100),
    )


def test_state_transitions_are_immutable_and_assign_monotonic_sequences() -> None:
    state = make_state()
    next_state = state.append_event(EventKind.MODEL_OUTPUT, {"raw": "<final>{\"answer\":\"2\"}</final>"}, 0)

    assert state.events == ()
    assert next_state.events[0].sequence == 0
    assert next_state.events[0].monotonic_ms == 0
    with pytest.raises(ValidationError):
        next_state.events[0].sequence = 1


def test_usage_and_budget_exhaustion_are_accumulated_without_mutation() -> None:
    state = make_state().with_usage(steps=1, tool_calls=1, python_seconds=0.5, generated_tokens=3)

    assert state.usage == Usage(steps=1, tool_calls=1, python_seconds=0.5, generated_tokens=3)
    assert not state.is_budget_exhausted()
    exhausted = state.with_usage(steps=1, python_seconds=0.5)
    assert exhausted.is_budget_exhausted()
    assert exhausted.budget_reason() is TerminationReason.MAX_STEPS


def test_terminated_state_rejects_further_transitions() -> None:
    state = make_state().terminate(TerminationReason.FINAL, final_answer="2")

    with pytest.raises(ValueError):
        state.append_event(EventKind.MODEL_OUTPUT, {"raw": "later"}, 1)
    with pytest.raises(ValueError):
        state.with_usage(steps=1)
