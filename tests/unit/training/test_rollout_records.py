import pytest

from adaptive_math.agent.state import TerminationReason, Usage
from adaptive_math.agent.trace import Trajectory
from adaptive_math.training.rollout_records import (
    RolloutStep,
    RolloutTrajectory,
    validate_group_policy_versions,
)
from adaptive_math.verifier.service import VerifierResult, VerifierStatus


def _trajectory() -> Trajectory:
    return Trajectory(trace_id="trace", task_id="task", events=(), final_answer="2", termination_reason=TerminationReason.FINAL, usage=Usage(steps=1), runtime_version="v1")


def _rollout(group_id: str, policy_version: str) -> RolloutTrajectory:
    return RolloutTrajectory(
        steps=(RolloutStep(env_id="e", group_id=group_id, policy_version=policy_version, input_ids=(1,), generated_ids=(2,), assistant_mask=(0, 1), action_text='<final>{"answer":"2"}</final>', observation_text=None, reward=1.0, done=True, info={}),),
        trajectory=_trajectory(),
        verifier_result=VerifierResult(status=VerifierStatus.CORRECT, reward=1.0, normalized_prediction="2", normalized_reference="2"),
    )


def test_rollout_record_preserves_original_model_tokens() -> None:
    rollout = _rollout("g", "p1")

    assert rollout.steps[0].generated_ids == (2,)


def test_group_cannot_mix_policy_versions() -> None:
    with pytest.raises(ValueError, match="policy_version"):
        validate_group_policy_versions([_rollout("g", "p1"), _rollout("g", "p2")])
