from adaptive_math.agent.state import TerminationReason, Usage
from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.types import Budget
from adaptive_math.reward import R2_DEFAULT, RewardContext, compute_reward
from adaptive_math.training.reward_bridge import reward_for_trajectory
from adaptive_math.verifier.service import VerifierResult, VerifierStatus


def test_reward_bridge_matches_production_reward_function() -> None:
    trajectory = Trajectory(trace_id="t", task_id="task", events=(), final_answer="2", termination_reason=TerminationReason.FINAL, usage=Usage(steps=1, tool_calls=1, generated_tokens=10), runtime_version="v1")
    verdict = VerifierResult(status=VerifierStatus.CORRECT, reward=1.0, normalized_prediction="2", normalized_reference="2")
    budget = Budget(max_steps=2, max_tool_calls=4, max_python_seconds=12, max_observation_chars=100)

    actual = reward_for_trajectory(trajectory, verdict, budget, max_generated_tokens=100, config=R2_DEFAULT)
    expected = compute_reward(RewardContext(verifier_result=verdict, tool_calls=1, python_seconds=0, invalid_action_count=0, generated_tokens=10, max_generated_tokens=100, budget=budget), R2_DEFAULT)

    assert actual == expected
