import asyncio

from adaptive_math.core.types import AnswerType, Budget, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.training.verl_environment import MathRolloutManager


def _task(task_id: str) -> LabeledMathTask:
    return LabeledMathTask(
        task=MathTask(task_id=task_id, problem="1+1", answer_type=AnswerType.INTEGER, dataset="unit", split="train", source_hash="a" * 64, pipeline_version="v1"),
        reference=ReferenceAnswer(value="2", answer_type=AnswerType.INTEGER),
    )


def test_vectorized_manager_keeps_group_and_environment_state_isolated() -> None:
    manager = MathRolloutManager(Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100), ToolRegistry([]))
    observations = manager.reset([_task("a"), _task("b")], group_size=2, policy_version="p1")

    transitions = asyncio.run(manager.step(['<final>{"answer":"2"}</final>'] * 4))

    assert len(observations) == 4
    assert {transition.group_id for transition in transitions} == {"a", "b"}
    assert all(transition.done and transition.reward == 1.0 for transition in transitions)
    assert all(transition.info["policy_version"] == "p1" for transition in transitions)
