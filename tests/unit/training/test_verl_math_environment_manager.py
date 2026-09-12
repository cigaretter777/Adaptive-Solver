import asyncio
from types import SimpleNamespace

from adaptive_math.core.types import AnswerType, Budget, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.reward import RewardConfig
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.training.verl_environment import MathRolloutManager, VerlMathEnvironmentManager


def _task() -> LabeledMathTask:
    return LabeledMathTask(
        task=MathTask(
            task_id="unit:verl",
            problem="1+1",
            answer_type=AnswerType.INTEGER,
            dataset="unit",
            split="train",
            source_hash="a" * 64,
            pipeline_version="v1",
        ),
        reference=ReferenceAnswer(value="2", answer_type=AnswerType.INTEGER),
    )


def test_verl_manager_uses_production_env_for_terminal_reward() -> None:
    config = SimpleNamespace(env=SimpleNamespace(rollout=SimpleNamespace(n=1)))
    manager = VerlMathEnvironmentManager(
        [_task()],
        Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100),
        ToolRegistry([]),
        config,
        policy_version="p1",
    )

    initial, infos = manager.reset({})
    next_observations, rewards, dones, step_infos = manager.step(['<final>{"answer":"2"}</final>'])

    assert initial["text"] == ["1+1"]
    assert infos == [{}]
    assert next_observations["text"] == []
    assert rewards.tolist() == [1.0]
    assert dones.tolist() == [True]
    assert step_infos[0]["policy_version"] == "p1"
    assert manager.success_evaluator(
        total_batch_list=[[{"active_masks": True}]], total_infos=[[step_infos[0]]]
    )["success_rate"].tolist() == [1.0]


def test_verl_manager_rejects_non_grouped_config() -> None:
    config = SimpleNamespace(env=SimpleNamespace(rollout=SimpleNamespace(n=0)))
    try:
        VerlMathEnvironmentManager(
            [_task()],
            Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100),
            ToolRegistry([]),
            config,
            policy_version="p1",
        )
    except ValueError as exc:
        assert "group_size" in str(exc)
    else:
        raise AssertionError("expected a non-grouped rollout config to be rejected")


def test_math_rollout_uses_versioned_reward_breakdown_at_terminal() -> None:
    reward = RewardConfig(
        version="r2-test",
        variant="r2",
        tool_weight=0.15,
        python_weight=0.10,
        invalid_weight=0.10,
        invalid_cap=3,
        token_weight=0.0,
        clip_min=-1.0,
        clip_max=1.0,
    )
    manager = MathRolloutManager(
        Budget(max_steps=2, max_tool_calls=1, max_python_seconds=1, max_observation_chars=100),
        ToolRegistry([]),
        reward_config=reward,
    )
    manager.reset([_task()], group_size=1, policy_version="p1")

    transition = asyncio.run(manager.step(['<final>{"answer":"2"}</final>']))[0]

    assert transition.reward == 1.0
    assert transition.info["reward_version"] == "r2-test"
    assert transition.info["reward_components"] == {
        "correct": 1.0,
        "invalid_penalty": 0.0,
        "python_cost": 0.0,
        "token_cost": 0.0,
        "tool_cost": 0.0,
    }
