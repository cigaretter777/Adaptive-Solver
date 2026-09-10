import asyncio

from adaptive_math.agent.environment import ProductMathEnv
from adaptive_math.agent.loop import AgentLoop
from adaptive_math.agent.model_client import GenerationConfig, ModelTurn
from adaptive_math.agent.state import TerminationReason
from adaptive_math.core.types import AnswerType, Budget, MathTask
from adaptive_math.tools.registry import ToolRegistry


class ScriptedModel:
    def __init__(self, turns: list[str]) -> None:
        self._turns = iter(turns)

    async def generate(self, messages: tuple[object, ...], config: GenerationConfig) -> ModelTurn:
        return ModelTurn(text=next(self._turns), prompt_tokens=1, generated_tokens=2, finish_reason="stop", model_id="scripted")


def _environment() -> ProductMathEnv:
    task = MathTask(
        task_id="unit:loop", problem="What is 1 + 1?", answer_type=AnswerType.INTEGER,
        dataset="unit", split="test", source_hash="a" * 64, pipeline_version="unit-v1"
    )
    return ProductMathEnv(task, Budget(max_steps=3, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100), ToolRegistry([]))


def test_loop_runs_direct_final_to_terminal_trajectory() -> None:
    trajectory = asyncio.run(
        AgentLoop().run(_environment(), ScriptedModel(['<final>{"answer":"2"}</final>']), GenerationConfig())
    )

    assert trajectory.final_answer == "2"
    assert trajectory.termination_reason is TerminationReason.FINAL
    assert trajectory.usage.generated_tokens == 2


def test_loop_allows_invalid_action_then_recovery() -> None:
    trajectory = asyncio.run(
        AgentLoop().run(
            _environment(),
            ScriptedModel(["not protocol", '<final>{"answer":"2"}</final>']),
            GenerationConfig(),
        )
    )

    assert trajectory.final_answer == "2"
    assert trajectory.usage.invalid_actions == 1
