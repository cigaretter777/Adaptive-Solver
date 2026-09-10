import asyncio

from adaptive_math.agent.model_client import ChatMessage, GenerationConfig, ModelTurn
from adaptive_math.core.types import AnswerType, Budget, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.training.teacher_rollout import collect_verified_rollouts


class DirectTeacher:
    async def generate(
        self, messages: tuple[ChatMessage, ...], config: GenerationConfig
    ) -> ModelTurn:
        return ModelTurn(text='<final>{"answer":"2"}</final>', prompt_tokens=1, generated_tokens=2, finish_reason="stop", model_id="teacher")


def test_teacher_rollout_keeps_only_verifier_correct_trajectories() -> None:
    labeled = LabeledMathTask(
        task=MathTask(task_id="unit:teacher", problem="1+1", answer_type=AnswerType.INTEGER, dataset="unit", split="train", source_hash="a" * 64, pipeline_version="v1"),
        reference=ReferenceAnswer(value="2", answer_type=AnswerType.INTEGER),
    )
    records = asyncio.run(
        collect_verified_rollouts([labeled], DirectTeacher(), Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100), ToolRegistry([]), GenerationConfig())
    )

    assert len(records) == 1
    assert records[0].trajectory.final_answer == "2"
