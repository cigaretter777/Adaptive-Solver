"""Run the product math agent with a local Transformers model or a test action."""

import argparse
import asyncio
from pathlib import Path

import orjson
import yaml

from adaptive_math.agent.environment import ProductMathEnv
from adaptive_math.agent.loop import AgentLoop
from adaptive_math.agent.model_client import ChatMessage, GenerationConfig, ModelTurn
from adaptive_math.agent.transformers_client import TransformersModelClient
from adaptive_math.core.hashing import make_source_hash, make_task_id
from adaptive_math.core.types import AnswerType, Budget, MathTask
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.tools.sympy_tool import SympyTool


class _ScriptedModel:
    def __init__(self, action: str) -> None:
        self._action = action

    async def generate(
        self, messages: tuple[ChatMessage, ...], config: GenerationConfig
    ) -> ModelTurn:
        del messages, config
        return ModelTurn(
            text=self._action,
            prompt_tokens=0,
            generated_tokens=0,
            finish_reason="scripted",
            model_id="scripted",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--answer-type", required=True, choices=[kind.value for kind in AnswerType])
    parser.add_argument("--model", required=False)
    parser.add_argument("--config", type=Path, default=Path("configs/agent/default.yaml"))
    parser.add_argument("--scripted-action", help="test-only protocol action; never use for evaluation")
    args = parser.parse_args()
    if args.model is None and args.scripted_action is None:
        parser.error("--model is required unless using --scripted-action for a local test")
    budget = Budget.model_validate(yaml.safe_load(args.config.read_text()))
    task = MathTask(
        task_id=make_task_id("interactive", args.problem),
        problem=args.problem,
        answer_type=AnswerType(args.answer_type),
        dataset="interactive",
        split="product",
        source_hash=make_source_hash(args.problem.encode()),
        pipeline_version="runtime-v1",
    )
    model = _ScriptedModel(args.scripted_action) if args.scripted_action else TransformersModelClient.from_pretrained(args.model)
    environment = ProductMathEnv(task, budget, ToolRegistry([SympyTool()]))
    trajectory = asyncio.run(AgentLoop().run(environment, model, GenerationConfig()))
    print(orjson.dumps(trajectory.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
