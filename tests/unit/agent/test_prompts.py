from adaptive_math.agent.prompts import render_initial_messages
from adaptive_math.core.types import AnswerType, Budget, MathTask
from adaptive_math.tools.registry import ToolRegistry


def test_agent_budget_presets_validate_against_the_shared_contract() -> None:
    config_dir = Path("configs/agent")

    for path in config_dir.glob("*.yaml"):
        Budget.model_validate(yaml.safe_load(path.read_text()))


def test_prompt_contains_public_task_budget_and_action_protocol_without_reward_language() -> None:
    task = MathTask(
        task_id="unit:prompt",
        problem="Compute 17 * 19.",
        answer_type=AnswerType.INTEGER,
        dataset="unit",
        split="test",
        source_hash="a" * 64,
        pipeline_version="unit-v1",
    )
    messages = render_initial_messages(
        task, Budget(max_steps=2, max_tool_calls=1, max_python_seconds=3, max_observation_chars=100), ToolRegistry([])
    )

    text = "\n".join(message.content for message in messages)
    assert "Compute 17 * 19." in text
    assert "exactly one action" in text
    assert "<tool_call>" in text and "<final>" in text
    assert "reference" not in text.lower()
    assert "reward" not in text.lower()
from pathlib import Path

import yaml
