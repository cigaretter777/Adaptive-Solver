import json
from pathlib import Path
from types import SimpleNamespace

from adaptive_math.training.verl_agent_adapter import (
    ADAPTER_MARKER,
    apply_verl_agent_environment_patch,
    make_adaptive_math_envs,
)


def _config(pool: Path) -> SimpleNamespace:
    return SimpleNamespace(
        env=SimpleNamespace(
            env_name="adaptive_math",
            rollout=SimpleNamespace(n=2),
            task_pool_path=str(pool),
            budget_path="configs/agent/default.yaml",
            reward_path="configs/reward/r0.yaml",
        )
    )


def test_patch_registers_only_adaptive_math_branch_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "env_manager.py"
    source.write_text("def make_envs(config):\n    return ('upstream', 'upstream')\n")

    first = apply_verl_agent_environment_patch(source)
    second = apply_verl_agent_environment_patch(source)
    patched = source.read_text()

    assert first.changed is True
    assert second.changed is False
    assert patched.count(ADAPTER_MARKER) == 1
    assert "if config.env.env_name == 'adaptive_math':" in patched
    assert "return ('upstream', 'upstream')" in patched


def test_math_factory_uses_only_reset_task_ids_from_private_pool(tmp_path: Path) -> None:
    pool = tmp_path / "tasks.jsonl"
    pool.write_text(
        json.dumps(
            {
                "task": {
                    "task_id": "unit:one",
                    "problem": "1+1",
                    "answer_type": "integer",
                    "dataset": "unit",
                    "split": "train",
                    "source_hash": "a" * 64,
                    "pipeline_version": "v1",
                    "metadata": {},
                },
                "reference": {"value": "2", "answer_type": "integer", "acceptable_forms": []},
            }
        )
        + "\n"
    )

    train, validation = make_adaptive_math_envs(_config(pool))
    initial, _ = train.reset({"task_ids": ["unit:one", "unit:one"]})

    assert validation is not train
    assert initial["text"] == ["1+1", "1+1"]
