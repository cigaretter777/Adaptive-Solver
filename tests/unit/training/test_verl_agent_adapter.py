import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from adaptive_math.training.verl_agent_adapter import (
    ADAPTER_MARKER,
    apply_verl_agent_environment_patch,
    make_adaptive_math_envs,
    task_ids_from_reset_kwargs,
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
    # Upstream passes one env_kwargs dict per gen-batch row; a GRPO group is
    # two consecutive rows carrying the same task id, not a flat id list.
    initial, _ = train.reset(
        np.array([{"task_ids": ["unit:one"]}, {"task_ids": ["unit:one"]}], dtype=object)
    )

    assert validation is not train
    assert len(initial["text"]) == 2
    assert all("1+1" in text for text in initial["text"])


def test_task_ids_from_reset_kwargs_accepts_per_row_dicts() -> None:
    rows = np.array(
        [
            {"task_ids": np.array(["a", "a", "a", "a"])},
            {"task_ids": ["b", "b"]},
            {"task_ids": ("c",)},
        ],
        dtype=object,
    )
    assert task_ids_from_reset_kwargs(rows) == ["a", "b", "c"]


def test_task_ids_from_reset_kwargs_rejects_missing_column() -> None:
    with pytest.raises(TypeError, match="env_kwargs"):
        task_ids_from_reset_kwargs(None)
    with pytest.raises(TypeError, match="env_kwargs"):
        task_ids_from_reset_kwargs({"task_ids": ["a", "a"]})


def test_task_ids_from_reset_kwargs_rejects_malformed_rows() -> None:
    with pytest.raises(TypeError, match="row 0 must be a dict"):
        task_ids_from_reset_kwargs(np.array(["not-a-dict"], dtype=object))
    with pytest.raises(TypeError, match="non-empty 'task_ids'"):
        task_ids_from_reset_kwargs(np.array([{"task_ids": []}], dtype=object))
    with pytest.raises(TypeError, match="must be strings"):
        task_ids_from_reset_kwargs(np.array([{"task_ids": [1, 1]}], dtype=object))
    with pytest.raises(ValueError, match="must repeat one task id"):
        task_ids_from_reset_kwargs(np.array([{"task_ids": ["a", "b"]}], dtype=object))
