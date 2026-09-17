"""Contract test: the real upstream TrajectoryCollector drives the adaptive-math env.

Runs verl-agent's production ``vanilla_multi_turn_loop`` (pinned checkout,
importable in the training environment) against ``VerlMathEnvironmentManager``
with a stubbed actor worker group, so no GPU or vLLM is needed.  It locks the
contract boundary: per-row env_kwargs, one env per batch row, fixed-width
steps across heterogeneous termination, and tool results reaching the model.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from adaptive_math.core.types import Budget, LabeledMathTask
from adaptive_math.tools.python_tool import PythonTool
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.tools.sandboxfusion import SandboxFusionClient
from adaptive_math.tools.sympy_tool import SympyTool
from adaptive_math.training.verl_environment import VerlMathEnvironmentManager

REPO_ROOT = Path(__file__).parents[2]
TOKENIZER_DIR = REPO_ROOT / "artifacts/models/qwen3_1_7b_sft_dp_v1_merged"
TRAIN_PARQUET = REPO_ROOT / "artifacts/task_pools/rl_smoke_10x4_seed42.parquet"
TASK_POOL = REPO_ROOT / "artifacts/task_pools/rl_smoke.jsonl"


def _skip_reasons() -> str | None:
    if not TRAIN_PARQUET.is_file() or not TASK_POOL.is_file():
        return f"smoke task pool missing: {TRAIN_PARQUET}"
    if not (TOKENIZER_DIR / "tokenizer_config.json").is_file():
        return f"local model snapshot missing: {TOKENIZER_DIR}"
    try:
        import agent_system.multi_turn_rollout  # noqa: F401
        from verl import DataProto  # noqa: F401
    except ImportError as exc:
        return f"pinned verl-agent is not importable: {exc}"
    return None


pytestmark = pytest.mark.skipif(_skip_reasons() is not None, reason=_skip_reasons() or "")


def _load_task_lookup() -> dict[str, LabeledMathTask]:
    tasks: dict[str, LabeledMathTask] = {}
    for line in TASK_POOL.read_text().splitlines():
        if not line.strip():
            continue
        task = LabeledMathTask.model_validate(json.loads(line))
        tasks[task.task.task_id] = task
    return tasks


class _StubActorRolloutWG:
    """Returns scripted per-row responses instead of running a policy."""

    world_size = 1

    def __init__(self, tokenizer, scripts: list[list[str]]) -> None:
        self._tokenizer = tokenizer
        self._scripts = scripts
        self._step = 0

    def generate_sequences(self, batch_input):
        import torch
        from verl import DataProto

        texts = self._scripts[self._step]
        self._step += 1
        assert len(texts) == len(batch_input.batch["input_ids"])
        encoded = [self._tokenizer.encode(text, add_special_tokens=False) for text in texts]
        width = max(len(ids) for ids in encoded)
        responses = np.full((len(encoded), width), self._tokenizer.pad_token_id, dtype=np.int64)
        for row, ids in enumerate(encoded):
            responses[row, : len(ids)] = ids
        responses = torch.from_numpy(responses)
        # Mirror the real vLLM rollout output shape: prompts, responses,
        # whole-sequence input_ids, attention_mask and position_ids.
        prompts = batch_input.batch["input_ids"]
        input_ids = torch.cat([prompts, responses], dim=-1)
        attention_mask = torch.cat(
            [batch_input.batch["attention_mask"], torch.ones_like(responses)], dim=-1
        )
        position_ids = torch.cat(
            [batch_input.batch["position_ids"], torch.ones_like(responses)], dim=-1
        )
        return DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "input_ids": input_ids,
                "rollout_log_probs": torch.zeros_like(responses, dtype=torch.float32),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )


class _RecordingEnv:
    """Delegates to the real manager while capturing each step's observations."""

    def __init__(self, manager: VerlMathEnvironmentManager) -> None:
        self._manager = manager
        self.obs_history: list[dict[str, object]] = []

    def reset(self, kwargs):
        observations, infos = self._manager.reset(kwargs)
        self.obs_history.append(observations)
        return observations, infos

    def step(self, text_actions):
        observations, rewards, dones, infos = self._manager.step(text_actions)
        self.obs_history.append(observations)
        return observations, rewards, dones, infos

    def success_evaluator(self, **kwargs):
        return self._manager.success_evaluator(**kwargs)


def test_upstream_collector_runs_adaptive_math_rollout(tmp_path: Path) -> None:
    """2 prompts x group 2 with heterogeneous termination and a wrong answer."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    from agent_system.multi_turn_rollout import TrajectoryCollector

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_config = OmegaConf.create(
        {
            "cache_dir": str(tmp_path / "dataset_cache"),
            "use_shm": False,
            "prompt_key": "prompt",
            # Multi-turn observations carry the full conversation, so the
            # prompt budget must cover several turns, not one problem.
            "max_prompt_length": 2048,
            "return_raw_chat": True,
            "truncation": "error",
            "filter_overlong_prompts": False,
            "shuffle": False,
        }
    )
    dataset = RLHFDataset(
        data_files=[str(TRAIN_PARQUET)], tokenizer=tokenizer, processor=None, config=data_config
    )
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
    batch = DataProto.from_single_dict(next(iter(dataloader)))
    gen_batch = batch.pop(
        batch_keys=["input_ids", "attention_mask", "position_ids"],
        non_tensor_batch_keys=[
            "raw_prompt_ids",
            "data_source",
            "raw_prompt",
            "tools_kwargs",
            "env_kwargs",
        ],
    )

    lookup = _load_task_lookup()
    # SequentialSampler: the first two pool rows are the two batch tasks.
    pool_ids = [task.task.task_id for task in lookup.values()][:2]
    answers = [lookup[task_id].reference.value for task_id in pool_ids]

    def final(answer: str) -> str:
        return f"<final>{json.dumps({'answer': answer})}</final>"

    tool_call = (
        '<tool_call>{"name": "sympy", "arguments": '
        '{"operation": "numeric", "expression": "1+1"}}</tool_call>'
    )
    # Interleaved repeat yields rows [t0, t0, t1, t1]; rows 0 and 1 are the
    # same task, as are rows 2 and 3.
    scripts = [
        [tool_call, tool_call, tool_call, tool_call],
        [
            final(answers[0]),  # row 0 terminates correct
            tool_call,  # row 1 keeps going (heterogeneous termination)
            final(answers[1]),  # row 2 terminates correct
            final("999"),  # row 3 terminates wrong
        ],
        [tool_call, final(answers[0]), tool_call, tool_call],
    ]
    actor_rollout_wg = _StubActorRolloutWG(tokenizer, scripts)

    manager = VerlMathEnvironmentManager(
        [],
        Budget(max_steps=6, max_tool_calls=4, max_python_seconds=12.0, max_observation_chars=8000),
        ToolRegistry([SympyTool(), PythonTool(SandboxFusionClient())]),
        OmegaConf.create({"env": {"env_name": "adaptive_math", "rollout": {"n": 2}}}),
        policy_version="contract-test",
        task_lookup=lookup,
    )
    recorder = _RecordingEnv(manager)
    collector_config = OmegaConf.create(
        {
            "env": {"rollout": {"n": 2}, "max_steps": 3},
            "data": {
                "max_prompt_length": 2048,
                "truncation": "error",
                "return_raw_chat": True,
                "apply_chat_template_kwargs": {},
            },
            "algorithm": {"filter_groups": {"enable": False}},
        }
    )
    collector = TrajectoryCollector(config=collector_config, tokenizer=tokenizer, processor=None)

    output = collector.multi_turn_loop(
        gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, envs=recorder, is_train=True
    )

    # Fixed-width contract: every step observes exactly the 4 batch rows,
    # including steps after rows terminated.
    assert all(len(obs["text"]) == 4 for obs in recorder.obs_history)
    # Tool results reach the model in the next observation.
    assert any("2.00000000000000" in text for text in recorder.obs_history[1]["text"])
    # Every active step row of an environment carries that environment's
    # final cumulative reward: three correct environments (7 step rows)
    # repeat 1.0 and the wrong environment (2 step rows) stays at 0.0.
    episode_rewards = output.non_tensor_batch["episode_rewards"].tolist()
    assert set(episode_rewards) == {0.0, 1.0}
    assert episode_rewards.count(1.0) == 7 and episode_rewards.count(0.0) == 2
    assert float(np.mean(output.non_tensor_batch["success_rate"])) == pytest.approx(0.75)
    # One row used two tool calls; rows are grouped into two uids.
    assert max(output.non_tensor_batch["tool_callings"].tolist()) == 2
    assert len(set(output.non_tensor_batch["uid"].tolist())) == 2
    assert len(set(output.non_tensor_batch["traj_uid"].tolist())) == 4
    assert output.batch["responses"].shape[0] == len(output.batch["input_ids"])
