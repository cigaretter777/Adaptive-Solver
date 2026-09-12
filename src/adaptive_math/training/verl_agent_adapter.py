"""Auditable registration bridge from the pinned ``verl-agent`` checkout.

The pinned upstream resolves environment names inside its Ray actor, so a
driver-only monkey patch is not sufficient.  This module applies one narrow,
idempotent registration branch to the verified upstream checkout and provides
the factory used by that branch.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import yaml

from adaptive_math.core.types import LabeledMathTask
from adaptive_math.reward import RewardConfig
from adaptive_math.tools.base import Tool
from adaptive_math.tools.python_tool import PythonTool
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.tools.sandboxfusion import SandboxFusionClient
from adaptive_math.tools.sympy_tool import SympyTool
from adaptive_math.training.sft_materialization import load_budget_file
from adaptive_math.training.verl_environment import VerlMathEnvironmentManager

ADAPTER_MARKER = "# adaptive-math-rl: verl-agent environment registration"


class _RolloutConfig(Protocol):
    n: int


class _EnvironmentConfig(Protocol):
    env_name: str
    rollout: _RolloutConfig
    task_pool_path: str
    budget_path: str
    reward_path: str


class _Config(Protocol):
    env: _EnvironmentConfig


@dataclass(frozen=True)
class PatchResult:
    path: Path
    changed: bool


def apply_verl_agent_environment_patch(path: Path) -> PatchResult:
    """Register ``adaptive_math`` without altering behavior of upstream names."""
    source = path.read_text()
    if ADAPTER_MARKER in source:
        return PatchResult(path=path, changed=False)
    needle = "def make_envs(config):\n"
    if source.count(needle) != 1:
        raise ValueError(f"cannot locate a unique make_envs function in {path}")
    registration = (
        f"{needle}{ADAPTER_MARKER}\n"
        "    if config.env.env_name == 'adaptive_math':\n"
        "        from adaptive_math.training.verl_agent_adapter import make_adaptive_math_envs\n"
        "        return make_adaptive_math_envs(config)\n"
    )
    path.write_text(source.replace(needle, registration, 1))
    return PatchResult(path=path, changed=True)


def make_adaptive_math_envs(config: _Config) -> tuple[VerlMathEnvironmentManager, VerlMathEnvironmentManager]:
    """Build train/validation managers backed by one immutable private task pool."""
    if config.env.env_name != "adaptive_math":
        raise ValueError("adaptive-math factory requires env.env_name='adaptive_math'")
    tasks = _load_task_pool(Path(config.env.task_pool_path))
    if not tasks:
        raise ValueError("adaptive-math task pool is empty")
    budget = load_budget_file(Path(config.env.budget_path))
    reward = _load_reward_file(Path(config.env.reward_path))
    registry = ToolRegistry(
        [cast(Tool, SympyTool()), cast(Tool, PythonTool(SandboxFusionClient()))]
    )
    manager = VerlMathEnvironmentManager(
        [],
        budget,
        registry,
        config,
        policy_version="upstream-managed",
        task_lookup=tasks,
        reward_config=reward,
    )
    validation_config = deepcopy(config)
    validation_config.env.rollout.n = 1
    validation = VerlMathEnvironmentManager(
        [],
        budget,
        registry,
        validation_config,
        policy_version="upstream-managed",
        task_lookup=tasks,
        reward_config=reward,
    )
    # Validation evaluates one trajectory per prompt rather than GRPO groups.
    return manager, validation


def _load_task_pool(path: Path) -> dict[str, LabeledMathTask]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read adaptive-math task pool {path}: {exc}") from exc
    tasks: dict[str, LabeledMathTask] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            task = LabeledMathTask.model_validate(json.loads(line))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid task pool row {number} in {path}: {exc}") from exc
        if task.task.task_id in tasks:
            raise ValueError(f"duplicate task id {task.task.task_id!r} in {path}")
        tasks[task.task.task_id] = task
    return tasks


def _load_reward_file(path: Path) -> RewardConfig:
    try:
        raw = yaml.safe_load(path.read_text())
        return RewardConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"cannot load reward file {path}: {exc}") from exc


def task_ids_from_reset_kwargs(kwargs: object, *, group_size: int) -> list[str]:
    """Recover one task id per upstream-repeated rollout group."""
    if not isinstance(kwargs, dict):
        raise TypeError("adaptive-math reset requires env_kwargs with task_ids")
    values = kwargs.get("task_ids")
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise TypeError("adaptive-math env_kwargs.task_ids must be a list of strings")
    if not values or len(values) % group_size:
        raise ValueError("task_ids must contain a non-empty whole number of rollout groups")
    ids: list[str] = []
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        if len(set(group)) != 1:
            raise ValueError("each GRPO group must repeat exactly one task id")
        ids.append(cast(str, group[0]))
    return ids
