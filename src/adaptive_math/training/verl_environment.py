"""Backend-neutral vectorized core for a future pinned verl-agent adapter."""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from adaptive_math.agent.environment import OfflineMathEnv
from adaptive_math.agent.parser import parse_action
from adaptive_math.core.types import Budget, JSONValue, LabeledMathTask
from adaptive_math.tools.registry import ToolRegistry

if TYPE_CHECKING:
    class _EnvironmentManagerBase:
        def __init__(self, envs: object, projection_f: object, config: object) -> None:
            self.envs = envs
            self.projection_f = projection_f
            self.config = config
else:
    try:  # Imported only in the cloud image where the pinned backend is installed.
        from agent_system.environments.base import EnvironmentManagerBase as _EnvironmentManagerBase
    except ImportError:  # pragma: no cover - exercised by the cloud backend contract.
        class _EnvironmentManagerBase:
            def __init__(self, envs: object, projection_f: object, config: object) -> None:
                self.envs = envs
                self.projection_f = projection_f
                self.config = config


@dataclass(frozen=True)
class RolloutTransition:
    env_id: str
    group_id: str
    policy_version: str
    observation: str | None
    reward: float
    done: bool
    info: dict[str, JSONValue]


class MathRolloutManager:
    """Owns isolated OfflineMathEnv instances for synchronous policy groups."""

    def __init__(self, budget: Budget, registry: ToolRegistry) -> None:
        self._budget = budget
        self._registry = registry
        self._environments: dict[str, OfflineMathEnv] = {}
        self._groups: dict[str, str] = {}
        self._policy_version = ""
        self._step_index = 0

    def reset(
        self, tasks: list[LabeledMathTask], *, group_size: int, policy_version: str
    ) -> list[str]:
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if not policy_version:
            raise ValueError("policy_version is required")
        self._environments = {}
        self._groups = {}
        self._policy_version = policy_version
        self._step_index = 0
        observations: list[str] = []
        for task in tasks:
            for sample_index in range(group_size):
                env_id = f"{task.task.task_id}:{sample_index}"
                self._environments[env_id] = OfflineMathEnv(
                    task, self._budget, self._registry, trace_id=f"rollout:{env_id}"
                )
                self._groups[env_id] = task.task.task_id
                observations.append(task.task.problem)
        return observations

    async def step(self, text_actions: list[str]) -> list[RolloutTransition]:
        active = [
            (env_id, environment)
            for env_id, environment in self._environments.items()
            if environment.state.termination_reason is None
        ]
        if len(text_actions) != len(active):
            raise ValueError("text_actions must have one entry for each active environment")
        self._step_index += 1

        async def one(env_id: str, environment: OfflineMathEnv, text: str) -> RolloutTransition:
            environment.record_model_output(text, generated_tokens=0, monotonic_ms=self._step_index)
            result = await environment.step(parse_action(text).action, self._step_index)
            evaluation = environment.evaluate() if result.terminated else None
            reward = evaluation.reward if evaluation is not None else 0.0
            info: dict[str, JSONValue] = {
                "trace_id": environment.trace_id,
                "policy_version": self._policy_version,
                "termination_reason": result.state.termination_reason.value
                if result.state.termination_reason is not None
                else None,
                "verifier_status": evaluation.status.value if evaluation is not None else None,
            }
            return RolloutTransition(
                env_id=env_id,
                group_id=self._groups[env_id],
                policy_version=self._policy_version,
                observation=result.observation.content if result.observation is not None else None,
                reward=reward,
                done=result.terminated,
                info=info,
            )

        return list(await asyncio.gather(*(one(env_id, environment, text) for (env_id, environment), text in zip(active, text_actions, strict=True))))

    def build_text_obs(self) -> list[str]:
        return [
            environment.state.task.problem
            for environment in self._environments.values()
            if environment.state.termination_reason is None
        ]


class VerlMathEnvironmentManager(_EnvironmentManagerBase):
    """Pinned verl-agent environment adapter backed only by production runtime code.

    The class is importable in the lightweight local environment.  In the
    cloud image it subclasses the exact ``EnvironmentManagerBase`` supplied by
    the SHA recorded in ``third_party/manifest.json``.
    """

    def __init__(
        self,
        tasks: list[LabeledMathTask],
        budget: Budget,
        registry: ToolRegistry,
        config: object,
        *,
        policy_version: str,
    ) -> None:
        try:
            group_size = int(config.env.rollout.n)  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("config.env.rollout.n must be a positive group_size") from exc
        if group_size <= 0:
            raise ValueError("config.env.rollout.n must be a positive group_size")
        if not tasks:
            raise ValueError("at least one labeled task is required")
        super().__init__(None, lambda actions: (actions, [True] * len(actions)), config)
        self._tasks = tasks
        self._group_size = group_size
        self._policy_version = policy_version
        self._manager = MathRolloutManager(budget, registry)

    def reset(self, kwargs: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
        del kwargs
        text = self._manager.reset(
            self._tasks, group_size=self._group_size, policy_version=self._policy_version
        )
        return {"text": text, "image": None, "anchor": text.copy()}, [{} for _ in text]

    def step(
        self, text_actions: list[str]
    ) -> tuple[dict[str, object], np.ndarray, np.ndarray, list[dict[str, JSONValue]]]:
        transitions = asyncio.run(self._manager.step(text_actions))
        infos: list[dict[str, JSONValue]] = []
        for transition in transitions:
            info = dict(transition.info)
            info["won"] = float(transition.reward > 0.0)
            info["is_action_valid"] = True
            infos.append(info)
        return (
            {"text": self._manager.build_text_obs(), "image": None, "anchor": None},
            np.asarray([transition.reward for transition in transitions], dtype=np.float32),
            np.asarray([transition.done for transition in transitions], dtype=bool),
            infos,
        )

    def success_evaluator(self, *args: object, **kwargs: object) -> dict[str, np.ndarray]:
        del args
        total_batch_list = kwargs.get("total_batch_list")
        total_infos = kwargs.get("total_infos")
        if not isinstance(total_batch_list, list) or not isinstance(total_infos, list):
            raise TypeError("total_batch_list and total_infos are required")
        successes: list[float] = []
        for batches, infos in zip(total_batch_list, total_infos, strict=True):
            if not isinstance(batches, list) or not isinstance(infos, list):
                raise TypeError("total batch entries must be lists")
            for batch, info in zip(reversed(batches), reversed(infos), strict=True):
                if isinstance(batch, dict) and batch.get("active_masks") and isinstance(info, dict):
                    won = info.get("won")
                    if not isinstance(won, (float, int)):
                        raise ValueError("terminal rollout info lacks won")
                    successes.append(float(won))
                    break
            else:
                raise ValueError("each batch requires an active terminal step")
        return {"success_rate": np.asarray(successes, dtype=np.float32)}
