"""Backend-neutral vectorized core for a future pinned verl-agent adapter."""

import asyncio
from dataclasses import dataclass

from adaptive_math.agent.environment import OfflineMathEnv
from adaptive_math.agent.parser import parse_action
from adaptive_math.core.types import Budget, JSONValue, LabeledMathTask
from adaptive_math.tools.registry import ToolRegistry


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
