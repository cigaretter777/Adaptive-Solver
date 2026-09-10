"""Collect verifier-correct teacher trajectories through the production runtime."""

import asyncio
from dataclasses import dataclass

from adaptive_math.agent.environment import OfflineMathEnv
from adaptive_math.agent.loop import AgentLoop
from adaptive_math.agent.model_client import GenerationConfig, ModelClient
from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.types import Budget, LabeledMathTask
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.verifier.service import VerifierStatus


@dataclass(frozen=True)
class VerifiedRollout:
    task_id: str
    trajectory: Trajectory


async def collect_verified_rollouts(
    tasks: list[LabeledMathTask],
    model: ModelClient,
    budget: Budget,
    registry: ToolRegistry,
    generation: GenerationConfig,
    *,
    max_concurrency: int = 1,
) -> list[VerifiedRollout]:
    """Run teachers through OfflineMathEnv and discard all non-correct terminals."""
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(task: LabeledMathTask) -> VerifiedRollout | None:
        async with semaphore:
            environment = OfflineMathEnv(task, budget, registry, trace_id=f"teacher:{task.task.task_id}")
            trajectory = await AgentLoop().run(environment, model, generation)
            evaluation = environment.evaluate()
            if evaluation is None or evaluation.status is not VerifierStatus.CORRECT:
                return None
            return VerifiedRollout(task_id=task.task.task_id, trajectory=trajectory)

    candidates = await asyncio.gather(*(one(task) for task in tasks))
    return [candidate for candidate in candidates if candidate is not None]
