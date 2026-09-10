"""Token-preserving records exchanged between agent rollout and GRPO backend."""

from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.types import JSONValue
from adaptive_math.verifier.service import VerifierResult


class RolloutStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    env_id: str
    group_id: str
    policy_version: str
    input_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    assistant_mask: tuple[int, ...]
    action_text: str
    observation_text: str | None
    reward: float
    done: bool
    info: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _mask_matches_generated_tokens(self) -> "RolloutStep":
        if len(self.assistant_mask) != len(self.input_ids) + len(self.generated_ids):
            raise ValueError("assistant_mask must cover input_ids plus generated_ids")
        return self


class RolloutTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: tuple[RolloutStep, ...]
    trajectory: Trajectory
    verifier_result: VerifierResult

    @model_validator(mode="after")
    def _is_ordered_and_terminal(self) -> "RolloutTrajectory":
        if not self.steps or not self.steps[-1].done:
            raise ValueError("rollout trajectory requires a terminal final step")
        return self


def validate_group_policy_versions(rollouts: list[RolloutTrajectory]) -> None:
    """Reject stale-policy mixing before group-relative advantages are computed."""
    versions: dict[str, set[str]] = defaultdict(set)
    for rollout in rollouts:
        for step in rollout.steps:
            versions[step.group_id].add(step.policy_version)
    mixed = {group_id: values for group_id, values in versions.items() if len(values) > 1}
    if mixed:
        raise ValueError(f"group_id has mixed policy_version values: {mixed}")
