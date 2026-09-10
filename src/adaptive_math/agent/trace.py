"""Portable deterministic trajectory contract shared by runtime and replay."""

from pydantic import BaseModel, ConfigDict

from adaptive_math.agent.state import TerminationReason, TraceEvent, Usage


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    task_id: str
    events: tuple[TraceEvent, ...]
    final_answer: str | None
    termination_reason: TerminationReason
    usage: Usage
    runtime_version: str
