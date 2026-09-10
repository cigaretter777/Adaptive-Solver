"""Convert verified production trajectories into SFT demonstrations."""

from adaptive_math.agent.model_client import ChatMessage
from adaptive_math.agent.prompts import render_initial_messages
from adaptive_math.agent.state import EventKind, TerminationReason
from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.types import Budget, LabeledMathTask
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.training.sft_records import SFTBehavior, SFTTrajectoryRecord
from adaptive_math.verifier.service import VerifierStatus, verify_answer

SFT_TRANSFORM_VERSION = "sft-v1"


def build_sft_record(
    labeled_task: LabeledMathTask,
    trajectory: Trajectory,
    budget: Budget,
    registry: ToolRegistry,
    *,
    tokenizer_revision: str = "unresolved",
) -> SFTTrajectoryRecord:
    """Reconstruct one demonstration from an actual, terminal runtime trace."""
    if trajectory.task_id != labeled_task.task.task_id:
        raise ValueError("trajectory task_id does not match the labeled task")
    if trajectory.termination_reason is not TerminationReason.FINAL or trajectory.final_answer is None:
        raise ValueError("only terminal final-answer trajectories can be used for SFT")
    verdict = verify_answer(trajectory.final_answer, labeled_task.reference, task_id=trajectory.task_id)
    if verdict.status is not VerifierStatus.CORRECT:
        raise ValueError("trajectory final answer is not verifier-correct")
    messages = list(render_initial_messages(labeled_task.public_view(), budget, registry))
    tool_names: list[str] = []
    tool_execution_ids: list[str] = []
    saw_recovery = False
    for event in trajectory.events:
        if event.kind is EventKind.MODEL_OUTPUT:
            raw = event.payload.get("raw")
            if not isinstance(raw, str):
                raise ValueError("model output event lacks raw text")
            messages.append(ChatMessage(role="assistant", content=raw))
        elif event.kind is EventKind.INVALID_ACTION:
            saw_recovery = True
        elif event.kind is EventKind.TOOL_CALL:
            name = event.payload.get("name")
            if not isinstance(name, str):
                raise ValueError("tool call event lacks name")
            tool_names.append(name)
        elif event.kind is EventKind.TOOL_RESULT:
            result = event.payload.get("result")
            if not isinstance(result, dict):
                raise ValueError("tool result event lacks result payload")
            output = result.get("output")
            ok = result.get("ok")
            if not isinstance(output, str) or not isinstance(ok, bool):
                raise ValueError("invalid tool result payload")
            messages.append(ChatMessage(role="tool", content=output))
            tool_execution_ids.append(f"{trajectory.trace_id}:{event.sequence}")
            saw_recovery = saw_recovery or not ok
    behavior = _behavior(tool_names, saw_recovery)
    return SFTTrajectoryRecord(
        task_id=trajectory.task_id,
        behavior=behavior,
        messages=tuple(messages),
        final_answer=trajectory.final_answer,
        verifier_result=verdict,
        source_trace_ids=(trajectory.trace_id,),
        tool_execution_ids=tuple(tool_execution_ids),
        tokenizer_revision=tokenizer_revision,
        transform_version=SFT_TRANSFORM_VERSION,
    )


def _behavior(tool_names: list[str], saw_recovery: bool) -> SFTBehavior:
    if saw_recovery:
        return SFTBehavior.RECOVERY
    if "python" in tool_names:
        return SFTBehavior.PYTHON
    if "sympy" in tool_names:
        return SFTBehavior.SYMPY
    if tool_names:
        raise ValueError(f"unsupported tool in SFT trace: {tool_names!r}")
    return SFTBehavior.DIRECT
