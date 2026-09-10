"""Validated multi-turn demonstration records for agent SFT."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from adaptive_math.agent.actions import FinalAction, ToolAction
from adaptive_math.agent.model_client import ChatMessage
from adaptive_math.agent.parser import parse_action
from adaptive_math.verifier.service import VerifierResult, VerifierStatus


class SFTBehavior(StrEnum):
    DIRECT = "direct"
    PYTHON = "python"
    SYMPY = "sympy"
    RECOVERY = "recovery"


class SFTTrajectoryRecord(BaseModel):
    """A correct, protocol-valid agent conversation ready for tokenization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    behavior: SFTBehavior
    messages: tuple[ChatMessage, ...]
    final_answer: str
    verifier_result: VerifierResult
    source_trace_ids: tuple[str, ...]
    tool_execution_ids: tuple[str, ...]
    tokenizer_revision: str
    transform_version: str

    @model_validator(mode="after")
    def _validate_conversation(self) -> "SFTTrajectoryRecord":
        if len(self.messages) < 3 or [message.role for message in self.messages[:2]] != [
            "system",
            "user",
        ]:
            raise ValueError("messages must begin with system then user")
        previous_action: ToolAction | FinalAction | None = None
        assistant_actions: list[FinalAction | ToolAction] = []
        for index, message in enumerate(self.messages[2:], start=2):
            if message.role == "assistant":
                parsed = parse_action(message.content)
                if parsed.action is None:
                    raise ValueError(f"assistant message {index} violates action protocol")
                previous_action = parsed.action
                assistant_actions.append(parsed.action)
            elif message.role == "tool":
                if not isinstance(previous_action, ToolAction):
                    raise ValueError("tool observation must follow a tool action")
                previous_action = None
            else:
                raise ValueError("only assistant and tool messages may follow the initial task")
        if not assistant_actions or not isinstance(assistant_actions[-1], FinalAction):
            raise ValueError("last assistant message must be a final action")
        if assistant_actions[-1].answer != self.final_answer:
            raise ValueError("final answer does not match terminal assistant action")
        if self.verifier_result.status is not VerifierStatus.CORRECT:
            raise ValueError("SFT record requires a correct verifier result")
        return self
