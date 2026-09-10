import pytest
from pydantic import TypeAdapter, ValidationError

from adaptive_math.agent.actions import AgentAction, FinalAction, ToolAction, ToolCall


def test_tool_call_accepts_only_a_valid_lowercase_tool_name_and_object_arguments() -> None:
    call = ToolCall(name="sympy_tool", arguments={"expression": "x**2 - 1"})
    action = ToolAction(call=call)

    assert action.kind == "tool_call"
    assert TypeAdapter(AgentAction).validate_python(action.model_dump()) == action


@pytest.mark.parametrize(
    ("name", "arguments"),
    [("Python", {}), ("python-tool", {}), ("", {}), ("python", ["print(1)"])],
)
def test_tool_call_rejects_invalid_names_or_non_object_arguments(name: str, arguments: object) -> None:
    with pytest.raises(ValidationError):
        ToolCall(name=name, arguments=arguments)


def test_final_action_requires_a_non_empty_bounded_answer() -> None:
    assert FinalAction(answer="42").kind == "final"
    with pytest.raises(ValidationError):
        FinalAction(answer="")
    with pytest.raises(ValidationError):
        FinalAction(answer="x" * 8193)
