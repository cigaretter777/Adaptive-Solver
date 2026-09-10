import inspect

from adaptive_math.agent.environment import ProductMathEnv
from adaptive_math.agent.state import AgentState


def test_product_environment_and_public_state_expose_no_reference_answer_path() -> None:
    parameters = inspect.signature(ProductMathEnv.__init__).parameters

    assert "reference" not in parameters
    assert "labeled_task" not in parameters
    assert "reference" not in AgentState.model_fields
    assert "reference" not in ProductMathEnv.__dict__
