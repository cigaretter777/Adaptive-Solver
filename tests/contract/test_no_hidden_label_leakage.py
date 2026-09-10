"""Hidden labels must never appear in agent-facing serializations.

The offline/online boundary is enforced by TYPE: LabeledMathTask and
VerifierResult carry reference/reward data by design and live only in
training/evaluation code paths. Every public, agent-facing model (MathTask,
its public_view, Budget) must serialize with no key matching the forbidden
label vocabulary. Extend FORBIDDEN_KEYS to the runtime plan's agent-facing
models as they are implemented.
"""

from adaptive_math.core.types import (
    AnswerType,
    Budget,
    LabeledMathTask,
    MathTask,
    ReferenceAnswer,
)

FORBIDDEN_KEYS = {
    "answer",
    "reference",
    "reference_answer",
    "ground_truth",
    "expected_output",
    "verifier_handle",
    "reward",
}


def find_forbidden_keys(value: object, path: str = "$") -> list[str]:
    """Recursively collect paths whose dict key exactly matches a forbidden key."""
    hits: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{path}.{key}"
            if key in FORBIDDEN_KEYS:
                hits.append(current)
            hits.extend(find_forbidden_keys(item, current))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            hits.extend(find_forbidden_keys(item, f"{path}[{index}]"))
    return hits


def make_labeled_task() -> LabeledMathTask:
    return LabeledMathTask(
        task=MathTask(
            task_id="aime_2024:0123456789abcdef0123",
            problem="What is 1+1?",
            answer_type=AnswerType.INTEGER,
            dataset="aime_2024",
            split="train",
            source_hash="a" * 64,
            pipeline_version="canonicalize-v1",
        ),
        reference=ReferenceAnswer(value="2", answer_type=AnswerType.INTEGER),
    )


def test_math_task_serialization_has_no_forbidden_keys() -> None:
    task = make_labeled_task().task
    assert find_forbidden_keys(task.model_dump()) == []
    assert find_forbidden_keys(task.model_dump(mode="json")) == []
    import json

    assert find_forbidden_keys(json.loads(task.model_dump_json())) == []


def test_public_view_has_no_forbidden_keys() -> None:
    public = make_labeled_task().public_view()
    assert isinstance(public, MathTask)
    assert find_forbidden_keys(public.model_dump()) == []
    assert find_forbidden_keys(public.model_dump_json()) == []


def test_budget_serialization_has_no_forbidden_keys() -> None:
    budget = Budget(max_steps=6, max_tool_calls=4, max_python_seconds=12.0, max_observation_chars=8000)
    assert find_forbidden_keys(budget.model_dump()) == []


def test_offline_types_carry_labels_by_design() -> None:
    # The boundary is typed: the offline-only model DOES carry the reference.
    # This inverted assertion pins the contract so nobody "helpfully" strips it.
    labeled = make_labeled_task()
    assert "reference" in labeled.model_dump()
    assert find_forbidden_keys(labeled.model_dump()) != []
