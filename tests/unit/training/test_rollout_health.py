from adaptive_math.training.rollout_health import (
    row_to_labeled_task,
    summarize_group_rewards,
)


def test_row_to_labeled_task_builds_hidden_reference() -> None:
    row = {
        "task_id": "demo:1",
        "problem": "What is 2+2?",
        "answer_type": "integer",
        "dataset": "demo",
        "split": "rl_dev",
        "source_hash": "abc",
        "pipeline_version": "test-v1",
        "metadata": '{"source_index":"1"}',
        "reference_value": "4",
        "reference_acceptable_forms": ["4.0"],
    }

    labeled = row_to_labeled_task(row)

    assert labeled.task.problem == "What is 2+2?"
    assert labeled.task.metadata == {"source_index": "1"}
    assert labeled.reference.value == "4"
    assert labeled.reference.acceptable_forms == ("4.0",)


def test_summarize_group_rewards_distinguishes_effective_groups() -> None:
    groups = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    summary = summarize_group_rewards(groups)

    assert summary["group_count"] == 3
    assert summary["mixed_group_count"] == 1
    assert summary["all_zero_group_count"] == 1
    assert summary["all_one_group_count"] == 1
    assert summary["effective_group_count"] == 1


def test_select_task_rows_is_deterministic_without_replacement() -> None:
    from adaptive_math.training.rollout_health import select_task_rows

    rows = [
        {"task_id": f"task-{i}"}
        for i in range(20)
    ]

    first = select_task_rows(rows, count=10, seed=42)
    second = select_task_rows(rows, count=10, seed=42)

    first_ids = [row["task_id"] for row in first]
    second_ids = [row["task_id"] for row in second]

    assert first_ids == second_ids
    assert len(first_ids) == 10
    assert len(set(first_ids)) == 10


def test_normalize_acceptable_forms_accepts_numpy_arrays() -> None:
    import numpy as np

    from adaptive_math.training.rollout_health import _normalize_acceptable_forms

    assert _normalize_acceptable_forms(
        np.asarray([], dtype=object)
    ) == ()

    assert _normalize_acceptable_forms(
        np.asarray(["4.0", "4"], dtype=object)
    ) == ("4.0", "4")
