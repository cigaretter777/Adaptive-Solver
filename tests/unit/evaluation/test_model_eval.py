"""Paired direct-model evaluation uses real extraction and verification."""

from pathlib import Path

import pytest

from adaptive_math.agent.model_client import ModelTurn
from adaptive_math.core.types import AnswerType, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.evaluation.model_eval import evaluate_pair, write_artifacts


def task(task_id: str, answer: str) -> LabeledMathTask:
    return LabeledMathTask(
        task=MathTask(
            task_id=task_id, problem=f"Solve {task_id}", answer_type=AnswerType.INTEGER,
            dataset="omni_math", split="frozen_eval", source_hash="a" * 64,
            pipeline_version="test",
        ),
        reference=ReferenceAnswer(value=answer, answer_type=AnswerType.INTEGER),
    )


def test_paired_results_use_same_tasks_and_report_improvement_and_regression(tmp_path: Path) -> None:
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    def generate(arm: str, item: MathTask, messages: tuple) -> ModelTurn:
        assert not hasattr(item, "reference")
        seen.append((arm, item.task_id, tuple(message.content for message in messages)))
        answer = {
            ("base", "a"): "0", ("sft", "a"): "1",
            ("base", "b"): "2", ("sft", "b"): "0",
        }[(arm, item.task_id)]
        return ModelTurn(text=f'<final>{{"answer":"{answer}"}}</final>',
                         prompt_tokens=8, generated_tokens=4, finish_reason="stop", model_id=arm)

    result = evaluate_pair([task("b", "2"), task("a", "1")], set(), generate)
    assert [row["task_id"] for row in result["base_predictions"]] == ["a", "b"]
    assert [row["task_id"] for row in result["sft_predictions"]] == ["a", "b"]
    assert [row["outcome"] for row in result["comparison"]] == ["improved", "regressed"]
    assert result["summary"]["base"]["verifier_accuracy"] == 0.5
    assert result["summary"]["sft"]["verifier_accuracy"] == 0.5
    assert result["summary"]["accuracy_delta"] == 0.0
    assert result["summary"]["mcnemar_pvalue"] == 1.0
    assert result["summary"]["paired_bootstrap_ci95"] == [-1.0, 1.0]
    assert seen[0][2] == seen[2][2]
    write_artifacts(tmp_path / "result", result, {"git_sha": "test"})
    assert (tmp_path / "result" / "COMPLETE").is_file()
    assert len((tmp_path / "result" / "comparison.jsonl").read_text().splitlines()) == 2


def test_overlap_fails_before_any_generation() -> None:
    def forbidden(*_args):
        raise AssertionError("generation must not start")

    with pytest.raises(ValueError, match="leakage"):
        evaluate_pair([task("a", "1")], {"a"}, forbidden)


def test_invalid_answer_is_counted_separately() -> None:
    def generate(_arm: str, _item: MathTask, _messages: tuple) -> ModelTurn:
        return ModelTurn(text="no terminal answer", prompt_tokens=1, generated_tokens=3,
                         finish_reason="stop", model_id="test")

    result = evaluate_pair([task("a", "1")], set(), generate)
    assert result["summary"]["base"]["invalid_prediction"] == 1
    assert result["summary"]["base"]["valid_answer_rate"] == 0.0
