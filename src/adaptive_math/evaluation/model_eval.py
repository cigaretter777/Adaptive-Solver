"""Deterministic paired, direct-answer Base versus SFT evaluation."""

import json
import math
import os
import random
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

from adaptive_math.agent.model_client import ChatMessage, ModelTurn
from adaptive_math.agent.prompts import PROMPT_VERSION, render_initial_messages
from adaptive_math.core.hashing import sha256_hex
from adaptive_math.core.types import Budget, LabeledMathTask, MathTask
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.verifier import ExtractStatus, VerifierStatus, extract, verify_answer

Generator = Callable[[str, MathTask, tuple[ChatMessage, ...]], ModelTurn]


class EvalResult(TypedDict):
    base_predictions: list[dict[str, object]]
    sft_predictions: list[dict[str, object]]
    comparison: list[dict[str, object]]
    summary: dict[str, object]


_DIRECT_BUDGET = Budget(max_steps=1, max_tool_calls=0, max_python_seconds=0,
                        max_observation_chars=1)


class EvaluationJournal:
    """Append-only, manifest-bound progress state for an interruptible evaluation."""

    def __init__(self, directory: Path, immutable_manifest: dict[str, object]) -> None:
        self.directory = directory
        self._manifest_path = directory / "run_manifest.json"
        self._progress_path = directory / "progress.json"
        if self._manifest_path.exists():
            stored = json.loads(self._manifest_path.read_text())
            if stored != immutable_manifest:
                raise ValueError("evaluation journal immutable manifest does not match")
        else:
            directory.mkdir(parents=True, exist_ok=False)
            _atomic_json(self._manifest_path, immutable_manifest)
            _atomic_json(self._progress_path, {"completed": {"base": 0, "sft": 0}})

    @property
    def progress(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self._progress_path.read_text()))

    def rows(self, arm: str) -> list[dict[str, object]]:
        path = self._rows_path(arm)
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        task_ids = [row.get("task_id") for row in rows]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"evaluation journal has duplicate {arm} task IDs")
        return cast(list[dict[str, object]], rows)

    def append(self, arm: str, row: dict[str, object]) -> None:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise TypeError("journal row requires a non-empty task_id")
        if task_id in {item["task_id"] for item in self.rows(arm)}:
            raise ValueError(f"evaluation journal already contains {arm}:{task_id}")
        path = self._rows_path(arm)
        with path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        progress = self.progress
        completed = progress.get("completed")
        if not isinstance(completed, dict):
            raise TypeError("evaluation journal progress is malformed")
        completed[arm] = len(self.rows(arm))
        _atomic_json(self._progress_path, progress)

    def _rows_path(self, arm: str) -> Path:
        if arm not in {"base", "sft"}:
            raise ValueError(f"unknown evaluation arm: {arm}")
        return self.directory / f"{arm}_predictions.jsonl"


def _atomic_json(path: Path, value: object) -> None:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(rendered)
    os.replace(temporary, path)


def evaluate_pair(
    tasks: list[LabeledMathTask],
    sft_task_ids: set[str],
    generate: Generator,
    *,
    initial_predictions: dict[str, list[dict[str, object]]] | None = None,
    on_prediction: Callable[[str, dict[str, object]], None] | None = None,
) -> EvalResult:
    """Evaluate both arms on the exact same ordered tasks and public prompts."""
    ordered = sorted(tasks, key=lambda item: item.task.task_id)
    ids = [item.task.task_id for item in ordered]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("evaluation requires nonempty, unique task IDs")
    if any(item.task.split != "frozen_eval" or item.task.dataset != "omni_math" for item in ordered):
        raise ValueError("evaluation requires Omni-MATH frozen_eval tasks")
    overlap = set(ids) & sft_task_ids
    if overlap:
        raise ValueError(f"SFT/eval task ID leakage: {sorted(overlap)[:5]}")
    prompts = {
        item.task.task_id: render_initial_messages(item.public_view(), _DIRECT_BUDGET, ToolRegistry([]))
        for item in ordered
    }
    initial_predictions = initial_predictions or {"base": [], "sft": []}
    predictions: dict[str, list[dict[str, object]]] = {}
    for arm in ("base", "sft"):
        existing = initial_predictions.get(arm, [])
        rows_by_task_id = {str(row["task_id"]): row for row in existing}
        if len(rows_by_task_id) != len(existing) or not set(rows_by_task_id).issubset(ids):
            raise ValueError(f"invalid resumed {arm} predictions")
        for item in ordered:
            if item.task.task_id in rows_by_task_id:
                continue
            turn = generate(arm, item.public_view(), prompts[item.task.task_id])
            extraction = extract(turn.text)
            verdict = (
                verify_answer(extraction.value, item.reference, task_id=item.task.task_id)
                if extraction.status is ExtractStatus.OK and extraction.value is not None
                else None
            )
            status = verdict.status.value if verdict else VerifierStatus.INVALID_PREDICTION.value
            row: dict[str, object] = {
                "task_id": item.task.task_id, "dataset": item.task.dataset,
                "split": item.task.split, "source_hash": item.task.source_hash,
                "prediction": extraction.value, "raw_output": turn.text,
                "extract_status": extraction.status.value, "verifier_status": status,
                "prompt_tokens": turn.prompt_tokens, "output_tokens": turn.generated_tokens,
                "finish_reason": turn.finish_reason,
            }
            rows_by_task_id[item.task.task_id] = row
            if on_prediction is not None:
                on_prediction(arm, row)
        predictions[arm] = [rows_by_task_id[task_id] for task_id in ids]
    comparison = []
    for base, sft in zip(predictions["base"], predictions["sft"], strict=True):
        base_correct = base["verifier_status"] == VerifierStatus.CORRECT.value
        sft_correct = sft["verifier_status"] == VerifierStatus.CORRECT.value
        outcome = "improved" if sft_correct and not base_correct else (
            "regressed" if base_correct and not sft_correct else "unchanged"
        )
        comparison.append({"task_id": base["task_id"], "base_correct": base_correct,
                           "sft_correct": sft_correct, "outcome": outcome})
    paired_statistics = _paired_statistics(comparison)
    return {
        "base_predictions": predictions["base"],
        "sft_predictions": predictions["sft"],
        "comparison": comparison,
        "summary": {
            "task_count": len(ids), "task_ids_sha256": sha256_hex("\n".join(ids).encode()),
            "prompt_version": PROMPT_VERSION,
            "base": _summarize(predictions["base"]),
            "sft": _summarize(predictions["sft"]),
            "paired": dict(Counter(row["outcome"] for row in comparison)),
            **paired_statistics,
        },
    }


def _paired_statistics(comparison: list[dict[str, object]]) -> dict[str, object]:
    """Task-level paired bootstrap interval and exact McNemar p-value."""
    deltas = [int(bool(row["sft_correct"])) - int(bool(row["base_correct"]))
              for row in comparison]
    n = len(deltas)
    rng = random.Random(20260913)
    samples = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(10_000))
    improved = deltas.count(1)
    regressed = deltas.count(-1)
    discordant = improved + regressed
    tail = sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
    pvalue = min(1.0, 2 * tail / 2**discordant)
    return {
        "accuracy_delta": sum(deltas) / n,
        "paired_bootstrap_ci95": [samples[249], samples[9749]],
        "bootstrap_seed": 20260913, "bootstrap_resamples": 10_000,
        "mcnemar_pvalue": pvalue,
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    statuses = Counter(str(row["verifier_status"]) for row in rows)
    n = len(rows)
    output_lengths = sorted(cast(int, row["output_tokens"]) for row in rows)
    valid = statuses[VerifierStatus.CORRECT.value] + statuses[VerifierStatus.INCORRECT.value]
    return {
        "total": n, "correct": statuses[VerifierStatus.CORRECT.value],
        "incorrect": statuses[VerifierStatus.INCORRECT.value],
        "invalid_prediction": statuses[VerifierStatus.INVALID_PREDICTION.value],
        "invalid_reference": statuses[VerifierStatus.INVALID_REFERENCE.value],
        "timeout": statuses[VerifierStatus.TIMEOUT.value],
        "internal_error": statuses[VerifierStatus.INTERNAL_ERROR.value],
        "verifier_accuracy": statuses[VerifierStatus.CORRECT.value] / n,
        "valid_answer_rate": valid / n,
        "mean_output_tokens": sum(output_lengths) / n,
        "median_output_tokens": (output_lengths[(n - 1) // 2] + output_lengths[n // 2]) / 2,
    }


def write_artifacts(output_dir: Path, result: EvalResult, manifest: dict[str, object]) -> None:
    """Publish complete results only after all records and hashes are written."""
    if output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = output_dir.with_name(output_dir.name + f".tmp-{os.getpid()}")
    if staged.exists():
        raise FileExistsError(f"staging directory already exists: {staged}")
    staged.mkdir()
    try:
        hashes = {}
        for key in ("base_predictions", "sft_predictions", "comparison"):
            payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in result[key]).encode()
            path = staged / f"{key}.jsonl"
            path.write_bytes(payload)
            hashes[path.name] = sha256_hex(payload)
        summary = json.dumps(result["summary"], ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        (staged / "summary.json").write_text(summary)
        hashes["summary.json"] = sha256_hex(summary.encode())
        manifest = {**manifest, "artifact_sha256": hashes,
                    "task_ids_sha256": result["summary"]["task_ids_sha256"]}
        (staged / "eval_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        (staged / "COMPLETE").write_text(sha256_hex((staged / "eval_manifest.json").read_bytes()) + "\n")
        os.replace(staged, output_dir)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
