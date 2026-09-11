import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from adaptive_math.training.sft_materialization import (
    export_labeled_task_pool,
    load_budget_file,
    read_task_pool_manifest,
)


def test_export_labeled_task_pool_writes_deterministic_private_input_artifact(tmp_path: Path) -> None:
    source = tmp_path / "train.parquet"
    pq.write_table(
        pa.table(
            {
                "task_id": ["unit:2", "unit:1"],
                "problem": ["2+2", "1+1"],
                "answer_type": ["integer", "integer"],
                "dataset": ["unit", "unit"],
                "split": ["train", "train"],
                "source_hash": ["b" * 64, "a" * 64],
                "pipeline_version": ["v1", "v1"],
                "metadata": ["{}", "{}"],
                "reference_value": ["4", "2"],
                "reference_acceptable_forms": [[], []],
            }
        ),
        source,
    )
    output = tmp_path / "teacher_input.jsonl"
    manifest_path = tmp_path / "teacher_input.manifest.json"

    manifest = export_labeled_task_pool(source, output, manifest_path, limit=1)

    row = json.loads(output.read_text())
    assert row["task"]["task_id"] == "unit:1"
    assert row["reference"]["value"] == "2"
    assert manifest.task_count == 1
    assert manifest.output_sha256
    assert read_task_pool_manifest(manifest_path) == manifest


def test_export_labeled_task_pool_rejects_non_labeled_schema(tmp_path: Path) -> None:
    source = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"task_id": ["unit:1"]}), source)

    try:
        export_labeled_task_pool(source, tmp_path / "out.jsonl", tmp_path / "out.json")
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("expected an incomplete parquet schema to be rejected")


def test_load_budget_file_accepts_the_agent_yaml_contract(tmp_path: Path) -> None:
    budget_file = tmp_path / "budget.yaml"
    budget_file.write_text(
        "max_steps: 2\nmax_tool_calls: 1\nmax_python_seconds: 3\nmax_observation_chars: 100\n"
    )

    budget = load_budget_file(budget_file)

    assert budget.max_steps == 2
    assert budget.max_python_seconds == 3
