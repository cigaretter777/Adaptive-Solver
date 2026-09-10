import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "data" / "build_dataset.py"
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "data" / "audit_dataset.py"


@pytest.fixture
def synthetic_env(tmp_path: Path) -> dict[str, Path]:
    near_train_problem = (
        "Let a sequence start at 2 and double at each of the next five steps. "
        "What integer is obtained after the fifth step?"
    )
    near_eval_problem = near_train_problem.removesuffix("?") + "!"
    train_records = [{"problem": f"What is the value of {i}+{i}?", "answer": str(2 * i)} for i in range(29)]
    train_records.append({"problem": near_train_problem, "answer": "64"})
    train_records.append({"problem": "What is the value of 1+1?", "answer": "2"})  # exact dup
    train_records.append({"problem": "nope", "answer": "abc"})  # unverifiable
    train_records.append({"problem": None, "answer": "2"})  # missing problem
    train_path = tmp_path / "train_records.jsonl"
    train_path.write_text("\n".join(json.dumps(r) for r in train_records))

    eval_records = [{"problem": f"What is the value of {i}*{i}?", "answer": str(i * i)} for i in range(9)]
    eval_records.append({"problem": near_eval_problem, "answer": "64"})  # cross-split 5-gram near-dup of a train problem
    eval_path = tmp_path / "eval_records.jsonl"
    eval_path.write_text("\n".join(json.dumps(r) for r in eval_records))

    registry = {
        "version": "1.0",
        "sources": {
            "syn_train": {
                "name": "syn_train",
                "uri": f"file://{train_path}",
                "revision": "a" * 40,
                "license": "apache-2.0",
                "intended_use": "train",
                "citation": "synthetic train source",
                "loader": "synthetic",
                "loader_params": {"problem_column": "problem", "answer_column": "answer"},
                "answer_type": "integer",
            },
            "syn_eval": {
                "name": "syn_eval",
                "uri": f"file://{eval_path}",
                "revision": "b" * 40,
                "license": "apache-2.0",
                "intended_use": "eval",
                "citation": "synthetic eval source",
                "loader": "synthetic",
                "loader_params": {"problem_column": "problem", "answer_column": "answer"},
                "answer_type": "integer",
            },
        },
    }
    registry_path = tmp_path / "registry.yaml"
    import yaml

    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False))
    return {
        "registry": registry_path,
        "output_dir": tmp_path / "processed",
        "manifest": tmp_path / "manifest.json",
    }


def run_cli(script: Path, *args: str, expect_ok: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    if expect_ok and result.returncode != 0:
        raise AssertionError(f"CLI failed ({result.returncode}): {result.stderr}")
    return result


def test_build_audit_and_byte_identical_rebuild(synthetic_env: dict[str, Path]) -> None:
    common = [
        "--registry",
        str(synthetic_env["registry"]),
        "--output-dir",
        str(synthetic_env["output_dir"]),
        "--manifest",
        str(synthetic_env["manifest"]),
        "--seed",
        "42",
    ]
    run_cli(BUILD_SCRIPT, *common)

    # manifest and split parquet files exist
    manifest = json.loads(synthetic_env["manifest"].read_text())
    assert manifest["seed"] == 42
    assert manifest["schema_version"] == "1.0"
    for split_info in manifest["splits"].values():
        parquet = synthetic_env["output_dir"] / split_info["file"]
        assert parquet.exists()
    quarantine = synthetic_env["output_dir"] / "quarantine.jsonl"
    assert quarantine.exists()
    quarantine_rows = [json.loads(l) for l in quarantine.read_text().splitlines()]
    assert sum(1 for r in quarantine_rows if r["reason"] == "unverifiable_reference") == 1
    assert sum(1 for r in quarantine_rows if r["reason"] == "missing_problem") == 1

    # exact dedup happened: 30 + 1 dup - 1 = 30 unique train-source records
    # near-dup across splits removed the train member: 30 - 1 = 29 train-source tasks
    train_count = manifest["splits"]["train"]["count"] + manifest["splits"]["sft_dev"]["count"] + manifest["splits"]["rl_dev"]["count"]
    assert train_count == 29
    assert manifest["splits"]["frozen_eval"]["count"] == 10
    assert manifest["dedup"]["exact_removed"] == 1
    assert manifest["dedup"]["near_removed"] == 1

    # audit passes
    run_cli(AUDIT_SCRIPT, "--manifest", str(synthetic_env["manifest"]))

    # rebuild with the same seed is byte-identical
    first_parquet = {name: (synthetic_env["output_dir"] / info["file"]).read_bytes() for name, info in manifest["splits"].items()}
    rebuild_dir = synthetic_env["output_dir"].parent / "processed_rebuild"
    rebuild_manifest = synthetic_env["manifest"].parent / "manifest_rebuild.json"
    run_cli(
        BUILD_SCRIPT,
        "--registry",
        str(synthetic_env["registry"]),
        "--output-dir",
        str(rebuild_dir),
        "--manifest",
        str(rebuild_manifest),
        "--seed",
        "42",
    )
    rebuild = json.loads(rebuild_manifest.read_text())
    for name, info in rebuild["splits"].items():
        assert (rebuild_dir / info["file"]).read_bytes() == first_parquet[name]
    assert rebuild == manifest


def test_dry_run_writes_nothing(synthetic_env: dict[str, Path]) -> None:
    run_cli(
        BUILD_SCRIPT,
        "--registry",
        str(synthetic_env["registry"]),
        "--output-dir",
        str(synthetic_env["output_dir"]),
        "--manifest",
        str(synthetic_env["manifest"]),
        "--seed",
        "42",
        "--dry-run",
    )
    assert not synthetic_env["output_dir"].exists()
    assert not synthetic_env["manifest"].exists()


def test_audit_detects_tampered_split(synthetic_env: dict[str, Path]) -> None:
    common = [
        "--registry",
        str(synthetic_env["registry"]),
        "--output-dir",
        str(synthetic_env["output_dir"]),
        "--manifest",
        str(synthetic_env["manifest"]),
        "--seed",
        "42",
    ]
    run_cli(BUILD_SCRIPT, *common)
    manifest = json.loads(synthetic_env["manifest"].read_text())
    parquet = synthetic_env["output_dir"] / manifest["splits"]["train"]["file"]
    original = parquet.read_bytes()
    parquet.write_bytes(original + b"tampered")
    result = run_cli(AUDIT_SCRIPT, "--manifest", str(synthetic_env["manifest"]), expect_ok=False)
    assert result.returncode != 0
