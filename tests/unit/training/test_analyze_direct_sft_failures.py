import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "data" / "analyze_direct_sft_failures.py"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_analyzer_reports_outcomes_and_compares_row_level_dispositions(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    baseline_input = tmp_path / "baseline.source-input.json"
    candidate_input = tmp_path / "candidate.source-input.json"
    _write_jsonl(
        baseline,
        [
            {"source_hash": "a", "task_id": "t1", "outcome": "accepted_strict"},
            {"source_hash": "b", "task_id": "t2", "outcome": "terminal_extract_missing"},
        ],
    )
    _write_jsonl(
        candidate,
        [
            {"source_hash": "a", "task_id": "t1", "outcome": "accepted_strict"},
            {"source_hash": "b", "task_id": "t2", "outcome": "strict_verifier_incorrect"},
        ],
    )
    source_input = {
        "source": "openr1_math_220k",
        "resolved_revision": "a" * 40,
        "raw_records_sha256": "b" * 64,
        "raw_record_count": 2,
        "mode": "direct",
    }
    baseline_input.write_text(json.dumps(source_input))
    candidate_input.write_text(json.dumps(source_input))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--diagnostics",
            str(baseline),
            "--compare",
            str(candidate),
            "--source-input",
            str(baseline_input),
            "--compare-source-input",
            str(candidate_input),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["summary"]["outcomes"] == {
        "accepted_strict": 1,
        "terminal_extract_missing": 1,
    }
    assert payload["comparison"]["changed_outcomes"] == 1
    assert payload["comparison"]["accepted_task_ids_match"] is True
    assert payload["source_input_comparison"]["matches"] is True
    assert payload["candidate_summary"]["outcomes"] == {
        "accepted_strict": 1,
        "strict_verifier_incorrect": 1,
    }
