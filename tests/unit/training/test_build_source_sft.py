import importlib.util
import json
from pathlib import Path

from adaptive_math.training.solution_traces import DirectRecordDiagnostic

REPO_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "build_source_sft", REPO_ROOT / "scripts" / "data" / "build_source_sft.py"
)
assert _spec is not None and _spec.loader is not None
build_source_sft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_source_sft)


def test_write_diagnostics_emits_one_json_object_per_source_row(tmp_path: Path) -> None:
    path = tmp_path / "diagnostics.jsonl"
    build_source_sft._write_diagnostics(
        path,
        (
            DirectRecordDiagnostic(
                source_hash="a" * 64,
                task_id="unit:1",
                outcome="accepted_strict",
                detail="",
                source_answer="2",
                solution_tail="Answer: 2",
            ),
        ),
    )

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {
            "detail": "",
            "outcome": "accepted_strict",
            "solution_tail": "Answer: 2",
            "source_answer": "2",
            "source_hash": "a" * 64,
            "task_id": "unit:1",
        }
    ]
