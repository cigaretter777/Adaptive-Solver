"""Summarize and compare per-row DIRECT SFT materialization diagnostics."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("source_hash"), str):
            raise TypeError(f"{path}:{line_number} is not a diagnostic row")
        if not isinstance(row.get("outcome"), str):
            raise TypeError(f"{path}:{line_number} has no string outcome")
        rows.append(row)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    outcomes = Counter(str(row["outcome"]) for row in rows)
    accepted = sum(count for outcome, count in outcomes.items() if outcome.startswith("accepted_"))
    examples: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        outcome = str(row["outcome"])
        examples.setdefault(outcome, [])
        if len(examples[outcome]) < 3:
            examples[outcome].append(
                {
                    key: row.get(key)
                    for key in ("task_id", "source_answer", "solution_tail", "detail")
                }
            )
    return {
        "rows": len(rows),
        "accepted": accepted,
        "acceptance_rate": accepted / len(rows) if rows else 0.0,
        "outcomes": dict(sorted(outcomes.items())),
        "examples": dict(sorted(examples.items())),
    }


def _comparison(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, object]:
    baseline_by_hash = {str(row["source_hash"]): row for row in baseline}
    candidate_by_hash = {str(row["source_hash"]): row for row in candidate}
    common = sorted(set(baseline_by_hash) & set(candidate_by_hash))
    changed = [
        source_hash
        for source_hash in common
        if baseline_by_hash[source_hash]["outcome"] != candidate_by_hash[source_hash]["outcome"]
    ]
    baseline_accepted = {
        row.get("task_id")
        for row in baseline
        if str(row["outcome"]).startswith("accepted_") and row.get("task_id") is not None
    }
    candidate_accepted = {
        row.get("task_id")
        for row in candidate
        if str(row["outcome"]).startswith("accepted_") and row.get("task_id") is not None
    }
    return {
        "baseline_rows": len(baseline),
        "candidate_rows": len(candidate),
        "common_source_hashes": len(common),
        "only_in_baseline": len(set(baseline_by_hash) - set(candidate_by_hash)),
        "only_in_candidate": len(set(candidate_by_hash) - set(baseline_by_hash)),
        "changed_outcomes": len(changed),
        "changed_source_hashes": changed[:20],
        "accepted_task_ids_match": baseline_accepted == candidate_accepted,
        "baseline_accepted": len(baseline_accepted),
        "candidate_accepted": len(candidate_accepted),
    }


def _source_input_comparison(baseline_path: Path, candidate_path: Path) -> dict[str, object]:
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    fields = (
        "source",
        "requested_revision",
        "resolved_revision",
        "raw_records_sha256",
        "raw_record_count",
        "mode",
    )
    differences = {
        field: {"baseline": baseline.get(field), "candidate": candidate.get(field)}
        for field in fields
        if baseline.get(field) != candidate.get(field)
    }
    return {"matches": not differences, "differences": differences}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--source-input", type=Path)
    parser.add_argument("--compare-source-input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    diagnostics = _read_jsonl(args.diagnostics)
    payload: dict[str, object] = {"summary": _summary(diagnostics)}
    if args.compare is not None:
        payload["comparison"] = _comparison(diagnostics, _read_jsonl(args.compare))
    if (args.source_input is None) != (args.compare_source_input is None):
        parser.error("--source-input and --compare-source-input must be supplied together")
    if args.source_input is not None and args.compare_source_input is not None:
        payload["source_input_comparison"] = _source_input_comparison(
            args.source_input, args.compare_source_input
        )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
