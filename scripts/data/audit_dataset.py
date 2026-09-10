"""Audit a materialized math dataset manifest without accessing raw sources."""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.core.types import AnswerType, ReferenceAnswer
from adaptive_math.data import load_manifest
from adaptive_math.verifier import VerifierStatus, verify_answer

EXPECTED_COLUMNS = {
    "task_id",
    "problem",
    "answer_type",
    "dataset",
    "split",
    "source_hash",
    "pipeline_version",
    "metadata",
    "reference_value",
    "reference_acceptable_forms",
}
KNOWN_LICENSES = {
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "mit",
    "odc-by-1.0",
}


def _resolve_split_file(manifest_path: Path, name: str, data_root: Path | None) -> Path | None:
    # Resolution order: explicit --data-root, then the documented project
    # convention (<root>/manifests/<stem>.json + <root>/processed/<stem>/),
    # then the manifest's own directory, then an unambiguous recursive search.
    roots = []
    if data_root is not None:
        roots.append(data_root)
    roots.append(manifest_path.parent.parent / "processed" / manifest_path.stem)
    roots.append(manifest_path.parent)
    for root in roots:
        candidate = root / name
        if candidate.exists():
            return candidate
    candidates = sorted(manifest_path.parent.rglob(name))
    return candidates[0] if len(candidates) == 1 else None


def audit(manifest_path: Path, data_root: Path | None = None) -> list[str]:
    manifest = load_manifest(manifest_path)
    errors: list[str] = []
    if manifest.schema_version != "1.0":
        errors.append(f"unsupported schema_version: {manifest.schema_version}")
    for name, source in manifest.sources.items():
        if source.license.lower() not in KNOWN_LICENSES:
            errors.append(f"unknown license for source {name}: {source.license}")

    seen_task_ids: set[str] = set()
    split_task_ids: dict[str, set[str]] = {}
    for split_name, info in manifest.splits.items():
        path = _resolve_split_file(manifest_path, info.file, data_root)
        if path is None:
            errors.append(f"cannot resolve split file for {split_name}: {info.file}")
            continue
        contents = path.read_bytes()
        if sha256_hex(contents) != info.file_hash:
            errors.append(f"file hash mismatch: {split_name}")
            continue
        table = pq.read_table(path)
        if set(table.column_names) != EXPECTED_COLUMNS:
            errors.append(f"schema mismatch: {split_name}")
            continue
        rows = table.to_pylist()
        ids = [str(row["task_id"]) for row in rows]
        if len(ids) != info.count:
            errors.append(f"count mismatch: {split_name}")
        if sha256_hex("\n".join(ids).encode()) != info.task_ids_hash:
            errors.append(f"task id hash mismatch: {split_name}")
        duplicates = set(ids) & seen_task_ids
        if duplicates:
            errors.append(f"duplicate task IDs across splits: {sorted(duplicates)!r}")
        seen_task_ids.update(ids)
        split_task_ids[split_name] = set(ids)
        for row in rows:
            try:
                reference = ReferenceAnswer(
                    value=str(row["reference_value"]),
                    answer_type=AnswerType(str(row["answer_type"])),
                    acceptable_forms=tuple(row["reference_acceptable_forms"]),
                )
            except (TypeError, ValueError):
                errors.append(f"invalid reference schema: {split_name}/{row['task_id']}")
                continue
            result = verify_answer(reference.value, reference, task_id=str(row["task_id"]))
            if result.status is not VerifierStatus.CORRECT:
                errors.append(f"unverifiable reference: {split_name}/{row['task_id']}")

    frozen = split_task_ids.get("frozen_eval", set())
    training = set().union(*(split_task_ids.get(name, set()) for name in ("train", "sft_dev", "rl_dev")))
    if frozen & training:
        errors.append("frozen_eval leakage into training splits")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a materialized math dataset")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="directory containing the split parquet files (default: project convention)",
    )
    args = parser.parse_args(argv)
    errors = audit(Path(args.manifest), args.data_root)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
