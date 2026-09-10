"""Deterministic dataset build: registry -> canonicalize -> verify ->
deduplicate -> split -> parquet + manifest.

Usage:
    uv run python scripts/data/build_dataset.py \
      --registry configs/data/sources.yaml \
      --output-dir data/processed/v1 \
      --manifest data/manifests/v1.json \
      --seed 20260910 \
      [--dry-run] [--sample-per-source N]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.data import (
    DataManifest,
    DedupManifest,
    QuarantineRecord,
    SourceManifest,
    SourceRegistry,
    SplitConfig,
    SplitManifest,
    assign_splits,
    canonicalize_source,
    deduplicate,
    load_source_records,
)
from adaptive_math.data.manifest import write_manifest

SCHEMA_VERSION = "1.0"
PARQUET_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("problem", pa.string()),
        ("answer_type", pa.string()),
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("source_hash", pa.string()),
        ("pipeline_version", pa.string()),
        ("metadata", pa.string()),
        ("reference_value", pa.string()),
        ("reference_acceptable_forms", pa.list_(pa.string())),
    ]
)


def _pipeline_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _task_to_row(task) -> dict[str, object]:
    serialized_task = task.task.model_dump(mode="json")
    return {
        "task_id": task.task.task_id,
        "problem": task.task.problem,
        "answer_type": task.task.answer_type.value,
        "dataset": task.task.dataset,
        "split": task.task.split,
        "source_hash": task.task.source_hash,
        "pipeline_version": task.task.pipeline_version,
        "metadata": json.dumps(serialized_task["metadata"], sort_keys=True),
        "reference_value": task.reference.value,
        "reference_acceptable_forms": list(task.reference.acceptable_forms),
    }


def _write_split(output_dir: Path, name: str, tasks: list) -> SplitManifest:
    rows = [_task_to_row(t) for t in sorted(tasks, key=lambda t: t.task.task_id)]
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    file_path = output_dir / f"{name}.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, file_path)
    task_ids = [r["task_id"] for r in rows]
    return SplitManifest(
        name=name,
        file=file_path.name,
        count=len(rows),
        task_ids_hash=sha256_hex("\n".join(task_ids).encode()),
        file_hash=sha256_hex(file_path.read_bytes()),
    )


def build(registry: SourceRegistry, output_dir: Path, manifest_path: Path, seed: int, sample: int | None, dry_run: bool) -> DataManifest:
    sources: dict[str, SourceManifest] = {}
    all_tasks = []
    all_quarantined: list[QuarantineRecord] = []
    for name, spec in registry.sources.items():
        records, resolved_revision = load_source_records(spec, sample)
        kept, quarantined = canonicalize_source(spec, records)
        resolved = resolved_revision or (
            spec.revision if len(spec.revision) == 40 else None
        )
        sources[name] = SourceManifest(
            name=spec.name,
            uri=spec.uri,
            revision=spec.revision,
            resolved_revision=resolved,
            license=spec.license,
            intended_use=spec.intended_use,
            citation=spec.citation,
            answer_type=spec.answer_type.value,
            loaded_count=len(records),
            kept_count=len(kept),
            quarantined_count=len(quarantined),
        )
        all_tasks.extend(kept)
        all_quarantined.extend(quarantined)

    dedup_result = deduplicate(
        all_tasks, seed=seed, eval_datasets=registry.eval_source_names()
    )
    splits = assign_splits(dedup_result.kept, registry, seed=seed, config=SplitConfig())

    manifest = DataManifest(
        pipeline_version=_pipeline_version(),
        seed=seed,
        sources=sources,
        splits={},
        dedup=DedupManifest(
            exact_removed=dedup_result.exact_removed,
            near_removed=dedup_result.near_removed,
        ),
        quarantine_total=len(all_quarantined),
    )
    if dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "source_counts": {
                        name: {"loaded": info.loaded_count, "kept": info.kept_count}
                        for name, info in sources.items()
                    },
                    "split_counts": {name: len(members) for name, members in splits.items()},
                    "dedup": manifest.dedup.model_dump(),
                    "quarantine_total": manifest.quarantine_total,
                },
                indent=2,
            )
        )
        return manifest

    for name, members in splits.items():
        manifest.splits[name] = _write_split(output_dir, name, members)
    quarantine_path = output_dir / "quarantine.jsonl"
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_text(
        "\n".join(
            json.dumps(q.model_dump(), sort_keys=True) for q in all_quarantined
        )
    )
    write_manifest(manifest, manifest_path)
    print(
        json.dumps(
            {
                "split_counts": {name: info.count for name, info in manifest.splits.items()},
                "dedup": manifest.dedup.model_dump(),
                "quarantine_total": manifest.quarantine_total,
            },
            indent=2,
        )
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic dataset build")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--sample-per-source", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    registry = SourceRegistry.model_validate(
        yaml.safe_load(Path(args.registry).read_text())
    )
    build(
        registry,
        Path(args.output_dir),
        Path(args.manifest),
        args.seed,
        args.sample_per_source,
        args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
