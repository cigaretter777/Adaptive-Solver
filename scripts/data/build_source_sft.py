"""Materialize verified DIRECT or Python-TIR SFT records from one pinned source."""

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.data.sources import SourceRegistry, load_source_records
from adaptive_math.tools.python_tool import PythonTool
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.tools.sandboxfusion import SandboxFusionClient
from adaptive_math.training.sft_io import write_records, write_sft_manifest
from adaptive_math.training.sft_materialization import load_budget_file
from adaptive_math.training.solution_traces import (
    DirectRecordDiagnostic,
    materialize_direct_records_batched,
    materialize_python_tir_records,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_diagnostics(path: Path, diagnostics: tuple[DirectRecordDiagnostic, ...]) -> None:
    """Write one disposition per source row before enforcing non-empty output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(item.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in diagnostics
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/data/sources.yaml"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=("direct", "python_tir"), required=True)
    parser.add_argument("--budget", type=Path, default=Path("configs/agent/default.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--traces-output", type=Path, required=True)
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=None,
        help="optional JSONL: one acceptance/rejection disposition per raw source row",
    )
    parser.add_argument("--tokenizer-revision", default="unresolved")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    registry = SourceRegistry.model_validate(yaml.safe_load(args.registry.read_text()))
    spec = registry.sources.get(args.source)
    if spec is None:
        parser.error(f"unknown source {args.source!r}; choose one of {', '.join(sorted(registry.sources))}")
    if spec.intended_use != "train":
        parser.error(f"source {spec.name!r} is not approved for training")
    raw_records, resolved_revision = load_source_records(spec, sample=args.limit)
    raw_json = json.dumps(raw_records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    input_manifest_path = args.manifest.with_name(args.manifest.stem + ".source-input.json")
    _write_json(
        input_manifest_path,
        {
            "schema_version": "source-sft-input-v1",
            "source": spec.name,
            "requested_revision": spec.revision,
            "resolved_revision": resolved_revision,
            "raw_records_sha256": sha256_hex(raw_json),
            "raw_record_count": len(raw_records),
            "mode": args.mode,
        },
    )
    budget = load_budget_file(args.budget)
    if args.mode == "direct":
        result = materialize_direct_records_batched(
            spec,
            raw_records,
            budget,
            ToolRegistry([]),
            batch_size=args.batch_size,
            tokenizer_revision=args.tokenizer_revision,
        )
    else:
        result = asyncio.run(
            materialize_python_tir_records(
                spec,
                raw_records,
                budget,
                ToolRegistry([PythonTool(SandboxFusionClient())]),
                tokenizer_revision=args.tokenizer_revision,
            )
        )
    if args.diagnostics_output is not None:
        _write_diagnostics(args.diagnostics_output, result.diagnostics)
    if not result.records:
        parser.error("no verifier-correct records were materialized")
    args.traces_output.parent.mkdir(parents=True, exist_ok=True)
    args.traces_output.write_text(
        "".join(trace.model_dump_json() + "\n" for trace in result.traces)
    )
    write_records(list(result.records), args.output)
    manifest = write_sft_manifest(
        args.output,
        args.manifest,
        source_task_pool_manifest_sha256=sha256_hex(input_manifest_path.read_bytes()),
        source_trace_sha256=sha256_hex(args.traces_output.read_bytes()),
    )
    print(
        json.dumps(
            {
                "source_input_manifest": str(input_manifest_path),
                "sft_manifest": manifest.model_dump(),
                "rejected": result.rejected,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
