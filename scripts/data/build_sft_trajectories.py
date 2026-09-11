"""Build verified SFT Parquet records from labeled tasks and runtime trace JSONL."""

import argparse
import json
from pathlib import Path

from adaptive_math.agent.replay import TraceEnvelope
from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.hashing import sha256_hex
from adaptive_math.core.types import LabeledMathTask
from adaptive_math.tools.python_tool import PythonTool
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.tools.sandboxfusion import SandboxFusionClient
from adaptive_math.tools.sympy_tool import SympyTool
from adaptive_math.training.sft_builder import build_sft_record
from adaptive_math.training.sft_io import audit_records, write_records, write_sft_manifest
from adaptive_math.training.sft_materialization import load_budget_file


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-tasks-jsonl", type=Path, required=True)
    parser.add_argument("--traces-jsonl", type=Path, required=True)
    parser.add_argument(
        "--budget",
        type=Path,
        required=True,
        help="shared Agent budget YAML (the same contract used for teacher rollout)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-pool-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", default="unresolved")
    args = parser.parse_args()
    tasks = {
        task.task.task_id: task
        for row in _jsonl(args.labeled_tasks_jsonl)
        for task in [LabeledMathTask.model_validate(row)]
    }
    budget = load_budget_file(args.budget)
    traces: list[Trajectory] = []
    for row in _jsonl(args.traces_jsonl):
        traces.append(TraceEnvelope.model_validate(row).trajectory if "trajectory" in row else Trajectory.model_validate(row))
    registry = ToolRegistry([SympyTool(), PythonTool(SandboxFusionClient())])
    records = [
        build_sft_record(tasks[trace.task_id], trace, budget, registry, tokenizer_revision=args.tokenizer_revision)
        for trace in traces
    ]
    audit = audit_records(records)
    write_records(records, args.output)
    manifest = write_sft_manifest(
        args.output,
        args.manifest,
        source_task_pool_manifest_sha256=sha256_hex(args.task_pool_manifest.read_bytes()),
        source_trace_sha256=sha256_hex(args.traces_jsonl.read_bytes()),
    )
    print(json.dumps({"audit": audit.model_dump(), "manifest": manifest.model_dump()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
