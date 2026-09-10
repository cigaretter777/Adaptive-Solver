"""Generate verifier-correct public teacher traces for later SFT conversion."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from adaptive_math.agent.model_client import GenerationConfig
from adaptive_math.agent.replay import TraceEnvelope
from adaptive_math.agent.transformers_client import TransformersModelClient
from adaptive_math.core.types import Budget, LabeledMathTask
from adaptive_math.tools.python_tool import PythonTool
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.tools.sandboxfusion import SandboxFusionClient
from adaptive_math.tools.sympy_tool import SympyTool
from adaptive_math.training.teacher_rollout import collect_verified_rollouts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-tasks-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget", type=Path, default=Path("configs/agent/default.yaml"))
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    tasks = [
        LabeledMathTask.model_validate(json.loads(line))
        for line in args.labeled_tasks_jsonl.read_text().splitlines()
        if line.strip()
    ]
    budget = Budget.model_validate(yaml.safe_load(args.budget.read_text()))
    registry = ToolRegistry([SympyTool(), PythonTool(SandboxFusionClient())])
    model = TransformersModelClient.from_pretrained(args.model)
    verified = asyncio.run(
        collect_verified_rollouts(
            tasks,
            model,
            budget,
            registry,
            GenerationConfig(max_new_tokens=args.max_new_tokens),
            max_concurrency=args.max_concurrency,
        )
    )
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            TraceEnvelope.create(
                item.trajectory, prompt_version="agent-v1", created_at=created_at
            ).model_dump_json()
            + "\n"
            for item in verified
        )
    )
    print(json.dumps({"requested": len(tasks), "verified": len(verified), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
