"""Validate immutable cloud-training inputs before renting a GPU."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.training.upstreams import validate_manifest


def _file_evidence(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"required file does not exist: {path}")
    return {"path": str(path), "sha256": sha256_hex(path.read_bytes())}


def _gpu_evidence(required: bool) -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not required:
        return {"required": False, "nvidia_smi": nvidia_smi}
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required for a GPU training run")
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"nvidia-smi did not report a usable GPU: {result.stderr.strip()}")
    return {"required": True, "devices": result.stdout.splitlines()}


def preflight(
    *,
    task_manifest: Path,
    upstream_manifest: Path,
    require_gpu: bool,
    require_sandbox: bool,
) -> dict[str, Any]:
    """Return evidence or fail before model loading and cloud cost are incurred."""
    task_evidence = _file_evidence(task_manifest)
    upstream_evidence = _file_evidence(upstream_manifest)
    try:
        validate_manifest(json.loads(upstream_manifest.read_text()))
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid upstream manifest: {exc}") from exc
    if require_sandbox and not os.environ.get("ADAPTIVE_MATH_SANDBOX_URL"):
        raise RuntimeError("ADAPTIVE_MATH_SANDBOX_URL is required when sandbox checks are enabled")
    return {
        "ok": True,
        "inputs": {"task_manifest": task_evidence, "upstream_manifest": upstream_evidence},
        "gpu": _gpu_evidence(require_gpu),
        "sandbox": {"required": require_sandbox, "url_configured": bool(require_sandbox)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--upstream-manifest", type=Path, default=Path("third_party/manifest.json"))
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-sandbox", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            preflight(
                task_manifest=args.task_manifest,
                upstream_manifest=args.upstream_manifest,
                require_gpu=args.require_gpu,
                require_sandbox=args.require_sandbox,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
