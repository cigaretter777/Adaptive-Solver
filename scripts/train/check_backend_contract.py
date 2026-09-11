"""Fail closed if the cloud image cannot provide the pinned verl-agent API."""

import argparse
import importlib
import inspect
import json
from pathlib import Path

from adaptive_math.training.upstreams import validate_manifest

_METHODS = ("reset", "step", "build_text_obs", "success_evaluator")


def _manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    validate_manifest(manifest)
    return manifest


def check_backend_contract(manifest_path: Path) -> dict[str, object]:
    manifest = _manifest(manifest_path)
    try:
        base_module = importlib.import_module("agent_system.environments.base")
        hgpo_module = importlib.import_module("recipe.hgpo.main_hgpo")
    except ImportError as exc:
        raise RuntimeError(
            "pinned verl-agent is not importable; install it from the resolved SHA before training"
        ) from exc
    base = getattr(base_module, "EnvironmentManagerBase", None)
    if not inspect.isclass(base):
        raise RuntimeError("pinned backend lacks EnvironmentManagerBase")
    missing = [name for name in _METHODS if not callable(getattr(base, name, None))]
    if missing:
        raise RuntimeError(f"EnvironmentManagerBase missing methods: {', '.join(missing)}")
    if not callable(getattr(hgpo_module, "run_ppo", None)):
        raise TypeError("pinned backend lacks recipe.hgpo.main_hgpo.run_ppo")
    return {
        "ok": True,
        "verl_agent_sha": manifest["upstreams"]["verl-agent"]["resolved_sha"],
        "environment_manager_methods": list(_METHODS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("third_party/manifest.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = _manifest(args.manifest) if args.dry_run else check_backend_contract(args.manifest)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
