import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
_spec = importlib.util.spec_from_file_location(
    "cloud_preflight", REPO_ROOT / "scripts" / "cloud" / "preflight.py"
)
cloud_preflight = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(cloud_preflight)


def test_preflight_dry_run_validates_immutable_inputs_without_gpu(tmp_path: Path) -> None:
    manifest = tmp_path / "tasks.manifest.json"
    manifest.write_text(json.dumps({"schema_version": "sft-task-pool-v1"}))
    upstreams = tmp_path / "upstreams.json"
    upstreams.write_text((REPO_ROOT / "third_party" / "manifest.json").read_text())

    report = cloud_preflight.preflight(
        task_manifest=manifest,
        upstream_manifest=upstreams,
        require_gpu=False,
        require_sandbox=False,
    )

    assert report["ok"] is True
    assert report["inputs"]["task_manifest"]["sha256"]
    assert report["gpu"]["required"] is False


def test_preflight_rejects_gpu_required_without_nvidia_smi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "tasks.manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(cloud_preflight.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="nvidia-smi"):
        cloud_preflight.preflight(
            task_manifest=manifest,
            upstream_manifest=REPO_ROOT / "third_party" / "manifest.json",
            require_gpu=True,
            require_sandbox=False,
        )
