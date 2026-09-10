import json
from pathlib import Path

import pytest

from adaptive_math.training.upstreams import validate_manifest

REQUIRED = {"verl-agent", "verl", "sandboxfusion"}


def test_training_upstream_manifest_is_immutable_and_complete() -> None:
    manifest = json.loads(Path("third_party/manifest.json").read_text())

    assert set(manifest["upstreams"]) == REQUIRED
    for name, entry in manifest["upstreams"].items():
        assert entry["name"] == name
        assert entry["repository"].startswith("https://github.com/")
        assert len(entry["resolved_sha"]) == 40
        assert all(char in "0123456789abcdef" for char in entry["resolved_sha"])
        assert entry["license"]
        assert entry["resolved_at"].endswith("Z")
        assert entry["compatibility_notes"]


def test_manifest_rejects_movable_or_missing_sha() -> None:
    path = Path("third_party/manifest.json")
    manifest = json.loads(path.read_text())
    manifest["upstreams"]["verl"]["resolved_sha"] = "main"

    with pytest.raises(ValueError, match="resolved_sha"):
        validate_manifest(manifest)
