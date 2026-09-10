"""Validation for immutable third-party training backend pins."""

import re

SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_NAMES = {"verl-agent", "verl", "sandboxfusion"}
REQUIRED_FIELDS = {
    "name",
    "repository",
    "requested_ref",
    "resolved_sha",
    "license",
    "resolved_at",
    "compatibility_notes",
}


def validate_manifest(manifest: dict[str, object]) -> None:
    upstreams = manifest.get("upstreams")
    if not isinstance(upstreams, dict) or set(upstreams) != REQUIRED_NAMES:
        raise ValueError("upstreams must contain exactly the required training backends")
    for name, raw in upstreams.items():
        if not isinstance(raw, dict) or set(raw) != REQUIRED_FIELDS:
            raise ValueError(f"{name}: invalid manifest fields")
        if raw["name"] != name or not isinstance(raw["repository"], str):
            raise ValueError(f"{name}: invalid identity")
        sha = raw["resolved_sha"]
        if not isinstance(sha, str) or SHA_PATTERN.fullmatch(sha) is None:
            raise ValueError(f"{name}: resolved_sha must be a lowercase 40-character SHA")
