"""Exit non-zero when rolling JSONL metrics exceed conservative cloud stop gates."""

import argparse
import json
import math
from pathlib import Path


def unhealthy(metrics_path: Path) -> list[str]:
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not rows:
        return ["no metrics recorded"]
    latest = rows[-1]
    reasons: list[str] = []
    for key in ("loss", "reward"):
        value = latest.get(key)
        if isinstance(value, float) and math.isnan(value):
            reasons.append(f"NaN {key}")
    limits = {
        "verifier_error_rate": 0.02,
        "invalid_action_rate": 0.25,
        "sandbox_timeout_rate": 0.10,
        "ineffective_group_rate": 0.80,
    }
    for key, limit in limits.items():
        value = latest.get(key)
        if isinstance(value, (int, float)) and value > limit:
            reasons.append(f"{key}={value} exceeds {limit}")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reasons = unhealthy(args.metrics)
    print(json.dumps({"ok": not reasons, "reasons": reasons}, sort_keys=True))
    return 0 if not reasons or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
