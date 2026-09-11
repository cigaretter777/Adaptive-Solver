"""Export a deterministic private task pool for cloud teacher rollout.

The output contains references and is therefore a training artifact: do not
serve it, commit it, or expose it to the product environment.
"""

import argparse
from pathlib import Path

from adaptive_math.training.sft_materialization import export_labeled_task_pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="v1 train or sft_dev Parquet")
    parser.add_argument("--output", type=Path, required=True, help="private labeled JSONL")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    manifest = export_labeled_task_pool(args.source, args.output, args.manifest, limit=args.limit)
    print(manifest.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
