#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
: "${ADAPTIVE_MATH_RUN_DIR:?set completed run directory}"
: "${ADAPTIVE_MATH_ARTIFACT_PREFIX:?set s3:// bucket/prefix}"
if [[ $# -ne 0 ]]; then echo "usage: $0 [--dry-run]" >&2; exit 2; fi
case "$ADAPTIVE_MATH_ARTIFACT_PREFIX" in s3://*) ;; *) echo "artifact prefix must start s3://" >&2; exit 2;; esac
manifest="$ADAPTIVE_MATH_RUN_DIR/artifact_checksums.sha256"
echo "run_dir=${ADAPTIVE_MATH_RUN_DIR} artifact_prefix=${ADAPTIVE_MATH_ARTIFACT_PREFIX} dry_run=${dry_run}"
if "$dry_run"; then exit 0; fi
find "$ADAPTIVE_MATH_RUN_DIR" -type f \( -name COMPLETE -o -name '*.jsonl' -o -name '*.json' -o -name '*.yaml' -o -name '*.log' \) -print0 | sort -z | xargs -0 sha256sum >"$manifest"
aws s3 sync --only-show-errors "$ADAPTIVE_MATH_RUN_DIR" "$ADAPTIVE_MATH_ARTIFACT_PREFIX/$(basename "$ADAPTIVE_MATH_RUN_DIR")"
aws s3 cp --only-show-errors "$manifest" "$ADAPTIVE_MATH_ARTIFACT_PREFIX/$(basename "$ADAPTIVE_MATH_RUN_DIR")/artifact_checksums.sha256"
