#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
: "${ADAPTIVE_MATH_RUN_DIR:?set run directory to inspect}"
: "${ADAPTIVE_MATH_RESUME_COMMAND:?set exact training command accepting --resume_from_checkpoint}"
if [[ $# -ne 0 ]]; then echo "usage: $0 [--dry-run]" >&2; exit 2; fi
checkpoint=$(find "$ADAPTIVE_MATH_RUN_DIR" -type f -name COMPLETE -print | sort | tail -n 1 || true)
if [[ -z "$checkpoint" ]]; then echo "no complete checkpoint under $ADAPTIVE_MATH_RUN_DIR" >&2; exit 1; fi
checkpoint_dir=$(dirname "$checkpoint")
[[ -f "$ADAPTIVE_MATH_RUN_DIR/resolved_config.yaml" ]]
echo "resume_checkpoint=${checkpoint_dir} dry_run=${dry_run}"
if "$dry_run"; then exit 0; fi
"${ADAPTIVE_MATH_RESUME_COMMAND}" --resume_from_checkpoint "$checkpoint_dir"
