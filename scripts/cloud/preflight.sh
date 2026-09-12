#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
if [[ $# -ne 0 ]]; then echo "usage: $0 [--dry-run]" >&2; exit 2; fi
: "${ADAPTIVE_MATH_TASK_MANIFEST:?set path to private task-pool manifest}"
: "${ADAPTIVE_MATH_SANDBOX_URL:?set SandboxFusion URL}"
echo "preflight task_manifest=${ADAPTIVE_MATH_TASK_MANIFEST} sandbox_url=${ADAPTIVE_MATH_SANDBOX_URL} dry_run=${dry_run}"
python scripts/cloud/preflight.py --task-manifest "$ADAPTIVE_MATH_TASK_MANIFEST" --require-gpu --require-sandbox
if "$dry_run"; then exit 0; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[[ $(nvidia-smi -L | wc -l | tr -d ' ') -ge ${ADAPTIVE_MATH_MIN_GPU_COUNT:-1} ]]
[[ $(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}') -ge ${ADAPTIVE_MATH_MIN_DISK_GB:-500} ]]
[[ $(free -g | awk '/Mem:/ {print $2}') -ge ${ADAPTIVE_MATH_MIN_RAM_GB:-128} ]]
curl --fail --silent --show-error "${ADAPTIVE_MATH_SANDBOX_URL%/}/health" >/dev/null
python scripts/train/check_backend_contract.py --manifest third_party/manifest.json
