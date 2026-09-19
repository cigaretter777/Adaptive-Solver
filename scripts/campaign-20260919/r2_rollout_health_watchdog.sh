#!/usr/bin/env bash
# R2 rollout-health diagnostic watchdog (2026-09-19): relaunches the
# tool_call=0 diagnosis run if it dies, exits when COMPLETE appears.
# Run detached: setsid nohup ... &
set -u
REPO=/root/autodl-tmp/Adaptive-Solver-main-git
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
MERGED=$REPO/artifacts/models/qwen3_1_7b_sft_dp_v1_merged
ADAPTER=$REPO/artifacts/runs/grpo_qwen3_1_7b_r2/r2_adapter
OUT=$REPO/artifacts/rollout_health/r2_diag_10x4_seed42
LOG=$REPO/artifacts/runs/r2_rollout_health_diag.log

export HF_ENDPOINT=https://hf-mirror.com
export ADAPTIVE_MATH_SANDBOX_URL=http://127.0.0.1:8080
unset OMP_NUM_THREADS

launch() {
    setsid nohup "$PY" "$REPO/scripts/eval/run_rollout_health.py" \
        --data "$REPO/data/processed/v1/rl_dev.parquet" \
        --model "$MERGED" \
        --adapter "$ADAPTER" \
        --agent-config "$REPO/configs/agent/default.yaml" \
        --reward-config "$REPO/configs/reward/r2.yaml" \
        --task-count 10 \
        --group-size 4 \
        --selection-seed 42 \
        --output-dir "$OUT" \
        >> "$LOG" 2>&1 </dev/null &
}

while :; do
    if [ -f "$OUT/COMPLETE" ]; then
        echo "$(date -u +%H:%M:%SZ) watchdog: COMPLETE marker found, exiting" >> "$LOG"
        exit 0
    fi
    if ! pgrep -f "[r]un_rollout_health.py.*r2_diag_10x4" >/dev/null; then
        echo "$(date -u +%H:%M:%SZ) watchdog: rollout-health not running, (re)launching" >> "$LOG"
        launch
    fi
    sleep 120
done
