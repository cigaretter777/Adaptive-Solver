# Overnight Campaign Record (2026-09-19)

Status: **complete with open issue**. Formal R0 (50 steps) and R2 (12
steps) GRPO runs completed and adapters exported; both evals completed;
the eval pipeline itself is suspect (see [Open issue](#open-issue)).
Numbers below are recorded from run artifacts only.

## Chain summary

| Step | Result | Artifacts |
|---|---|---|
| R0 GRPO training (50 steps) | ✅ complete | `artifacts/runs/grpo_qwen3_1_7b_r0/` (checkpoints, `r0_adapter/COMPLETE`) |
| R2 GRPO training (12 steps) | ✅ complete | `artifacts/runs/grpo_qwen3_1_7b_r2/` (checkpoints global_step_6/12, `r2_adapter/COMPLETE`) |
| R0 eval (200 tasks, adapter-only) | ✅ complete | `artifacts/eval/r0_omnimath_200/` (`COMPLETE`, `summary.json`) |
| R2 eval (200 tasks, adapter-only) | ✅ complete after auto-recovery | `artifacts/eval/r2_omnimath_200/` (`COMPLETE`, `summary.json`) |

## Eval results (OmniMath frozen_eval, 200 tasks, prompt `agent-v1`)

| Arm | Correct | Invalid prediction | Valid-answer rate | p50 latency | Tokens/s |
|---|---:|---:|---:|---:|---:|
| R0 adapter | 22/200 (11.0%) | 55 | 72.5% | 26.5 s | 23.5 |
| R2 adapter | 25/200 (12.5%) | 69 | 65.5% | 29.5 s | 23.7 |

Paired statistics are not yet computed (`comparison.jsonl` is empty;
pairing is offline against frozen Base/SFT predictions).

## Run incidents

1. **Chain v1 eval crash** — first eval chain
   (`artifacts/runs/overnight_chain.log`) failed at 04:56Z with
   `KeyError: 'base'` in
   `src/adaptive_math/evaluation/model_eval.py:324`
   (`_render_evaluation_report`), because the adapter-only summary has
   no `base` key. Fixed by the uncommitted working-tree changes to
   `scripts/eval/run_model_eval.py` and
   `src/adaptive_math/evaluation/model_eval.py` (chain v2).
2. **Container restart killed chain v2** — the AutoDL container stopped
   at ~14:29 local and rebooted at 16:59, killing the R2 eval at 3/200.
   Relaunched with `setsid nohup` plus a watchdog
   (`artifacts/runs/r2_eval_watchdog.sh`, 120 s check loop: relaunch on
   death, exit on `COMPLETE`). The journal resumed from 3/200.
3. **Mid-run `EOFError`** — the resumed eval died once at 10:37Z
   (sandbox/verifier connection error); the watchdog relaunched it at
   10:39Z and it completed 200/200 at 12:15Z. Full log:
   `artifacts/runs/r2_eval_resume.log`.

## Open issue

**Eval pipeline is suspect.** Both adapters score 11–12.5% verifier
accuracy with a rising invalid-prediction rate (27.5% → 34.5%) — far
below what a Qwen3-1.7B SFT model should achieve on OmniMath. R2 vs R0
is +3 correct / +14 invalid, i.e. noise; no conclusion about RL
effectiveness can be drawn from these numbers.

Hypotheses to investigate, in order of suspicion:

1. Verifier/judging: why ~1/3 of predictions are invalid; check
   verifier accuracy on a sample of invalid predictions (agent-v1
   prompt, sandbox at `http://127.0.0.1:8080`).
2. Eval data alignment: OmniMath frozen_eval split vs SFT parquet
   pairing, extractor config.
3. Generation config: `do_sample=False` with invalid
   `temperature/top_p/top_k` flags being ignored (warning present in
   every eval log).

Do not start further RL rounds until (1) is resolved.

## How to reproduce

```bash
REPO=/root/autodl-tmp/Adaptive-Solver-main-git
PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python
export HF_ENDPOINT=https://hf-mirror.com WANDB_MODE=disabled \
       ADAPTIVE_MATH_SANDBOX_URL=http://127.0.0.1:8080

"$PY" "$REPO/scripts/eval/run_model_eval.py" \
    --sft-parquet "$REPO/data/processed/sft_dp_v1/train.parquet" \
    --sft-manifest "$REPO/data/manifests/sft_dp_v1_split.json" \
    --adapter "$REPO/artifacts/runs/grpo_qwen3_1_7b_r2/r2_adapter" \
    --output-dir "$REPO/artifacts/eval/r2_omnimath_200" \
    --limit 200 --rl-adapter --adapter-only \
    --model-id "$REPO/artifacts/models/qwen3_1_7b_sft_dp_v1_merged"
```

Run under `setsid nohup` + the watchdog pattern; this container reboots
unexpectedly and kills attached processes.

## Tracked copies

Runtime artifacts live under `artifacts/` (gitignored); the pieces worth
versioning are tracked here:

- Campaign scripts: `scripts/campaign-20260919/` (`overnight_chain.sh`,
  `r2_eval_watchdog.sh`, `export_verl_lora.py`, `run_grpo_direct.py`,
  `build_rl_pool.py`)
- Eval evidence: `docs/results/campaign-2026-09-19/r0/` and
  `docs/results/campaign-2026-09-19/r2/` (`evaluation_report.md`,
  `summary.json`)
