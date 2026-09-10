# Decision 0001: Low-cost training route (1.7B mainline, 4090/Kaggle compute)

Date: 2026-09-10
Status: approved by project owner
Supersedes: the 2×H100 80GB / Qwen3-4B formal-training assumption in the
Training & Cloud Plan and spec §12.1/§16 (which remain valid as the
"budget allows" comparison track).

## Decision

The formal training mainline becomes:

| Dimension | Original plan | Approved low-cost route |
|---|---|---|
| Formal model | Qwen3-4B | **Qwen3-1.7B** (4B optional comparison if budget allows) |
| GPU | 2×H100 80GB | **1×RTX 4090 24GB (AutoDL, ~¥2/h)** |
| Window | 72h upper bound | on-demand rental, target **20–30 GPU-hours** |
| GRPO params | group=8, 8192 traj tokens | group=4, 6144 traj tokens, **no KL term** (DAPO-style; drops the reference-model forward) |
| Reward runs | R0 + R2 (+R1/R3 ablation) | **R0 full + R2 short**, single seed |
| Frozen eval | full slices | frozen_eval subset + MATH-500, ~300–500 tasks |
| Budget | ~¥1,000–1,500 | **~¥40–60 cash** + free Kaggle quota where technically possible |

## Kaggle free quota (30 GPU-h/week) — technical boundaries

Kaggle provides T4×2 (16GB) / P100. Turing (T4) constraints, verified
against stack requirements:

- **No native bf16** (needs SM80+): SFT on T4 would require an fp16
  deviation, recorded per-run in resolved_config.
- **No FlashAttention-2**: sdpa/xformers fallbacks only; the pinned
  verl/vLLM stack is not validated on SM75 — GRPO on Kaggle is
  best-effort experimentation, never the mainline.
- 12h max per session, 30h/week, internet access requires phone
  verification.

Approved Kaggle uses: SFT pilot (fp16 deviation), evaluation batch
inference (vLLM fp16 on T4 is supported), CPU-side data work.
GRPO mainline runs on AutoDL 4090 (Ada SM89: native bf16, vLLM
first-class support).

## Why this still meets the bar

The research question — whether budget-aware GRPO beats fixed tool
policies — is scale-independent. Spec §16 itself designated 1.7B for the
full reward ablations. Tool use tends to help weaker models more, so the
R0/R2 contrast may be clearer at 1.7B. The plan's documented fallbacks
(group 8→4, 8192→6144) are activated here with this record as the
required "new resolved config + paired baseline" justification.

## What we honestly give up (must appear in the final report)

- Single seed → conclusions labeled single-run; stability unverified.
- Smaller frozen eval → wider confidence intervals, reported as-is.
- 1.7B → no claim about 4B-scale capability; AIME-level tasks excluded
  from the training curriculum (all-zero reward groups), retained only as
  a reported eval slice if measurable at all.
- No KL term → drift monitored via response length / frozen mini-eval
  instead; any instability recorded as a negative finding.

## Consequences for pending work

- `configs/sft/qwen3_1_7b_lora.yaml` becomes the primary SFT config;
  the 4B config stays as the comparison-track file.
- Cloud runbook (Training Plan Task 8) will be written for AutoDL:
  base image + `requirements.cuda.lock` + `environment.json` replaces
  the immutable Dockerfile.train (AutoDL has no Docker-in-Docker);
  SandboxFusion runs via remote URL (`ADAPTIVE_MATH_SANDBOX_URL`) —
  public bytedance endpoint for smoke, self-hosted VPS for the formal
  window if rate limits bite.
- Gate A "training image builds" is equivalently restated as "locked
  install reproduces from the AutoDL base image".
