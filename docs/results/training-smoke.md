# Training Smoke Record

Status: **pending GPU execution** (low-cost route per
[decision 0001](../decisions/0001-low-cost-training-route.md)).

This file records SFT pilot results and the formal learning-rate selection
with their evidence, as required by Training Plan Task 4. Numbers are
appended only after the corresponding run completes; nothing here may be
filled in from expectation.

## SFT pilot table (to be filled on the GPU host)

| Run | Config | LR | Data | Loss first→last | Protocol validity | Verdict |
|---|---|---|---|---|---|---|
| overfit-smoke | qwen3_1_7b_smoke.yaml | 2e-5 | 200 synthetic DIRECT | pending | pending (gate ≥98%) | pending |
| pilot-lr-5e-6 | qwen3_1_7b_lora.yaml --set learning_rate=5e-6 | 5e-6 | sft_v1 | pending | pending | pending |
| pilot-lr-1e-5 | qwen3_1_7b_lora.yaml | 1e-5 | sft_v1 | pending | pending | pending |
| pilot-lr-2e-5 | qwen3_1_7b_lora.yaml --set learning_rate=2e-5 | 2e-5 | sft_v1 | pending | pending | pending |

## Formal LR selection

Selection rule (fixed before running, per plan): choose on SFT-dev loss,
protocol validity (≥98%) and the frozen mini-eval — **not** on training
loss alone. Record the chosen run's resolved_config hash here.

- Selected LR: pending
- Evidence: pending (metrics.jsonl + eval artifacts links)

## Checkpoint gate (before GRPO)

- [ ] parse success ≥98% on SFT dev
- [ ] all four behavior categories present in dev generations
- [ ] tool observation used after ≥90% of successful tool calls
- [ ] Base-Direct mini-eval degradation ≤2 absolute points

## Execution notes

- Host: pending (planned: AutoDL RTX 4090 24GB; Kaggle T4 requires an
  fp16 deviation record — T4 has no native bf16)
- Command: `uv run python scripts/train/run_sft.py --config configs/sft/qwen3_1_7b_lora.yaml --data <sft_v1.parquet> --data-manifest <sft_v1.json>`
- Artifacts: `artifacts/sft/*/resolved_config.yaml`, `environment.json`,
  `metrics.jsonl`, `adapter/` (+COMPLETE marker), `rejects.jsonl` if any
