# Paired Base vs SFT Evaluation

- Tasks: 200
- Prompt version: `agent-v1`
- Accuracy delta (SFT − Base): 0.0950
- Paired bootstrap 95% CI: [0.05, 0.145]
- McNemar p-value: 0.000156522

| Arm | Correct | Valid-answer rate | Verifier accuracy | p50 latency (ms) | p95 latency (ms) | Tokens/s | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 7/200 | 25.50% | 3.50% | 29965.43 | 39623.34 | 257.85 | 4.51 GiB |
| SFT | 26/200 | 71.50% | 13.00% | 50651.48 | 56477.22 | 161.29 | 4.51 GiB |

## Paired outcomes

- Improved: 22
- Regressed: 3
- Unchanged: 175

Machine-readable provenance and checksums are in `eval_manifest.json`; per-task outputs are JSONL.
