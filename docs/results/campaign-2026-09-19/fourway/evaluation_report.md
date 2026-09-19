# Four-way Paired Evaluation (offline join)

- Tasks: 200 (OmniMath `frozen_eval`, task_ids_sha256 `1fc257f2…`)
- Prompt version: `agent-v1`
- Verifier/extractor/prompt/parquet hashes identical across all arms (checked against eval manifests), so per-task verdicts are directly comparable.

| Arm | Correct | Invalid prediction | Valid-answer rate |
| --- | ---: | ---: | ---: |
| SFT baseline (09-14, dp_v1 merged) | 27/200 | 71 | 64.5% |
| R0 adapter (50-step GRPO, answer-only reward) | 22/200 | 55 | 72.5% |
| R2 adapter (12-step GRPO, tool-weighted reward) | 25/200 | 69 | 65.5% |

| Pair | Δ accuracy | Improved | Regressed | Unchanged | McNemar p | Bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT → R0 | -2.5% | 5 | 10 | 185 | 0.302 | [-6.0%, +1.0%] |
| SFT → R2 | -1.0% | 1 | 3 | 196 | 0.625 | [-3.0%, +1.0%] |
| R0 → R2 | +1.5% | 8 | 5 | 187 | 0.581 | [-2.0%, +5.0%] |

## Conclusion

None of the three pairwise differences is significant (McNemar exact, paired
bootstrap). Both RL adapters are statistically indistinguishable from the SFT
baseline on frozen OmniMath-200. The only visible signal is R0's protocol
conformance improvement: invalid predictions dropped from 71 to 55
(valid-answer rate 64.5% → 72.5%) without an accuracy change.

The R0/R2 numbers are consistent with the 09-14 SFT baseline under identical
verifier/extractor/prompt hashes, so the low absolute accuracy (~11–13.5%) is
a property of the 1.7B model + strict protocol on this pool, not an artifact of
a broken eval pipeline.
