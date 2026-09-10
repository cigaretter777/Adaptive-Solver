# Foundation Validation Report

Platform: macOS arm64 (Darwin 25.5.0), Python 3.12 (uv-managed venv)
Recorded: 2026-09-10
Scope: Foundation Plan Tasks 1-8 gates (offline core: contracts, verifier,
reward, data pipeline, quality gates)

## Gate results

```text
uv lock --check                      -> OK (63 packages resolved)
uv run ruff check src tests scripts  -> All checks passed
uv run mypy                          -> Success: 20 source files (strict)
uv run pytest tests/unit tests/contract -q --cov=adaptive_math
                                     -> 153 passed, 10.76-15.61s
```

## Coverage by module

| Module | Statements | Missed | Coverage |
|---|---|---|---|
| core/__init__.py | 3 | 0 | 100% |
| core/hashing.py | 13 | 0 | 100% |
| core/types.py | 89 | 3 | 97% |
| data/__init__.py | 6 | 0 | 100% |
| data/canonicalize.py | 76 | 9 | 88% |
| data/deduplicate.py | 59 | 1 | 98% |
| data/manifest.py | 41 | 0 | 100% |
| data/sources.py | 72 | 8 | 89% |
| data/split.py | 32 | 2 | 94% |
| reward/__init__.py | 3 | 0 | 100% |
| reward/functions.py | 27 | 2 | 93% |
| reward/types.py | 38 | 1 | 97% |
| verifier/__init__.py | 5 | 0 | 100% |
| verifier/extractor.py | 137 | 13 | 91% |
| verifier/normalizer.py | 22 | 0 | 100% |
| verifier/numeric.py | 179 | 19 | 89% |
| verifier/service.py | 156 | 12 | 92% |
| verifier/symbolic.py | 120 | 22 | 82% |
| verifier/worker.py | 121 | 38 | 69% |
| **TOTAL** | **1200** | **130** | **89%** |

Notes on the lower-coverage modules:

- `verifier/worker.py` (69%): the worker loop runs in a spawned process that
  pytest-cov does not trace; its behavior is pinned by four process-level
  contract tests (wall-clock kill-and-replace, CPU limit survival, crash
  detection, adversarial fixture consumption).
- `verifier/symbolic.py` (82%): the uncovered lines are defensive branches
  (unparseable input, evaluation overflow, singularity edge cases); the
  adversarial fixture drives the externally visible outcomes.
- Coverage is reported per module by design; the global percentage is not the
  quality claim.

## Unfinished-marker scan

```text
rg -n "TODO|TBD|FIXME|NotImplementedError|random\.random|np\.random" src/adaptive_math scripts tests configs
```

Only hit: `verifier/symbolic.py` catching `NotImplementedError` from SymPy
`singularities` (deliberate). No TODO/FIXME/random markers; the `pass`
statements in `worker.py` are documented best-effort cleanup paths.

## Data pipeline validation

Sample build (100 records per source, seed 20260910) against the pinned
registry:

```text
uv run python scripts/data/build_dataset.py --registry configs/data/sources.yaml \
  --output-dir /tmp/adaptive_math_sample --manifest /tmp/adaptive_math_sample.json \
  --seed 20260910 --sample-per-source 100

split_counts:  train 179, sft_dev 10, rl_dev 10, frozen_eval 62
dedup:         exact_removed 0, near_removed 0
quarantine:    139 of 400 loaded records (35%)
```

Audit:

```text
uv run python scripts/data/audit_dataset.py --manifest /tmp/adaptive_math_sample.json
-> {"ok": true, "errors": []}
```

Quarantine breakdown by source and reason:

| Source | ambiguous_answer | unverifiable_reference | missing_answer | total |
|---|---|---|---|---|
| openr1_math_220k | 57 | 20 | 7 | 84 |
| omni_math | 0 | 38 | 0 | 38 |
| numinamath_tir | 11 | 6 | 0 | 17 |
| dapo_math_17k | 0 | 0 | 0 | 0 |

**Finding (RESOLVED — solution-mode extraction):** OpenR1-Math-220k
solutions initially lost 84% of the sample because the strict
trajectory-output extractor treated multi-`$` solution prose as ambiguous.
Fix: a dedicated `extract_solution_answer` for dataset worked solutions —
last `\boxed{...}` in a bounded tail window, then prose anchors
("final answer is" / "answer is:" / "answer:") scanned last-to-first, then
whole-text single `$...$`/`$$...$$` spans ("$D$" solutions). Anchored values
stop at the first non-math word. The strict extractor for model output is
unchanged (multiple candidates remain AMBIGUOUS there).

Measured on the same 100-record sample (seed 20260910):

| Stage | OpenR1 kept | quarantine reasons |
|---|---|---|
| before fix | 16 | 57 ambiguous, 20 unverifiable, 7 missing |
| solution mode v1 | 25 | 68 missing, 7 unverifiable |
| + anchor/span rules | **29** | 64 missing, 7 unverifiable |

The remaining 64 were inspected: answers embedded in prose without anchors,
grading-rubric notes, literature references and multi-solution equations.
Extracting those would require guessing, which the fail-closed principle
forbids (a wrong label is worse than a lost row). At 220k scale, 29%
retention still yields ~64k DIRECT candidates — far above the 8,000-record
SFT bucket requirement; NuminaMath-TIR (83% kept) adds further supply.
Omni-MATH's 38% `unverifiable_reference` is expected: the evaluation plan
freezes only its rule-verifiable subset. DAPO kept 100/100.

### Full v1 build (completed after the extraction fix)

```text
uv run python scripts/data/build_dataset.py --registry configs/data/sources.yaml \
  --output-dir data/processed/v1 --manifest data/manifests/v1.json --seed 20260910

loaded:    openr1 93,733 | dapo 1,791,700 | numinamath_tir 72,441 | omni_math 4,428
kept:      openr1 21,604 (23%) | dapo 1,791,700 (0 quarantined) |
           numinamath_tir 68,870 (95%) | omni_math 3,472 (78%)
dedup:     exact_removed 1,782,412 | near_removed 2,430
quarantine total: 76,656
splits:    train 87,620 | sft_dev 4,868 | rl_dev 4,868 | frozen_eval 3,448
           (100,804 unique tasks)
build time: ~25 min on Apple M5 / 16 GB (reference self-checks short-circuit
           on canonical equality, so expression rows verify in milliseconds)
```

Audit of the full manifest:

```text
uv run python scripts/data/audit_dataset.py --manifest data/manifests/v1.json
-> {"ok": true, "errors": []}
```

Notes:

- DAPO-Math-17k stores each of its 17,917 problems ~100x (1,791,700 rows);
  exact deduplication collapses it to the nominal 17,917 unique problems,
  confirming the registry entry is sound despite the surprising row count.
- Full-build OpenR1 retention (23%) is below the 100-record sample estimate
  (29%) — sample variance; absolute supply (~21.6k OpenR1 + ~68.9k
  NuminaMath verified problems) still exceeds the 20k SFT trajectory target
  and the 8–12k RL prompt requirement.
- frozen_eval (3,448 rows) is the rule-verifiable Omni-MATH subset frozen
  for offline evaluation; the Evaluation Plan will pin it further.

## Deviations from plan (recorded)

1. math-verify 0.9's high-level `parse()` extracts numeric answers from
   solution text only; the symbolic verifier uses its `parse_latex_cached()`
   parsing layer instead (handles symbolic expressions).
2. Thousand-separator normalization moved from `normalize_surface` into the
   typed numeric parser: string-level removal cannot distinguish `1,234`
   from set elements like `{99,100}`.
3. Source registry narrowed to four apache-2.0-approved sources
   (openr1_math_220k, dapo_math_17k, numinamath_tir, omni_math); MATH,
   MATH-500 and AIME were excluded by the governance review
   (`tests/contract/test_source_registry.py` pins this decision).
4. `MathTask` gained a required `pipeline_version` field and is frozen with
   recursively immutable metadata (deep-copy-safe).
5. mypy runs as `uv run mypy` (config-driven) because the legacy
   `src/__init__.py` makes explicit path arguments resolve to a second
   module name.
