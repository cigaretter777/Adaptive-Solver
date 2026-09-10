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

**Finding (open before the full build):** OpenR1-Math-220k solutions lose 84%
of the sample. The dominant cause is `ambiguous_answer` from "N dollar
groups found": those solutions contain multiple `$...$` spans and no
`\boxed{}`, so the final-answer extractor (designed for short answer
strings) cannot locate a single answer. OpenR1's solution format is
heterogeneous (`upstream_source` varies); the fix is a solution-mode
extraction path (last `\boxed`, then "final answer is"-style prose anchors,
then trailing expression) with per-source audit before the full build.
Omni-MATH's 38% `unverifiable_reference` reflects its intentionally hard
answer formats (sets, intervals, radicals) under the expression verifier.
DAPO's integer configuration kept 100/100 on the sample.

Full `data/manifests/v1.json` build: deferred by design to the training data
materialization step (Training Plan Task 2); the sample build exercises the
identical code path and the audit runs against the sample manifest.

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
