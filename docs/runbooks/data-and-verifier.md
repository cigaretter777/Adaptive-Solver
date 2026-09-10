# Data and Verifier Runbook

Offline core operations: dataset sources, manifests, quarantine and answer
verification. All commands run from the repository root with `uv`.

## Rebuild the dataset

```bash
uv run python scripts/data/build_dataset.py \
  --registry configs/data/sources.yaml \
  --output-dir data/processed/v1 \
  --manifest data/manifests/v1.json \
  --seed 20260910
```

The build is deterministic for a fixed registry, seed, code revision and
dependency set:

1. loads raw records per source (pinned revision in the registry),
2. canonicalizes each record into a `LabeledMathTask` (content-addressed
   `task_id`, SHA-256 `source_hash`, per-source `answer_type`),
3. self-checks every reference answer; anything not `CORRECT` is quarantined,
4. deduplicates exactly on canonical problem hash, then near-duplicates via
   character 5-gram MinHash (threshold 0.90); in cross-split clusters the
   evaluation member wins,
5. assigns seeded splits (eval-intended sources are `frozen_eval` only),
6. writes one Parquet file per split plus `quarantine.jsonl` and the manifest.

`--dry-run` runs the full pipeline without writing anything.
`--sample-per-source N` limits each source to N records (smoke builds).

Never commit downloaded data: `data/processed/` and `data/raw/` are ignored.
`data/manifests/` and `data/README.md` stay tracked.

## Audit a materialized dataset

```bash
uv run python scripts/data/audit_dataset.py --manifest data/manifests/v1.json
```

Exits nonzero when any of these fail: schema version mismatch, unknown
license, missing or hash-mismatched split files, schema/count/task-id-hash
mismatch, duplicate task IDs across splits, `frozen_eval` leakage into
training splits, or a reference that no longer self-verifies.

## Inspect quarantine rows

```bash
cat data/processed/v1/quarantine.jsonl   # one JSON object per row
```

Reasons: `missing_problem`, `missing_answer`, `ambiguous_answer`,
`unverifiable_reference`. Each row records `source`, the SHA-256 of the
original source record, the reason and a human-readable detail.

## Add a dataset source

1. Append an entry to `configs/data/sources.yaml` with name, URI, an
   immutable 40-hex revision, license (allowlisted in
   `scripts/data/audit_dataset.py`), `intended_use` (`train` or `eval`),
   citation, loader, column mapping and `answer_type`.
2. If the loader does not exist, add it to `src/adaptive_math/data/sources.py`
   (dispatch is in `load_source_records`).
3. Extend `tests/contract/test_source_registry.py` to pin the new entry.
4. Run a `--sample-per-source 100` build, inspect quarantine counts, then run
   the audit. Pin the resolved revision into the registry before the full build.

## Rotate a verifier version

Verifier behavior is versioned through the pipeline: every task carries
`pipeline_version`, and reward configs carry their own `version`/hash.

1. Change verifier code and bump the task `pipeline_version` constant used by
   `canonicalize_source` in the same commit.
2. Run the full verifier suite (golden, adversarial, worker contracts):
   `uv run pytest tests/unit/verifier tests/contract/test_verifier_worker.py -q`
3. Rebuild and re-audit the dataset; a reference that no longer self-verifies
   under the new code is quarantined, never silently relabeled.

## Reproduce one verification decision

```bash
uv run python -c "
from adaptive_math.verifier import verify_answer
from adaptive_math.core.types import ReferenceAnswer, AnswerType
print(verify_answer('42', ReferenceAnswer(value='42', answer_type=AnswerType.INTEGER), task_id='repro').model_dump_json(indent=2))
"
```

Symbolic (EXPRESSION) comparisons run in a spawned worker process with a
per-request CPU timer and a parent wall-clock deadline; see
`src/adaptive_math/verifier/worker.py`. The decision evidence order is
canonical equality, domain-aware simplify difference, numeric cross-check at
task_id-seeded points.
