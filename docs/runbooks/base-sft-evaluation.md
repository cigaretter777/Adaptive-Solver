# Base versus SFT frozen evaluation

Run the direct-answer comparison before Agent or GRPO experiments. Both arms use
the same pinned Qwen3-1.7B model, tokenizer, Omni-MATH tasks, one-turn prompt,
greedy decoding, strict answer extractor, and verifier. The SFT arm alone enables
the LoRA adapter. The evaluator does not give either model the reference answer.

The cloud host needs the **same** SFT Parquet and manifest used to train the
adapter, plus the complete adapter directory containing `adapter_config.json`,
`adapter_model.safetensors`, and `COMPLETE`. The local `direct_openr1_5000`
Parquet has 1,148 records and is not the 1,447-record cloud training artifact described in the
September 13 report. Copy the original files and confirm their hashes before
evaluating. Install the project's runtime extras on a GPU host.

Run a read-only preflight first:

```bash
python scripts/eval/run_model_eval.py \
  --sft-parquet /path/to/sft_v1.parquet \
  --sft-manifest /path/to/sft_v1.manifest.json \
  --adapter /path/to/sft_run/adapter \
  --expected-adapter-sha256 6a97e2d4c3ccb2b9bc78efba8be1c2144619ace2fc8dfff5b7a718ec0e606fec \
  --output-dir artifacts/eval/sft_v1_omnimath_200 \
  --limit 200 --dry-run
```

Then remove `--dry-run` to run N=200. If its output is complete and the
invalid/error categories are understood, repeat with `--limit 500` and a new
output directory, such as `artifacts/eval/sft_v1_omnimath_500`. The ordered
N=200 task set is a prefix of N=500. Override `--model-revision` and
`--tokenizer-revision` if the training run used different pinned revisions.
Use `--max-new-tokens` to set one shared output budget for both arms.

Preflight verifies source hashes, row counts, adapter base model, immutable
revisions, import path, Omni-MATH split, and SFT/eval task-ID isolation before
loading the model. A completed output directory contains the two prediction
JSONL files, paired comparison JSONL, `summary.json`, `eval_manifest.json`,
and `COMPLETE`. Existing output directories are never reused. Prediction files
contain model outputs and verifier verdicts; do not publish hidden reference
answers alongside them.

The N=200 run is a pipeline smoke, not a statistical conclusion. Use the
N=500 paired result to report the accuracy difference and its uncertainty
before deciding whether the SFT adapter is an improvement. Keep direct-model
evaluation separate from later Agent/tool evaluation.
