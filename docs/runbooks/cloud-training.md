# Cloud training runbook

## Provision and image

Use a Linux GPU host with the capacity in `configs/cloud/h100_2x.yaml`. Build
only from an immutable CUDA image digest; never substitute a tag.

```bash
docker build -f docker/Dockerfile.train -t adaptive-math-rl:train \
  --build-arg CUDA_BASE_IMAGE='nvidia/cuda@sha256:<approved-digest>' .
```

Set `SANDBOXFUSION_IMAGE_DIGEST` and `SANDBOXFUSION_GIT_SHA`, then start the
separately pinned sandbox with `docker compose -f docker/compose.sandbox.yml up -d`.
Do not put cloud, Hugging Face, W&B, or S3 credentials in Git.

## Preflight and persistence

Copy the private task-pool JSONL and manifest, set `ADAPTIVE_MATH_TASK_MANIFEST`
and `ADAPTIVE_MATH_SANDBOX_URL`, then run `scripts/cloud/preflight.sh`. Start a
`tmux` session before training so SSH loss cannot terminate it.

```bash
tmux new -s adaptive-math
scripts/cloud/launch_grpo.sh --dry-run
scripts/cloud/launch_grpo.sh
```

Run `sync_artifacts.sh` after every complete checkpoint. It writes checksums
and copies the run evidence to the configured S3-compatible prefix.

## Emergency stop

Run `python scripts/cloud/stop_if_unhealthy.py --metrics <run>/metrics.jsonl`
between evaluation intervals. On a non-zero result, stop the trainer, sync
evidence, stop Ray and SandboxFusion, then terminate the GPU instance. Record
the reason and direct cost. Resume only through `resume_latest.sh` after the
resolved config and input hashes match.
