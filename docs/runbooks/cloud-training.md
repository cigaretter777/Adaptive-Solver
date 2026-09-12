# Cloud training runbook

## Provision and image

Use a Linux GPU host with the capacity in `configs/cloud/h100_2x.yaml`. Build
only from an immutable CUDA image digest; never substitute a tag.

The locked rollout stack is **Torch 2.8.0 + CUDA 12.8 wheels + vLLM 0.11.0**.
vLLM 0.11.0 requires Torch 2.8.0, so do not combine it with the older
Torch 2.6 / cu124 route. On AutoDL (where Docker is normally unavailable),
install the same lock in the data-disk Conda environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/adaptive-math
python -m pip install --force-reinstall \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install vllm==0.11.0
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Run `python -m pip check` and the GPU import check before starting SFT or
GRPO. Do not run these installations concurrently.

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
