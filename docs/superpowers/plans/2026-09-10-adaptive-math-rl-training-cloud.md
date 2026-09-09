# AdaptiveMath-RL Training and Cloud Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在可复现的云端环境中完成 Qwen3-4B LoRA SFT 与同步 Agentic GRPO，并在最多 72 小时正式训练窗口内产出 Base/SFT/R0/R2 可比较的 checkpoints、轨迹、配置和指标。

**Architecture:** 本项目保留数据、Agent 环境、token mask、reward 和实验配置的所有权；verl-agent/verl 只作为被固定版本的训练后端。SFT 先建立动作协议与工具恢复冷启动，GRPO 再以在线多轮 rollout、确定性终局验证和 group-relative advantage 优化。云端通过不可变镜像、分阶段 smoke gate、断点续训和对象存储同步控制三天风险。

**Tech Stack:** Qwen3-1.7B/Qwen3-4B、PyTorch bf16、Transformers、PEFT LoRA、Accelerate、verl-agent、verl、vLLM 0.11-compatible pinned stack、Ray、SandboxFusion、Hydra/OmegaConf、W&B 或 MLflow、Docker、NVIDIA H100 80GB。

**Spec:** [AdaptiveMath-RL 产品与技术设计报告](../specs/2026-09-09-adaptive-math-rl-product-design.md)

## Global Constraints

- [ ] Foundation 和 Agent Runtime 两份计划必须完成；训练代码不得复制另一套 parser、prompt、tool 或 verifier。
- [ ] 固定上游完整 Git SHA、容器 digest、模型 revision 与数据 manifest hash；只记录 branch/tag 不算固定版本。
- [ ] SFT 和 rollout 保存生成时的 token IDs；RL 不允许把文本事后重分词后当成原始 action tokens。
- [ ] 只有 assistant thinking/action token 的 loss mask 为 1；system、user、tool observation、环境警告和 padding 为 0。
- [ ] 同一 prompt group 的 G 条轨迹必须来自同一 policy version；V1 不做异步 stale rollout。
- [ ] 正式训练前必须完成 32-prompt 全链路反向与 checkpoint-resume smoke。
- [ ] 任一 frozen accuracy 下降而 train reward 持续上升时暂停训练，先审计 verifier exploit、数据偏移和成本项。
- [ ] 72 小时是租用 GPU 的正式窗口；镜像、数据、单测和小模型 smoke 必须提前完成。

---

## File Structure

~~~text
third_party/
└── manifest.json
docker/
├── Dockerfile.train
└── requirements.cuda.lock
configs/
├── sft/qwen3_1_7b_smoke.yaml
├── sft/qwen3_4b_lora.yaml
├── grpo/qwen3_1_7b_smoke_r0.yaml
├── grpo/qwen3_4b_r0.yaml
├── grpo/qwen3_4b_r1.yaml
├── grpo/qwen3_4b_r2.yaml
├── grpo/qwen3_4b_r3.yaml
└── cloud/h100_2x.yaml
src/adaptive_math/training/
├── __init__.py
├── config.py
├── sft_records.py
├── sft_builder.py
├── tokenization.py
├── collator.py
├── rollout_records.py
├── verl_environment.py
├── reward_bridge.py
├── metrics.py
└── artifacts.py
scripts/
├── setup/pin_upstreams.py
├── data/build_sft_trajectories.py
├── data/audit_sft_trajectories.py
├── train/run_sft.py
├── train/run_grpo.py
├── train/check_backend_contract.py
├── train/profile_rollout.py
└── cloud/
    ├── preflight.sh
    ├── launch_sft.sh
    ├── launch_grpo.sh
    ├── resume_latest.sh
    ├── sync_artifacts.sh
    └── stop_if_unhealthy.py
docs/runbooks/cloud-training.md
docs/results/training-smoke.md
tests/unit/training/
tests/contract/test_verl_agent_adapter.py
tests/integration/test_sft_overfit.py
tests/integration/test_grpo_smoke.py
~~~

---

### Task 1: Pin the Training Backend and Build a Compatibility Gate

**Files:**

- Create: third_party/manifest.json
- Create: scripts/setup/pin_upstreams.py
- Create: scripts/train/check_backend_contract.py
- Create: tests/unit/training/test_upstream_manifest.py
- Create: tests/contract/test_verl_agent_adapter.py
- Modify: pyproject.toml

- [ ] Write manifest schema tests requiring name, repository, requested_ref, resolved_sha with exactly 40 lowercase hex characters, license, resolved_at and compatibility_notes for verl-agent, verl and SandboxFusion.

- [ ] Implement pin_upstreams.py so resolve fetches a ref, resolves it to a full SHA, refuses a dirty vendored checkout, writes canonical JSON and prints a diff. It must never silently advance an existing resolved_sha; --update is required for changes.

- [ ] Resolve and review current upstream heads once, then commit the immutable result:

~~~bash
uv run python scripts/setup/pin_upstreams.py resolve --manifest third_party/manifest.json
uv run pytest tests/unit/training/test_upstream_manifest.py -q
~~~

- [ ] Add training dependencies in a cloud extra rather than the local core dependency set. The Docker build installs the recorded verl-agent checkout and uses the compatible verl code bundled/pinned by that checkout.

- [ ] Write check_backend_contract.py to import the pinned backend and assert these capabilities: Qwen3 model loading, LoRA actor support, grouped rollouts, GRPO estimator, dynamic sampling switch, multi-turn environment manager, checkpoint save/load and vLLM/SGLang rollout backend.

- [ ] In the adapter contract, assert the pinned EnvironmentManagerBase exposes reset(kwargs), step(text_actions), build_text_obs() and success_evaluator(...). Fail with the recorded SHA and missing symbol if upstream changes.

- [ ] Run in the training image:

~~~bash
uv run python scripts/train/check_backend_contract.py --manifest third_party/manifest.json
uv run pytest tests/contract/test_verl_agent_adapter.py -q
~~~

- [ ] Commit:

~~~bash
git add third_party/manifest.json scripts/setup scripts/train/check_backend_contract.py tests/unit/training/test_upstream_manifest.py tests/contract/test_verl_agent_adapter.py pyproject.toml uv.lock
git commit -m "build: pin verl agent training backend"
~~~

---

### Task 2: Build and Audit 20K SFT Agent Trajectories

**Files:**

- Create: src/adaptive_math/training/sft_records.py
- Create: src/adaptive_math/training/sft_builder.py
- Create: scripts/data/build_sft_trajectories.py
- Create: scripts/data/audit_sft_trajectories.py
- Create: tests/unit/training/test_sft_records.py
- Create: tests/unit/training/test_sft_builder.py
- Create: tests/contract/test_sft_data_quality.py

- [ ] Define SFTBehavior as DIRECT, PYTHON, SYMPY or RECOVERY and SFTTrajectoryRecord with task_id, behavior, messages, final_answer, verifier_result, source_trace_ids, tool_execution_ids, tokenizer_revision and transform_version.

- [ ] Test that every record follows this role sequence: system, user, then one or more assistant turns separated only by tool observations; the last assistant turn is FINAL; each assistant parses through the production action parser.

- [ ] Test that Python/SymPy records contain a real matching TOOL_CALL/TOOL_RESULT pair and that RECOVERY contains at least one failed/invalid result followed by a successful corrective step and a correct final answer.

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/training/test_sft_records.py tests/unit/training/test_sft_builder.py -q
~~~

- [ ] Build the initial target mix from verified NuminaMath-TIR, OpenR1-Math-220k, MATH train and project-generated runtime traces:

~~~text
DIRECT    8,000 records, 40%
PYTHON    7,000 records, 35%
SYMPY     3,000 records, 15%
RECOVERY  2,000 records, 10%
~~~

- [ ] For direct records, transform an existing verified solution into one think block plus FINAL. For tool records, execute every stored/generated call through the production tool registry and retain the actual observation. For missing behavior buckets, sample Qwen/Qwen3-8B teacher candidates through the same AgentLoop, without putting reference answers in prompts, and keep only verifier-correct trajectories.

- [ ] Reject duplicated task IDs within a behavior bucket, trajectories over 8192 tokens, unverifiable finals, tool observations that differ from replay, reference strings found in hidden metadata, and generated recoveries that did not actually encounter an error.

- [ ] Build to Parquet and produce a manifest. If quality filtering yields fewer than 18,000 unique records, keep the smaller valid set and fail the coverage gate instead of duplicating rows.

~~~bash
uv run python scripts/data/build_sft_trajectories.py --task-manifest data/manifests/v1.json --output data/processed/sft_v1.parquet --manifest data/manifests/sft_v1.json --seed 20260910
uv run python scripts/data/audit_sft_trajectories.py --manifest data/manifests/sft_v1.json --min-records 18000 --max-records 22000
~~~

- [ ] The audit must print counts, tokens, behavior, domain, difficulty, answer type, tool success, recovery type and source distribution; it exits nonzero if any source contributes over 60% or protocol validity is below 100%.

- [ ] Commit only code and the small manifest; keep Parquet and raw traces in artifact storage:

~~~bash
git add src/adaptive_math/training/sft_records.py src/adaptive_math/training/sft_builder.py scripts/data tests/unit/training tests/contract/test_sft_data_quality.py data/manifests/sft_v1.json
git commit -m "feat: build verified sft agent trajectories"
~~~

---

### Task 3: Implement Tokenization and Loss Masks Once

**Files:**

- Create: src/adaptive_math/training/tokenization.py
- Create: src/adaptive_math/training/collator.py
- Create: tests/unit/training/test_tokenization.py
- Create: tests/unit/training/test_collator.py
- Create: tests/fixtures/token_mask_golden.json

- [ ] Write a golden Qwen3 conversation with two assistant turns and one tool observation. Assert system/user/tool/padding labels are -100 and every assistant think/action token has its own token ID as label.

- [ ] Test action boundaries, empty think, Unicode, truncated-over-limit rejection, left/right padding, batch collation and tokenizer revision mismatch.

- [ ] Run failing tests with the real Qwen3 tokenizer cached in test setup:

~~~bash
uv run pytest tests/unit/training/test_tokenization.py tests/unit/training/test_collator.py -q
~~~

- [ ] Implement incremental chat-template rendering using tokenizer.apply_chat_template and prefix-difference spans. Set assistant spans to trainable and every other role to ignored. Reject a record if prefix-difference cannot align exactly; do not guess offsets from decoded text.

- [ ] Return TokenizedTrajectory(input_ids, attention_mask, labels, assistant_mask, message_spans, tokenizer_revision). Assert labels[i] equals input_ids[i] exactly where assistant_mask[i] is 1.

- [ ] Implement CausalLMCollator with explicit pad_token_id and max_length=8192. It rejects overlength examples instead of truncating a terminal answer.

- [ ] Save the golden token IDs and masks with the exact Qwen/Qwen3-4B tokenizer revision. Any tokenizer upgrade must deliberately regenerate and review this fixture.

- [ ] Run:

~~~bash
uv run pytest tests/unit/training/test_tokenization.py tests/unit/training/test_collator.py -q
uv run mypy src/adaptive_math/training/tokenization.py src/adaptive_math/training/collator.py
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/training/tokenization.py src/adaptive_math/training/collator.py tests/unit/training/test_tokenization.py tests/unit/training/test_collator.py tests/fixtures/token_mask_golden.json
git commit -m "feat: mask non-policy tokens during training"
~~~

---

### Task 4: Train and Gate the SFT LoRA

**Files:**

- Create: src/adaptive_math/training/config.py
- Create: scripts/train/run_sft.py
- Create: configs/sft/qwen3_1_7b_smoke.yaml
- Create: configs/sft/qwen3_4b_lora.yaml
- Create: tests/unit/training/test_sft_config.py
- Create: tests/integration/test_sft_overfit.py
- Create: docs/results/training-smoke.md

- [ ] Write config validation tests requiring immutable model/tokenizer revisions, bf16, max_length <= model context, LoRA target modules, effective token batch, seed, data manifest hash, output directory and logging interval.

- [ ] Set the initial 4B pilot config explicitly:

~~~yaml
model_id: Qwen/Qwen3-4B
precision: bf16
max_length: 8192
epochs: 1
learning_rate: 1.0e-5
warmup_ratio: 0.03
weight_decay: 0.0
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_targets: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
gradient_checkpointing: true
seed: 20260910
~~~

- [ ] Keep 5e-6 and 2e-5 as two short pilot overrides; select the formal learning rate on SFT dev loss, protocol validity and frozen mini-eval, not training loss alone.

- [ ] Implement run_sft.py using Transformers causal LM + PEFT LoRA + Accelerate. It writes resolved_config.yaml, environment.json, metrics.jsonl and adapter checkpoints atomically.

- [ ] Write a 200-record overfit integration test using Qwen3-1.7B: loss must decrease, checkpoint reload must match adapter weights, and at least 98% of training prompts must emit parseable terminal actions under greedy generation. Mark it cloud-smoke, not default CI.

- [ ] Run local config tests and cloud overfit:

~~~bash
uv run pytest tests/unit/training/test_sft_config.py -q
ADAPTIVE_MATH_RUN_GPU_TESTS=1 uv run pytest tests/integration/test_sft_overfit.py -q
accelerate launch scripts/train/run_sft.py --config configs/sft/qwen3_1_7b_smoke.yaml
~~~

- [ ] Run formal SFT only after the overfit gate:

~~~bash
accelerate launch --multi_gpu scripts/train/run_sft.py --config configs/sft/qwen3_4b_lora.yaml
~~~

- [ ] Gate the selected checkpoint on SFT dev: parse success >=98%, all four behavior categories present, tool observation used after at least 90% of successful tool calls, and Base-Direct mini-eval accuracy degradation no more than two absolute points.

- [ ] Record pilot table and selection reason in docs/results/training-smoke.md.

- [ ] Commit code/config/results, not checkpoint binaries:

~~~bash
git add src/adaptive_math/training/config.py scripts/train/run_sft.py configs/sft tests/unit/training/test_sft_config.py tests/integration/test_sft_overfit.py docs/results/training-smoke.md
git commit -m "feat: train qwen3 agent sft adapter"
~~~

---

### Task 5: Adapt MathAgentEnv to the Pinned verl-agent Contract

**Files:**

- Create: src/adaptive_math/training/rollout_records.py
- Create: src/adaptive_math/training/verl_environment.py
- Modify: tests/contract/test_verl_agent_adapter.py
- Create: tests/unit/training/test_rollout_records.py
- Create: tests/integration/test_vectorized_rollout.py

- [ ] Define RolloutStep with env_id, group_id, policy_version, input_ids, generated_ids, assistant_mask, action text, observation text, reward, done and info. Define RolloutTrajectory as an ordered tuple plus the production Trajectory and EvaluationRecord.

- [ ] Write a vectorized integration test with four tasks times four group samples. Use scripted outputs to produce correct, incorrect, tool, invalid and timeout trajectories; assert group IDs, policy version and per-environment state never cross.

- [ ] Implement VerlMathEnvironmentManager against the pinned EnvironmentManagerBase:

~~~python
def reset(self, kwargs: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]: ...
def step(self, text_actions: list[str]) -> tuple[dict[str, object], np.ndarray, np.ndarray, list[dict[str, object]]]: ...
def build_text_obs(self) -> list[str]: ...
def success_evaluator(self, *args: object, **kwargs: object) -> dict[str, np.ndarray]: ...
~~~

- [ ] reset creates one OfflineMathEnv per task/group copy and returns public initial observations. step parses each text action once, executes active environments concurrently, returns terminal reward only after FINAL/termination, and fills info with trace_id, verifier status, reward breakdown, usage and termination reason.

- [ ] Preserve generated token IDs and masks provided by the backend on RolloutStep. Assert policy_version equality across each group before reward calculation.

- [ ] Run:

~~~bash
uv run pytest tests/unit/training/test_rollout_records.py tests/contract/test_verl_agent_adapter.py tests/integration/test_vectorized_rollout.py -q
~~~

- [ ] Profile 32 concurrent environments with fake 10–500 ms tool latency and prove one slow sandbox request does not serialize unrelated environments.

- [ ] Commit:

~~~bash
git add src/adaptive_math/training/rollout_records.py src/adaptive_math/training/verl_environment.py tests/unit/training/test_rollout_records.py tests/contract/test_verl_agent_adapter.py tests/integration/test_vectorized_rollout.py
git commit -m "feat: connect math environment to verl agent"
~~~

---

### Task 6: Bridge Deterministic Rewards and GRPO Batches

**Files:**

- Create: src/adaptive_math/training/reward_bridge.py
- Create: src/adaptive_math/training/metrics.py
- Create: configs/grpo/qwen3_1_7b_smoke_r0.yaml
- Create: configs/grpo/qwen3_4b_r0.yaml
- Create: configs/grpo/qwen3_4b_r1.yaml
- Create: configs/grpo/qwen3_4b_r2.yaml
- Create: configs/grpo/qwen3_4b_r3.yaml
- Create: tests/unit/training/test_reward_bridge.py
- Create: tests/unit/training/test_group_advantages.py

- [ ] Write tests that compare every bridge output to direct compute_reward for R0–R3 and reject missing EvaluationRecord, policy-version mismatch, duplicate sample ID and non-finite reward.

- [ ] Write GRPO normalization tests for mixed, all-zero and all-one groups. Mixed groups use (r_i - group_mean) / (group_std + 1e-6). Constant groups are marked ineffective and produce zero advantages.

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/training/test_reward_bridge.py tests/unit/training/test_group_advantages.py -q
~~~

- [ ] Implement MathRewardManager as a thin adapter from terminal rollout info to the production reward function. Export per-trajectory reward component columns rather than a scalar-only log.

- [ ] Configure dynamic sampling to resample ineffective all-zero/all-one groups up to a bounded attempt count, while logging the original ineffective rate. Never discard groups after observing frozen-eval labels.

- [ ] Start R0/R2 from the same SFT adapter, RL manifest, prompt order and seed. Initial shared settings:

~~~yaml
model_id: Qwen/Qwen3-4B
algorithm: grpo
precision: bf16
lora_rank: 64
actor_learning_rate: 1.0e-6
group_size: 8
max_prompt_tokens: 2048
max_trajectory_tokens: 8192
max_steps: 6
max_tool_calls: 4
max_python_seconds: 12
temperature: 1.0
top_p: 0.95
sync_rollout: true
dynamic_sampling: true
dynamic_sampling_max_attempts: 3
seed: 20260910
~~~

- [ ] Treat group_size=4, max_trajectory_tokens=6144 and smaller prompt batch as documented throughput fallbacks. Any activation requires a new resolved config and paired baseline using the same values.

- [ ] Log reward mean/std, group variance, ineffective group rate, verifier status, invalid action, tool distribution, Python time, response length, KL, entropy, clip fraction and gradient norm.

- [ ] Run:

~~~bash
uv run pytest tests/unit/training/test_reward_bridge.py tests/unit/training/test_group_advantages.py -q
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/training/reward_bridge.py src/adaptive_math/training/metrics.py configs/grpo tests/unit/training/test_reward_bridge.py tests/unit/training/test_group_advantages.py
git commit -m "feat: bridge rlvr rewards into grpo"
~~~

---

### Task 7: Add End-to-End GRPO Smoke and Profiling Gates

**Files:**

- Create: scripts/train/run_grpo.py
- Create: scripts/train/profile_rollout.py
- Create: tests/integration/test_grpo_smoke.py
- Modify: docs/results/training-smoke.md

- [ ] Implement run_grpo.py as the project entrypoint that validates all paths/hashes, starts the pinned upstream trainer, registers VerlMathEnvironmentManager and MathRewardManager, and copies the upstream resolved Hydra config into the run directory.

- [ ] Write a GPU smoke using Qwen3-1.7B, 32 prompts, group size 4 and two optimizer steps. Assert rollout, actual tool call, terminal verifier, nonconstant group, backward, optimizer step, LoRA checkpoint, reload and resume to step three.

- [ ] Add a mask assertion inside the smoke: sum of policy loss weights over tool observation tokens is zero; sum over assistant action tokens is nonzero.

- [ ] Run:

~~~bash
ADAPTIVE_MATH_RUN_GPU_TESTS=1 uv run pytest tests/integration/test_grpo_smoke.py -q
torchrun --nproc_per_node=2 scripts/train/run_grpo.py --config configs/grpo/qwen3_1_7b_smoke_r0.yaml
~~~

- [ ] Profile the 4B configuration for 200 trajectories and write rollout_profile.json containing trajectories/hour, generated tokens/second, sandbox wait p50/p95, peak GPU memory, CPU utilization, mean steps and projected wall time.

~~~bash
torchrun --nproc_per_node=2 scripts/train/profile_rollout.py --config configs/grpo/qwen3_4b_r0.yaml --trajectories 200 --output artifacts/profiles/qwen3_4b_2xh100.json
~~~

- [ ] If projected time exceeds 54 training hours, apply fallbacks in order: group size 8 to 4, max trajectory 8192 to 6144, smaller prompt batch with same effective batch, then 4 GPUs. Re-profile after each change; do not reduce verification or sandbox isolation.

- [ ] Record the chosen configuration and measured reason in training-smoke.md.

- [ ] Commit:

~~~bash
git add scripts/train/run_grpo.py scripts/train/profile_rollout.py tests/integration/test_grpo_smoke.py docs/results/training-smoke.md
git commit -m "test: gate end to end agentic grpo"
~~~

---

### Task 8: Build the Immutable Cloud Image and Operations Scripts

**Files:**

- Create: docker/Dockerfile.train
- Create: docker/requirements.cuda.lock
- Create: configs/cloud/h100_2x.yaml
- Create: scripts/cloud/preflight.sh
- Create: scripts/cloud/launch_sft.sh
- Create: scripts/cloud/launch_grpo.sh
- Create: scripts/cloud/resume_latest.sh
- Create: scripts/cloud/sync_artifacts.sh
- Create: scripts/cloud/stop_if_unhealthy.py
- Create: tests/contract/test_cloud_scripts.py
- Create: docs/runbooks/cloud-training.md

- [ ] Write shell contract tests with ShellCheck plus a dry-run mode. Every script must use set -euo pipefail, quote paths, reject unset run ID, and print resolved non-secret settings.

- [ ] Build Dockerfile.train from an NVIDIA CUDA/PyTorch base compatible with the pinned backend. Install exact Python, CUDA wheel, vLLM and flash-attention versions; install project and pinned upstream source; run backend contract and unit tests during build.

- [ ] preflight.sh checks nvidia-smi, GPU count/model/memory, CUDA/PyTorch compatibility, NCCL all-reduce, disk >=500 GB, RAM >=128 GB, model/data hashes, sandbox health, artifact store credentials and clock sync.

- [ ] launch scripts create run_id from UTC timestamp plus Git SHA, refuse a dirty code tree unless ADAPTIVE_MATH_ALLOW_DIRTY_RUN=1, snapshot environment and configs, and stream logs to local JSONL plus one configured tracker.

- [ ] sync_artifacts.sh copies only atomic checkpoints, logs, traces, configs and manifests to the configured S3-compatible prefix. It writes checksums and verifies remote object sizes before marking sync complete.

- [ ] resume_latest.sh selects the newest checkpoint with a complete marker and matching config/data/model hashes. It refuses cross-experiment resume.

- [ ] stop_if_unhealthy.py terminates the trainer gracefully after any of these sustained conditions: NaN loss/reward, verifier infrastructure error >2%, invalid action >25% after warmup, sandbox timeout >10%, zero effective groups >80%, frozen mini-eval drop >5 points, disk <100 GB or no checkpoint progress for two configured intervals.

- [ ] Run dry gates:

~~~bash
uv run pytest tests/contract/test_cloud_scripts.py -q
shellcheck scripts/cloud/*.sh
docker build -f docker/Dockerfile.train -t adaptive-math-rl:train .
docker run --gpus all --rm adaptive-math-rl:train uv run python scripts/train/check_backend_contract.py --manifest third_party/manifest.json
~~~

- [ ] Write the runbook from server provisioning through teardown, including secrets, tmux/systemd persistence, sandbox startup, artifact restore, emergency stop and cost ledger.

- [ ] Commit:

~~~bash
git add docker configs/cloud scripts/cloud tests/contract/test_cloud_scripts.py docs/runbooks/cloud-training.md
git commit -m "ops: make cloud training resumable"
~~~

---

### Task 9: Execute the 72-Hour Training Schedule

**Files:**

- Create: docs/results/cloud-run-log.md
- Create: docs/results/experiment-registry.csv
- Generated: artifacts/runs/*/environment.json
- Generated: artifacts/runs/*/resolved_config.yaml
- Generated: artifacts/runs/*/metrics.jsonl
- Generated: artifacts/runs/*/checkpoints/
- Generated: artifacts/runs/*/traces/

- [ ] Before hour 0, upload the tested image, immutable data artifacts and Base/SFT mini-eval cache. Create experiment-registry rows for SFT, R0 and R2 with status planned and exact config hashes.

- [ ] Hours 0–3: run preflight, 1.7B GRPO smoke, 4B one-step smoke, checkpoint restore and 200-trajectory profile. Stop the rental if any correctness gate fails.

- [ ] Hours 3–12: run 4B SFT pilots/final if a gated SFT artifact does not already exist, then evaluate protocol validity and frozen mini-eval. Select one SFT checkpoint and freeze its hash as the shared R0/R2 start.

- [ ] Hours 12–42: run GRPO-R0. Evaluate every fixed training interval on rl_dev and every larger interval on frozen mini-eval. Preserve the last three atomic checkpoints.

- [ ] Hours 42–58: run GRPO-R2 from the identical SFT start, not from R0. Use identical prompts, policy steps and decoding unless the registry explicitly marks a paired deviation.

- [ ] Hours 58–66: use remaining budget for one of these in strict priority: R2 continuation if not converged; R1/R3 short ablation; independent-seed replication. Record why lower-priority items were skipped.

- [ ] Hours 66–70: run final frozen evaluation for Base, SFT, R0 and R2 using the Evaluation Plan. Do not select checkpoints on full test labels; use dev-selected checkpoints.

- [ ] Hours 70–72: sync artifacts, verify checksums from a fresh destination listing, export cost ledger, stop trainer/sandbox/Ray, terminate GPU instance, and mark registry rows completed, failed or stopped with reason.

- [ ] Every two hours append measured progress, ETA, GPU utilization, reward health and decisions to cloud-run-log.md. Never edit earlier entries; add corrections as new entries.

- [ ] Commit only lightweight evidence after download:

~~~bash
git add docs/results/cloud-run-log.md docs/results/experiment-registry.csv data/manifests third_party/manifest.json
git commit -m "docs: record three day training runs"
~~~

## Training Exit Criteria

- [ ] Base/SFT/R0/R2 all have immutable model or adapter revisions and resolved configs.
- [ ] At least one complete R0 and one complete R2 run passed smoke, checkpoint-resume and artifact checksum gates, or the registry explains the exact failed gate.
- [ ] Every rollout used the production Agent runtime and real sandboxed tools.
- [ ] Stored masks prove only model-generated thinking/action tokens contributed to policy loss.
- [ ] Training evidence includes ineffective group rate, invalid rate, verifier errors and tool-cost behavior, not only reward curves.
- [ ] Total GPU hours and direct cloud cost are reported, including failed pilots.

## Primary Implementation References

- verl-agent multi-turn environments, grouped RL and LoRA: https://github.com/langfengQ/verl-agent
- verl Agentic RL rollout architecture: https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst
- verl Qwen3-4B GRPO starting recipe: https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_4b_fsdp.sh
- Qwen3-4B model and chat-template behavior: https://huggingface.co/Qwen/Qwen3-4B
- SandboxFusion execution service: https://github.com/bytedance/SandboxFusion

