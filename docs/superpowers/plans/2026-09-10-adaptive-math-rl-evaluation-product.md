# AdaptiveMath-RL Evaluation and Product Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用冻结、成对、可复现的评测证明 Agentic RL 是否真正提升数学正确率和预算适应性，并把同一 Agent 核心包装成无答案泄漏的 API、轨迹演示和完整求职证据包。

**Architecture:** BenchmarkRegistry 固定数据与切分，EvaluationRunner 用统一模型客户端和 Agent runtime 运行 Base/SFT/R0/R2 与工具策略基线，统计层从逐题记录生成置信区间、显著性和 Pareto 结果。产品层通过 FastAPI 调用同一 AgentLoop，SSE 输出事件，Gradio 只做展示；线上自检不接触 reference，离线 evaluator 独立部署。

**Tech Stack:** Python 3.12、PyArrow/Pandas、NumPy、SciPy、Matplotlib/Seaborn、FastAPI、Pydantic、Uvicorn、Gradio、httpx、Prometheus/OpenTelemetry、pytest、Playwright 可选。

**Spec:** [AdaptiveMath-RL 产品与技术设计报告](../specs/2026-09-09-adaptive-math-rl-product-design.md)

## Global Constraints

- [ ] 测试集只用于最终报告；checkpoint、reward 系数、prompt 和预算配置只在 train/rl_dev/mini-eval 上选择。
- [ ] 比较模型使用相同题目顺序、模型上下文上限、答案 verifier 版本和可比的生成预算；任何差异进入 resolved eval config。
- [ ] 逐题 prediction record 是事实源；表格和图均由脚本生成，不能手工复制数字。
- [ ] 报告 accuracy 的同时报告置信区间、样本数、invalid、工具成本、token 和延迟。
- [ ] 产品请求、消息、提示和 trace schema 不包含 ground_truth、reference_answer、reward 或 verifier handle。
- [ ] 产品中“格式有效”或“工具复核完成”不得展示为“答案正确”。
- [ ] 如果 RL 未显著超过 SFT，发布负结果和失败诊断，不用某个子集或单次 seed 取代主结论。

---

## File Structure

~~~text
configs/eval/
├── frozen_v1.yaml
├── mini_dev.yaml
├── decoding_greedy.yaml
├── decoding_pass8.yaml
└── budget_sweep.yaml
src/adaptive_math/
├── evaluation/
│   ├── __init__.py
│   ├── registry.py
│   ├── baselines.py
│   ├── runner.py
│   ├── records.py
│   ├── metrics.py
│   ├── statistics.py
│   ├── failure_taxonomy.py
│   ├── report.py
│   └── plots.py
└── serving/
    ├── __init__.py
    ├── schemas.py
    ├── policy_client.py
    ├── session_store.py
    ├── service.py
    ├── api.py
    ├── telemetry.py
    └── security.py
scripts/
├── eval/run_benchmark.py
├── eval/audit_benchmark.py
├── eval/compare_runs.py
├── eval/build_report.py
└── serve/run_api.py
demo/
├── gradio_app.py
└── examples.json
docs/
├── model-card.md
├── evaluation-card.md
├── demo-script.md
├── results/final-report.md
└── results/figures/
tests/unit/evaluation/
tests/unit/serving/
tests/contract/test_frozen_eval.py
tests/contract/test_serving_no_labels.py
tests/integration/test_evaluation_runner.py
tests/integration/test_serving_api.py
tests/integration/test_demo_smoke.py
~~~

---

### Task 1: Freeze the Benchmark Registry and Contamination Audit

**Files:**

- Create: src/adaptive_math/evaluation/registry.py
- Create: configs/eval/frozen_v1.yaml
- Create: configs/eval/mini_dev.yaml
- Create: scripts/eval/audit_benchmark.py
- Create: tests/unit/evaluation/test_registry.py
- Create: tests/contract/test_frozen_eval.py
- Create: docs/evaluation-card.md

- [ ] Define BenchmarkSpec with name, source URI, immutable revision, license, split, answer types, inclusion filter, task_count, task_ids_hash and labels_hash. labels_hash is accessible only to offline evaluation commands.

- [ ] Write registry tests that reject mutable revisions, duplicate task IDs, overlap with any SFT/RL manifest, unknown licenses, unsupported answer types and post-freeze edits without a version bump.

- [ ] Freeze v1 as separately reported slices, not one blended score:

~~~text
MATH-500                     full verifier-compatible set
AIME 2024                    full numeric-answer set
AIME 2025                    full numeric-answer set
AMC                          pinned verifier-compatible subset
Omni-MATH                    pinned rule-verifiable subset
AIME 2026                    strict temporal holdout
Parameterized-Holdout-v1     200 project-generated, human-reviewed items
~~~

- [ ] Run failing tests before implementation:

~~~bash
uv run pytest tests/unit/evaluation/test_registry.py tests/contract/test_frozen_eval.py -q
~~~

- [ ] Implement exact SHA and near-duplicate checks against SFT/RL problem text, plus normalized formula/template fingerprints. Quarantine any collision and publish collision counts per source.

- [ ] Add contamination tiers to each benchmark: likely seen in pretraining, public-after-model-cutoff, or project-private. Do not claim that public MATH/AIME is contamination-free.

- [ ] Implement audit command that verifies remote/local source hashes, split isolation, answer self-verification and frozen config signature:

~~~bash
uv run python scripts/eval/audit_benchmark.py --config configs/eval/frozen_v1.yaml --train-manifest data/manifests/v1.json --sft-manifest data/manifests/sft_v1.json
~~~

- [ ] Document benchmark purpose, known contamination risk, verifier coverage, exclusions and label access policy in evaluation-card.md.

- [ ] Commit config, registry, hashes and documentation; do not commit restricted labels:

~~~bash
git add src/adaptive_math/evaluation/registry.py configs/eval scripts/eval/audit_benchmark.py tests/unit/evaluation/test_registry.py tests/contract/test_frozen_eval.py docs/evaluation-card.md
git commit -m "test: freeze math evaluation suite"
~~~

---

### Task 2: Implement Fair Baselines and Prediction Records

**Files:**

- Create: src/adaptive_math/evaluation/baselines.py
- Create: src/adaptive_math/evaluation/records.py
- Create: src/adaptive_math/evaluation/runner.py
- Create: configs/eval/decoding_greedy.yaml
- Create: configs/eval/decoding_pass8.yaml
- Create: configs/eval/budget_sweep.yaml
- Create: scripts/eval/run_benchmark.py
- Create: tests/unit/evaluation/test_baselines.py
- Create: tests/unit/evaluation/test_records.py
- Create: tests/integration/test_evaluation_runner.py
- Modify: pyproject.toml

- [ ] Define these evaluation systems explicitly:

~~~text
Base-Direct    Qwen3-4B, no tools, one terminal answer
Base-Agent     Qwen3-4B, production AgentLoop and learned tool choice
Never-Tool     evaluated checkpoint with max_tool_calls=0
Always-Python  same checkpoint, prompt requires one Python call before FINAL
Fixed-K2       same checkpoint, at most two tool calls then forced FINAL prompt
SFT-Agent      selected SFT adapter, production AgentLoop
GRPO-R0        selected R0 adapter, production AgentLoop
GRPO-R2        selected R2 adapter, production AgentLoop
~~~

- [ ] Baseline tests assert that wrappers change only the declared tool policy/budget, retain the same model revision and prompt version, and never see labels.

- [ ] Define PredictionRecord fields: run_id, system_id, checkpoint_hash, task_id, sample_id, decode_seed, budget_id, raw final, normalized final, verifier status/method, correct, generated tokens, steps, tool calls by name/status, Python seconds, latency, termination, trace URI, config hash and code SHA.

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/evaluation/test_baselines.py tests/unit/evaluation/test_records.py tests/integration/test_evaluation_runner.py -q
~~~

- [ ] Implement EvaluationRunner with deterministic sharding and resume. It writes one atomic Parquet shard per worker, rejects duplicate prediction keys and merges only shards with identical registry/config/code hashes.

- [ ] Add an evaluation optional dependency set containing numpy, scipy, pandas, matplotlib and seaborn, then refresh and check uv.lock.

- [ ] Configure primary pass@1 decoding as greedy where supported: temperature 0, one sample, fixed max trajectory tokens. Configure pass@8 separately: eight seeds, temperature 0.7, top_p 0.95. Never mix pass@8 samples into pass@1.

- [ ] Configure budget sweep B0/B1/B2/B4 with max tool calls 0/1/2/4 while holding max steps, Python seconds and decoding fixed except where zero tools implies zero Python time.

- [ ] Run the synthetic integration suite, then a five-task real model smoke:

~~~bash
uv run pytest tests/integration/test_evaluation_runner.py -q
uv run python scripts/eval/run_benchmark.py --benchmark configs/eval/mini_dev.yaml --decoding configs/eval/decoding_greedy.yaml --systems base_direct,base_agent --limit 5 --output artifacts/eval/smoke
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/evaluation/baselines.py src/adaptive_math/evaluation/records.py src/adaptive_math/evaluation/runner.py configs/eval scripts/eval/run_benchmark.py tests/unit/evaluation tests/integration/test_evaluation_runner.py pyproject.toml uv.lock
git commit -m "feat: run fair adaptive math baselines"
~~~

---

### Task 3: Compute Metrics, Confidence Intervals and Pareto Frontiers

**Files:**

- Create: src/adaptive_math/evaluation/metrics.py
- Create: src/adaptive_math/evaluation/statistics.py
- Create: src/adaptive_math/evaluation/plots.py
- Create: scripts/eval/compare_runs.py
- Create: tests/unit/evaluation/test_metrics.py
- Create: tests/unit/evaluation/test_statistics.py
- Create: tests/unit/evaluation/test_plots.py

- [ ] Write hand-calculated tests for accuracy, protocol validity, invalid rate, verifier failure rate, average tool calls, tool success, Python seconds, generated tokens, latency quantiles and termination distribution.

- [ ] Write pass@k tests using the unbiased estimator 1 - C(n-c,k)/C(n,k), including c=0, c=n and k>n validation.

- [ ] Write paired-statistics tests on synthetic records with known deltas. Use 10,000 task-level paired bootstrap resamples seeded by config for a 95% CI and exact McNemar test for pass@1 correctness disagreements.

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/evaluation/test_metrics.py tests/unit/evaluation/test_statistics.py tests/unit/evaluation/test_plots.py -q
~~~

- [ ] Implement all aggregates by benchmark and by domain/difficulty/answer type. Preserve numerator and denominator for every percentage; infrastructure failures appear separately and are not silently counted as ordinary incorrect answers.

- [ ] Define compute cost axes as generated tokens, tool calls, Python seconds and end-to-end latency. A system is Pareto-dominated if another has at least equal accuracy and no greater value on every selected cost axis, with one strict improvement.

- [ ] Generate these mandatory figures from prediction Parquet: accuracy with CI, Base-to-SFT-to-RL deltas, accuracy vs tool budget, accuracy vs generated tokens, tool-use distribution, invalid/verifier failure, latency distribution, difficulty breakdown and accuracy-cost Pareto frontier.

- [ ] compare_runs.py refuses comparisons without identical benchmark/task hashes and prints both absolute point delta and relative error reduction.

~~~bash
uv run python scripts/eval/compare_runs.py --baseline artifacts/eval/sft/predictions.parquet --candidate artifacts/eval/r2/predictions.parquet --output artifacts/eval/comparisons/sft_vs_r2.json --bootstrap-samples 10000 --seed 20260910
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/evaluation/metrics.py src/adaptive_math/evaluation/statistics.py src/adaptive_math/evaluation/plots.py scripts/eval/compare_runs.py tests/unit/evaluation
git commit -m "feat: quantify rl gains and tool cost"
~~~

---

### Task 4: Add Failure Taxonomy, Trace Cases and Generated Reports

**Files:**

- Create: src/adaptive_math/evaluation/failure_taxonomy.py
- Create: src/adaptive_math/evaluation/report.py
- Create: scripts/eval/build_report.py
- Create: tests/unit/evaluation/test_failure_taxonomy.py
- Create: tests/integration/test_report_generation.py
- Create: docs/results/final-report.md
- Create: docs/results/figures/.gitkeep

- [ ] Define nonexclusive machine labels: FORMAT_INVALID, NO_FINAL, WRONG_REASONING, WRONG_TOOL_CHOICE, TOOL_CODE_ERROR, TOOL_TIMEOUT, TOOL_RESULT_IGNORED, PREMATURE_STOP, BUDGET_EXHAUSTED, VERIFIER_UNSUPPORTED, VERIFIER_TIMEOUT and INFRASTRUCTURE_FAILURE.

- [ ] Write rule tests from golden traces. Machine labels must be reproducible; human-review labels live in separate annotation fields with annotator ID and adjudication status.

- [ ] Select trace cases deterministically from paired outcomes: ten SFT-wrong/R2-correct, ten SFT-correct/R2-wrong, five no-tool wins, five necessary-tool wins, five recovery wins, five reward-hacking candidates and all infrastructure failures up to 50.

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/evaluation/test_failure_taxonomy.py tests/integration/test_report_generation.py -q
~~~

- [ ] Implement build_report.py to read experiment registry, comparison JSON and figures, validate every cited number against its source, and render Markdown plus machine-readable summary.json.

- [ ] The report structure is fixed: research question, systems, data, training cost, primary results, budget/Pareto results, ablations, failure analysis, representative traces, threats to validity, negative results, limitations and reproduction commands.

- [ ] Never bake expected gains into tests. Report-generation tests use synthetic inputs and assert arithmetic/link integrity, not a favorable model ranking.

- [ ] Generate the report after final evaluation:

~~~bash
uv run python scripts/eval/build_report.py --registry docs/results/experiment-registry.csv --eval-root artifacts/eval/frozen_v1 --output docs/results/final-report.md --figures docs/results/figures
~~~

- [ ] Commit source and lightweight report artifacts. Large raw predictions and traces remain in versioned artifact storage with checksummed links.

~~~bash
git add src/adaptive_math/evaluation/failure_taxonomy.py src/adaptive_math/evaluation/report.py scripts/eval/build_report.py tests/unit/evaluation/test_failure_taxonomy.py tests/integration/test_report_generation.py docs/results
git commit -m "docs: generate evidence backed evaluation report"
~~~

---

### Task 5: Build a Label-Free FastAPI Product Service

**Files:**

- Create: src/adaptive_math/serving/schemas.py
- Create: src/adaptive_math/serving/policy_client.py
- Create: src/adaptive_math/serving/session_store.py
- Create: src/adaptive_math/serving/service.py
- Create: src/adaptive_math/serving/api.py
- Create: scripts/serve/run_api.py
- Create: tests/unit/serving/test_schemas.py
- Create: tests/unit/serving/test_service.py
- Create: tests/contract/test_serving_no_labels.py
- Create: tests/integration/test_serving_api.py
- Modify: pyproject.toml

- [ ] Define public request and response models:

~~~python
class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    problem: str = Field(min_length=1, max_length=16000)
    answer_type: AnswerType
    budget: Budget
    profile: Literal["fast", "balanced", "thorough"] = "balanced"


class SolveAccepted(BaseModel):
    session_id: str
    events_url: str
    cancel_url: str


class SolveResult(BaseModel):
    session_id: str
    final_answer: str | None
    termination_reason: TerminationReason
    usage: Usage
    structural_check: Literal["valid", "invalid", "not_available"]
~~~

- [ ] Schema tests must reject answer, reference, ground_truth, expected_answer, reward and verifier fields because extra=forbid.

- [ ] Implement POST /v1/solve, GET /v1/sessions/{id}, GET /v1/sessions/{id}/events as server-sent events, POST /v1/sessions/{id}/cancel and GET /healthz.

- [ ] PolicyClient speaks to a pinned vLLM/SGLang inference endpoint but implements the same ModelClient protocol used locally. Propagate request ID, timeout and cancellation; never log authentication headers.

- [ ] SessionStore persists public trace events and status with TTL. It never stores a labeled task, hidden evaluation record or reward. Initial implementation may use SQLite in WAL mode for a single-node demo; keep an interface for Redis/Postgres migration.

- [ ] Structural self-check only validates final-answer parseability and can optionally rerun a disclosed calculation supplied by the Agent. It never compares against a hidden answer and the UI copy explicitly says it is not a correctness guarantee.

- [ ] Add a serving optional dependency set containing fastapi, uvicorn, gradio, prometheus-client and OpenTelemetry packages, then refresh and check uv.lock.

- [ ] Run failing tests, then implementation:

~~~bash
uv run pytest tests/unit/serving tests/contract/test_serving_no_labels.py tests/integration/test_serving_api.py -q
~~~

- [ ] Add integration cases for success, invalid model actions, tool timeout, budget exhaustion, client disconnect, cancellation, policy unavailable, sandbox unavailable and concurrent sessions.

- [ ] Commit:

~~~bash
git add src/adaptive_math/serving scripts/serve tests/unit/serving tests/contract/test_serving_no_labels.py tests/integration/test_serving_api.py pyproject.toml uv.lock
git commit -m "feat: serve label free math agent api"
~~~

---

### Task 6: Add Security, Telemetry and Operational Limits

**Files:**

- Create: src/adaptive_math/serving/security.py
- Create: src/adaptive_math/serving/telemetry.py
- Create: tests/unit/serving/test_security.py
- Create: tests/integration/test_telemetry.py
- Create: docs/runbooks/serving.md

- [ ] Write tests for request size, per-IP concurrency, session ownership, event redaction, prompt-injection strings, code containing secrets-looking values and cancellation cleanup.

- [ ] Implement API-key authentication for nonlocal binding, token-bucket rate limits, max concurrent sessions, queue timeout, CSP/CORS allowlist and correlation IDs. Defaults bind to 127.0.0.1 and deny cross-origin requests.

- [ ] Redact configured secret patterns and environment values before logs/traces. Do not send application secrets to SandboxFusion or model messages.

- [ ] Emit Prometheus/OpenTelemetry metrics for request outcome, queue time, model latency, step count, tool latency/status, budget termination, cancellation and active sessions. Never use problem text or answer as a metric label.

- [ ] Add startup dependency checks for policy endpoint and sandbox. healthz reports ready=false if required dependencies fail; it does not expose internal URLs or credentials.

- [ ] Run:

~~~bash
uv run pytest tests/unit/serving/test_security.py tests/integration/test_telemetry.py -q
uv run ruff check src/adaptive_math/serving
uv run mypy src/adaptive_math/serving
~~~

- [ ] Document local, single-GPU and split policy/sandbox deployments; include TLS reverse proxy, secret injection, trace retention and incident cleanup.

- [ ] Commit:

~~~bash
git add src/adaptive_math/serving/security.py src/adaptive_math/serving/telemetry.py tests/unit/serving/test_security.py tests/integration/test_telemetry.py docs/runbooks/serving.md
git commit -m "feat: harden math agent serving"
~~~

---

### Task 7: Build the Trace-Centered Demo

**Files:**

- Create: demo/gradio_app.py
- Create: demo/examples.json
- Create: tests/integration/test_demo_smoke.py
- Create: docs/demo-script.md
- Modify: README.md

- [ ] Write a demo smoke test with a fake API that submits one problem, consumes SSE events, renders a tool result, displays final answer and surfaces budget termination.

- [ ] Build a minimal Gradio interface with problem input, answer type, fast/balanced/thorough profile, explicit budget controls, run/cancel buttons, final answer and an append-only trace timeline.

- [ ] For every turn show thinking only when the selected model/license/product policy permits it; otherwise show a concise action summary. Always show tool name, sanitized input, output/error, latency, remaining budget and termination reason.

- [ ] Add six fixed demo examples: direct arithmetic, Python enumeration, SymPy algebra, tool failure recovery, budget-limited problem and one honest failure. examples.json contains no hidden answers.

- [ ] Show model/checkpoint label, prompt version, budget and a disclaimer that public self-check is structural. Do not show offline reward in the product UI.

- [ ] Run:

~~~bash
uv run pytest tests/integration/test_demo_smoke.py -q
uv run python demo/gradio_app.py --api-url http://127.0.0.1:8000
~~~

- [ ] Create a 3–5 minute demo script: motivation, architecture, one direct solve, one adaptive tool solve, one recovery, trace replay, Base/SFT/RL comparison, limitations and reproduction link.

- [ ] Update README hero section with project claim phrased as a measured research question until final results exist. Add architecture, quickstart, evidence links, screenshots and explicit non-goals.

- [ ] Commit:

~~~bash
git add demo tests/integration/test_demo_smoke.py docs/demo-script.md README.md
git commit -m "feat: demonstrate adaptive math trajectories"
~~~

---

### Task 8: Produce the Final Job-Quality Evidence Package

**Files:**

- Create: docs/model-card.md
- Modify: docs/evaluation-card.md
- Modify: docs/results/final-report.md
- Modify: README.md
- Create: CITATION.cff
- Create: LICENSE
- Create: scripts/eval/verify_release.py
- Create: tests/contract/test_release_evidence.py

- [ ] Write release contract tests that parse every reported table value/link, confirm referenced artifacts and hashes exist, verify no secrets or hidden labels are tracked, and reject unsupported superiority claims.

- [ ] Complete model-card.md with base model/license, adapters, intended use, out-of-scope use, datasets, training method, reward equations, cloud hardware/time/cost, evaluation, limitations, verifier risks and ethical/security considerations.

- [ ] Make README answer these interview questions in under two screens: what problem, why Agent instead of longer chain-of-thought, why RL instead of SFT only, what is the environment, how reward avoids tool-saving exploits, how labels are isolated, what improved, what did not, and how much it cost.

- [ ] Include one architecture diagram, one training dataflow, one primary result table, one Pareto plot and links to at least ten replayable traces. All visuals must be generated from repository data or versioned source files.

- [ ] Implement release verification:

~~~bash
uv run python scripts/eval/verify_release.py --report docs/results/final-report.md --model-card docs/model-card.md --registry docs/results/experiment-registry.csv --artifact-root artifacts
~~~

- [ ] Run the full project gate from a clean environment:

~~~bash
uv sync --all-extras
uv lock --check
uv run ruff check src scripts tests demo
uv run mypy src/adaptive_math
uv run pytest tests/unit tests/contract tests/integration -q
uv run python scripts/eval/audit_benchmark.py --config configs/eval/frozen_v1.yaml --train-manifest data/manifests/v1.json --sft-manifest data/manifests/sft_v1.json
uv run python scripts/eval/verify_release.py --report docs/results/final-report.md --model-card docs/model-card.md --registry docs/results/experiment-registry.csv --artifact-root artifacts
~~~

- [ ] Search tracked files for secrets, labels and unfinished production markers; inspect every hit and document any intentional test fixture:

~~~bash
git grep -n -E "AKIA|BEGIN PRIVATE KEY|ground_truth|reference_answer|NotImplementedError|FIXME"
~~~

- [ ] Tag only after all evidence gates pass:

~~~bash
git add README.md docs CITATION.cff LICENSE scripts/eval/verify_release.py tests/contract/test_release_evidence.py
git commit -m "release: publish adaptive math rl evidence"
git tag -a v1.0.0 -m "AdaptiveMath-RL v1.0.0"
~~~

## Evaluation and Product Exit Criteria

- [ ] Base-Direct、Base-Agent、Never-Tool、Always-Python、Fixed-K2、SFT、R0、R2 在同一冻结任务哈希上可比较。
- [ ] 主结果同时有逐题数据、95% CI、paired test、成本轴和失败案例。
- [ ] 产品 API、存储、日志、指标和 demo 均不含隐藏标签或 reward。
- [ ] 每个外部数字都能追溯到 prediction record、resolved config、checkpoint、code SHA 和 data hash。
- [ ] README 和报告区分已实现、已测得、推断与未来工作。
- [ ] 即使 RL 结果为负，项目仍形成一套可信的 Agentic RL 环境、评测和失败分析资产。
