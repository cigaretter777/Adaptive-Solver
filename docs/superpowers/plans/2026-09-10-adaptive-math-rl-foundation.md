# AdaptiveMath-RL Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可信的数据、任务协议、答案验证与奖励核心，使后续 Agent 和 RL 训练建立在可测试、不可泄漏、可复现的离线基础上。

**Architecture:** 新建 adaptive_math Python 包，先定义带 public_view 的任务模型，再实现进程隔离的 typed verifier、纯函数 reward 与可追溯数据管线。所有数据处理产生 manifest；所有验证和奖励结果都是结构化对象且对同一输入确定。

**Tech Stack:** Python 3.12、uv、Pydantic v2、SymPy、math-verify、orjson、PyArrow、datasets、datasketch、pytest、Hypothesis、Ruff、mypy。

**Spec:** [AdaptiveMath-RL 产品与技术设计报告](../specs/2026-09-09-adaptive-math-rl-product-design.md)

## Global Constraints

- [ ] 执行本计划前完整阅读 master roadmap 和设计报告第 5、8、9、10、11、12、13、16 节。
- [ ] 测试先行；每个行为先看到目标测试失败，再写最小实现。
- [ ] 原始下载数据只读保存；清洗输出写入 data/processed，并以 manifest 记录 source revision 与 SHA-256。
- [ ] verifier 不导入 Agent、模型或网络客户端；reward 不重新解析答案，只消费 VerifierResult。
- [ ] 任何解析异常、超时或 reference 错误都返回枚举状态，不用布尔 False 掩盖基础设施故障。

---

## File Structure

~~~text
pyproject.toml
.python-version
.gitignore
src/adaptive_math/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── types.py
│   └── hashing.py
├── verifier/
│   ├── __init__.py
│   ├── extractor.py
│   ├── normalizer.py
│   ├── numeric.py
│   ├── symbolic.py
│   ├── worker.py
│   └── service.py
├── reward/
│   ├── __init__.py
│   ├── types.py
│   └── functions.py
└── data/
    ├── __init__.py
    ├── sources.py
    ├── canonicalize.py
    ├── deduplicate.py
    ├── split.py
    └── manifest.py
configs/data/sources.yaml
configs/reward/r0.yaml
configs/reward/r1.yaml
configs/reward/r2.yaml
configs/reward/r3.yaml
scripts/data/build_dataset.py
scripts/data/audit_dataset.py
data/README.md
tests/unit/core/
tests/unit/verifier/
tests/unit/reward/
tests/unit/data/
tests/contract/
tests/fixtures/verifier_cases.jsonl
tests/fixtures/adversarial_answers.jsonl
~~~

---

### Task 1: Bootstrap a Reproducible Python Package

**Files:**

- Create: .python-version
- Create: pyproject.toml
- Create: .gitignore
- Create: src/adaptive_math/__init__.py
- Create: tests/unit/test_package.py
- Modify: README.md

- [ ] Confirm the current directory has no Git metadata before changing anything:

~~~bash
git status --short
~~~

Expected before bootstrap: fatal: not a git repository.

- [ ] Initialize version control and preserve the current prototype as a baseline commit:

~~~bash
git init
git add README.md src tests examples autodl_train.py docs
git commit -m "chore: preserve adaptive solver prototype baseline"
~~~

If Git identity is not configured, set repository-local user.name and user.email, then repeat only the commit command.

- [ ] Write the failing package smoke test:

~~~python
from adaptive_math import __version__


def test_package_version_is_explicit() -> None:
    assert __version__ == "0.1.0"
~~~

- [ ] Run the test and verify collection fails because adaptive_math does not exist:

~~~bash
python3 -m pytest tests/unit/test_package.py -q
~~~

- [ ] Create pyproject.toml with a src-layout package, Python 3.12, and dependency groups:

~~~toml
[project]
name = "adaptive-math-rl"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "pydantic>=2.9,<3",
  "sympy>=1.13,<2",
  "math-verify>=0.7,<1",
  "orjson>=3.10,<4",
  "pyarrow>=17,<20",
  "datasets>=3,<4",
  "datasketch>=1.6,<2",
  "pyyaml>=6,<7",
]

[dependency-groups]
dev = [
  "pytest>=8,<9",
  "pytest-cov>=5,<7",
  "hypothesis>=6.112,<7",
  "ruff>=0.8,<1",
  "mypy>=1.13,<2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/adaptive_math"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers --strict-config"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
packages = ["adaptive_math"]
~~~

- [ ] Set .python-version to 3.12 and ignore only generated artifacts, secrets, local datasets, checkpoints and caches. Keep data/manifests and docs/results tracked.

- [ ] Export __version__ = "0.1.0", install, lock and run the smoke test:

~~~bash
uv sync --dev
uv lock --check
uv run pytest tests/unit/test_package.py -q
~~~

Expected: 1 passed.

- [ ] Add a concise README section with Python version, uv sync, pytest and the four-plan links.

- [ ] Verify formatting and typing:

~~~bash
uv run ruff check src tests
uv run mypy src/adaptive_math
~~~

- [ ] Commit:

~~~bash
git add .python-version .gitignore pyproject.toml uv.lock README.md src/adaptive_math tests/unit/test_package.py
git commit -m "build: bootstrap adaptive math package"
~~~

---

### Task 2: Define Public and Hidden Task Contracts

**Files:**

- Create: src/adaptive_math/core/types.py
- Create: src/adaptive_math/core/hashing.py
- Modify: src/adaptive_math/core/__init__.py
- Test: tests/unit/core/test_types.py
- Test: tests/unit/core/test_hashing.py

- [ ] Write tests that define the boundary:

~~~python
def test_public_view_removes_reference_answer(labeled_task: LabeledMathTask) -> None:
    public = labeled_task.public_view()
    assert isinstance(public, MathTask)
    assert "reference" not in public.model_dump()
    assert "answer" not in public.model_dump()


def test_task_id_is_content_addressed() -> None:
    first = make_task_id("aime_2024", "What is 1+1?")
    second = make_task_id("aime_2024", "  What   is 1+1? ")
    assert first == second
    assert first.startswith("aime_2024:")
~~~

- [ ] Run and confirm import failures:

~~~bash
uv run pytest tests/unit/core/test_types.py tests/unit/core/test_hashing.py -q
~~~

- [ ] Implement these exact public types:

~~~python
class AnswerType(StrEnum):
    INTEGER = "integer"
    RATIONAL = "rational"
    REAL = "real"
    EXPRESSION = "expression"
    SET = "set"
    TUPLE = "tuple"
    INTERVAL = "interval"


class MathTask(BaseModel):
    task_id: str
    problem: str
    answer_type: AnswerType
    dataset: str
    split: str
    source_hash: str
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ReferenceAnswer(BaseModel):
    value: str
    answer_type: AnswerType
    acceptable_forms: tuple[str, ...] = ()


class LabeledMathTask(BaseModel):
    task: MathTask
    reference: ReferenceAnswer

    def public_view(self) -> MathTask: ...


class Budget(BaseModel):
    max_steps: int = Field(gt=0)
    max_tool_calls: int = Field(ge=0)
    max_python_seconds: float = Field(ge=0)
    max_observation_chars: int = Field(gt=0)
~~~

- [ ] Implement canonical_text by Unicode NFKC normalization, line-ending normalization, whitespace collapse and trim. Implement source_hash as SHA-256 of the unmodified source record bytes and task_id as dataset plus the first 20 hex characters of SHA-256(canonical problem).

- [ ] Add JSONValue as a recursive type alias and reject NaN/Infinity in metadata serialization.

- [ ] Add tests for invalid zero budgets, stable JSON round trips, Unicode-equivalent text, and deliberately different math expressions.

- [ ] Run:

~~~bash
uv run pytest tests/unit/core -q
uv run mypy src/adaptive_math/core
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/core tests/unit/core
git commit -m "feat: define math task and budget contracts"
~~~

---

### Task 3: Extract and Normalize Final Answers Without Executing Input

**Files:**

- Create: src/adaptive_math/verifier/extractor.py
- Create: src/adaptive_math/verifier/normalizer.py
- Create: tests/unit/verifier/test_extractor.py
- Create: tests/unit/verifier/test_normalizer.py
- Create: tests/fixtures/verifier_cases.jsonl

- [ ] Write table-driven extractor tests for plain answers, nested boxed LaTeX, dollar delimiters, trailing punctuation, Unicode minus, fractions, sets and intervals. Include these required cases:

~~~python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"Therefore \\boxed{42}.", "42"),
        (r"\\boxed{\\frac{1}{2}}", r"\\frac{1}{2}"),
        ("<final>{\"answer\":\"-3\"}</final>", "-3"),
        ("No final answer", None),
    ],
)
def test_extract_final_answer(raw: str, expected: str | None) -> None:
    assert extract_final_answer(raw) == expected
~~~

- [ ] Add adversarial cases for unbalanced braces, 20,000 nested braces, multiple boxed answers, empty final, duplicate final tags and JSON with extra keys. The extractor must return an ExtractResult with status OK, MISSING or AMBIGUOUS instead of throwing.

- [ ] Run and confirm failures:

~~~bash
uv run pytest tests/unit/verifier/test_extractor.py -q
~~~

- [ ] Implement a single-pass balanced-brace scanner with max_input_chars=32768 and max_nesting=64. Do not use a greedy regular expression for nested LaTeX.

- [ ] Implement normalize_surface only for representation-preserving cleanup: NFKC, trim, Unicode minus to ASCII, remove one outer math delimiter, normalize frac/dfrac/tfrac, and remove safe thousands separators. It must not simplify algebra.

- [ ] Use Hypothesis to prove normalization is idempotent for arbitrary strings up to 2048 characters:

~~~python
@given(st.text(max_size=2048))
def test_surface_normalization_is_idempotent(value: str) -> None:
    once = normalize_surface(value)
    assert normalize_surface(once) == once
~~~

- [ ] Run:

~~~bash
uv run pytest tests/unit/verifier/test_extractor.py tests/unit/verifier/test_normalizer.py -q
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/verifier tests/unit/verifier tests/fixtures/verifier_cases.jsonl
git commit -m "feat: add bounded final answer extraction"
~~~

---

### Task 4: Implement Typed Numeric and Collection Verification

**Files:**

- Create: src/adaptive_math/verifier/numeric.py
- Create: src/adaptive_math/verifier/service.py
- Modify: src/adaptive_math/verifier/__init__.py
- Test: tests/unit/verifier/test_numeric.py
- Test: tests/unit/verifier/test_collections.py

- [ ] Define verifier result semantics in failing tests:

~~~python
class VerifierStatus(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INVALID_PREDICTION = "invalid_prediction"
    INVALID_REFERENCE = "invalid_reference"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


class VerifierResult(BaseModel):
    status: VerifierStatus
    reward: float
    normalized_prediction: str | None
    normalized_reference: str | None
    details: dict[str, JSONValue] = Field(default_factory=dict)
~~~

- [ ] Test exact integers and rationals: 2 equals 2.0 only when the answer type permits numeric coercion; 1/2 equals 0.5; denominator zero is invalid; 1/3 is not equal to 0.333 without a configured tolerance.

- [ ] Test real tolerance using abs(pred-ref) <= atol + rtol * abs(ref), with defaults atol=1e-9 and rtol=1e-9 recorded in details.

- [ ] Test typed collections: sets are order-independent and duplicate-insensitive; tuples are ordered; intervals preserve open/closed endpoints; scalar 2 never equals singleton set {2}.

- [ ] Run and see missing implementations:

~~~bash
uv run pytest tests/unit/verifier/test_numeric.py tests/unit/verifier/test_collections.py -q
~~~

- [ ] Implement Decimal/Fraction parsing with explicit length and exponent limits. Never call float on arbitrary huge strings before enforcing max_digits=256 and max_abs_exponent=1000.

- [ ] Implement recursive collection parsing with max_items=128 and max_depth=8. Reuse typed scalar verification for elements.

- [ ] Implement verify_answer(prediction, reference, config) as the only public verifier entry point; dispatch by ReferenceAnswer.answer_type and always set reward to 1.0 only for CORRECT, otherwise 0.0.

- [ ] Run tests and typing:

~~~bash
uv run pytest tests/unit/verifier -q
uv run mypy src/adaptive_math/verifier
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/verifier tests/unit/verifier
git commit -m "feat: add typed deterministic answer verification"
~~~

---

### Task 5: Isolate Symbolic Verification and Enforce Timeouts

**Files:**

- Create: src/adaptive_math/verifier/symbolic.py
- Create: src/adaptive_math/verifier/worker.py
- Modify: src/adaptive_math/verifier/service.py
- Create: tests/unit/verifier/test_symbolic.py
- Create: tests/contract/test_verifier_worker.py
- Create: tests/fixtures/adversarial_answers.jsonl

- [ ] Write symbolic equivalence tests: x+x equals 2*x; (x^2-1)/(x-1) does not globally equal x+1 because the domains differ; sin(x)^2+cos(x)^2 equals 1; free-symbol mismatch is incorrect; malformed LaTeX is invalid.

- [ ] Write a worker timeout contract test using an injected test comparator that sleeps longer than timeout_seconds. Assert TIMEOUT, elapsed wall time below two times the configured timeout, and successful verification on the next request.

- [ ] Add adversarial inputs for huge integer powers, deeply nested factorials, unknown functions, file-looking strings, Python syntax, shell syntax, network URLs and serialized objects. Assert no file, process or network side effect and status INVALID_PREDICTION or TIMEOUT.

- [ ] Run and confirm failures:

~~~bash
uv run pytest tests/unit/verifier/test_symbolic.py tests/contract/test_verifier_worker.py -q
~~~

- [ ] Implement symbolic comparison in a spawned worker process with per-request timeout_seconds=2.0, memory_mb=512 and cpu_seconds=2. Kill and replace the worker after timeout or crash.

- [ ] Parse through math-verify inside the isolated worker, restrict expression length to 8192, node count to 4096 and free symbols to 32, then compare in this order: canonical equality, SymPy simplify difference, numeric cross-check at five deterministic non-singular points seeded from task_id.

- [ ] Treat numeric cross-check only as rejection/support evidence. Return CORRECT from cross-check only when both expressions are defined at all five points and maximum error is below configured tolerance; record method and points in details.

- [ ] On macOS skip OS-specific memory assertions but still test process timeout; run full resource-limit contract in Linux CI.

- [ ] Run:

~~~bash
uv run pytest tests/unit/verifier tests/contract/test_verifier_worker.py -q
~~~

The test suite must consume every row in tests/fixtures/adversarial_answers.jsonl and assert that its consumed count equals the fixture line count.

- [ ] Commit:

~~~bash
git add src/adaptive_math/verifier tests/unit/verifier tests/contract tests/fixtures/adversarial_answers.jsonl
git commit -m "feat: isolate symbolic verifier in bounded workers"
~~~

---

### Task 6: Implement Pure, Versioned Reward Variants

**Files:**

- Create: src/adaptive_math/reward/types.py
- Create: src/adaptive_math/reward/functions.py
- Modify: src/adaptive_math/reward/__init__.py
- Create: configs/reward/r0.yaml
- Create: configs/reward/r1.yaml
- Create: configs/reward/r2.yaml
- Create: configs/reward/r3.yaml
- Test: tests/unit/reward/test_functions.py

- [ ] Write tests for exact equations and boundary clipping:

~~~python
def test_r2_cost_is_gated_by_correctness() -> None:
    correct = reward_r2(correct_context(tool_calls=2, max_tool_calls=4))
    wrong = reward_r2(wrong_context(tool_calls=2, max_tool_calls=4))
    assert correct.total < 1.0
    assert wrong.components["tool_cost"] == 0.0


def test_reward_is_deterministic(context: RewardContext, config: RewardConfig) -> None:
    assert compute_reward(context, config) == compute_reward(context, config)
~~~

- [ ] Define RewardContext with VerifierResult, tool_calls, python_seconds, invalid_action_count, generated_tokens and Budget. Define RewardBreakdown with total, components, reward_version and config_hash.

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/reward/test_functions.py -q
~~~

- [ ] Implement these exact formulas, with clip to [-1, 1]:

~~~text
R0 = correct
R1 = correct - invalid_weight * min(invalid_count, invalid_cap)
R2 = correct * (1 - tool_weight * tool_calls/max_tool_calls
                    - python_weight * python_seconds/max_python_seconds)
     - invalid_weight * min(invalid_count, invalid_cap)
R3 = R2 - correct * token_weight * generated_tokens/max_generated_tokens
~~~

If a budget denominator is zero, its cost component is exactly zero. correct is 1 only for VerifierStatus.CORRECT.

- [ ] Create YAML configs with initial hypothesis values, not hidden defaults:

~~~yaml
version: r2-v1
variant: r2
tool_weight: 0.15
python_weight: 0.10
invalid_weight: 0.10
invalid_cap: 3
token_weight: 0.0
clip_min: -1.0
clip_max: 1.0
~~~

R0 sets every weight to zero; R1 enables only invalid_weight; R3 starts from R2 and sets token_weight to 0.05.

- [ ] Add property tests: total is finite and bounded, increasing cost cannot increase a correct R2 reward, and incorrect valid trajectories receive exactly zero under R0/R2.

- [ ] Run:

~~~bash
uv run pytest tests/unit/reward -q
uv run mypy src/adaptive_math/reward
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/reward configs/reward tests/unit/reward
git commit -m "feat: add versioned reward ablations"
~~~

---

### Task 7: Build Traceable Dataset Canonicalization and Splits

**Files:**

- Create: configs/data/sources.yaml
- Create: src/adaptive_math/data/sources.py
- Create: src/adaptive_math/data/canonicalize.py
- Create: src/adaptive_math/data/deduplicate.py
- Create: src/adaptive_math/data/split.py
- Create: src/adaptive_math/data/manifest.py
- Create: scripts/data/build_dataset.py
- Create: scripts/data/audit_dataset.py
- Create: data/README.md
- Test: tests/unit/data/test_canonicalize.py
- Test: tests/unit/data/test_deduplicate.py
- Test: tests/unit/data/test_split.py
- Test: tests/contract/test_data_manifest.py

- [ ] Define source registry tests requiring name, URI, immutable revision, license, intended_use and citation for OpenR1-Math-220k, DAPO-Math-17k, NuminaMath, MATH, MATH-500, AIME and Omni-MATH.

- [ ] Write canonicalization tests that map source-specific records to LabeledMathTask and quarantine missing, ambiguous or unverifiable answers with a machine-readable reason.

- [ ] Write split tests proving that a seeded rebuild is byte-identical and that frozen_eval task IDs can never appear in train, sft_dev or rl_dev.

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/data tests/contract/test_data_manifest.py -q
~~~

- [ ] Implement exact deduplication on canonical problem SHA-256, then near-deduplication using character 5-gram MinHash with threshold 0.90. For cross-split clusters, keep the evaluation member and remove all training members.

- [ ] Add answer verification during build. Keep only CORRECT reference self-checks; quarantine INVALID_REFERENCE, TIMEOUT and parser failures. Do not infer labels from model output in this pipeline.

- [ ] Implement deterministic split materialization to Parquet plus manifest containing source revisions, input hashes, filter counts, dedup counts, seed, schema version, pipeline Git SHA and output hashes.

- [ ] The build CLI must support dry-run and explicit output roots:

~~~bash
uv run python scripts/data/build_dataset.py \
  --registry configs/data/sources.yaml \
  --output-dir data/processed/v1 \
  --manifest data/manifests/v1.json \
  --seed 20260910 \
  --dry-run
~~~

- [ ] Implement audit CLI that exits nonzero for hash mismatch, split leakage, duplicate task IDs, unknown license, unverifiable reference or schema mismatch.

- [ ] Run against synthetic fixtures first, then a 100-record-per-source sample:

~~~bash
uv run pytest tests/unit/data tests/contract/test_data_manifest.py -q
uv run python scripts/data/build_dataset.py --registry configs/data/sources.yaml --output-dir /tmp/adaptive_math_sample --manifest /tmp/adaptive_math_sample.json --seed 20260910 --sample-per-source 100
uv run python scripts/data/audit_dataset.py --manifest /tmp/adaptive_math_sample.json
~~~

- [ ] Commit source code, registry and tests; never commit downloaded datasets:

~~~bash
git add configs/data src/adaptive_math/data scripts/data data/README.md tests/unit/data tests/contract/test_data_manifest.py
git commit -m "feat: add reproducible math data pipeline"
~~~

---

### Task 8: Add Foundation Quality Gates and Migration Notes

**Files:**

- Create: .github/workflows/ci.yml
- Create: docs/runbooks/data-and-verifier.md
- Create: docs/results/foundation-validation.md
- Modify: README.md
- Test: tests/contract/test_no_hidden_label_leakage.py

- [ ] Write a leakage contract test that serializes MathTask, every public Agent-facing model from the runtime plan if present, and dataset training prompts. Fail on keys matching answer, reference, ground_truth, expected_output or verifier_handle.

- [ ] Configure CI jobs for Python 3.12 on Linux: lint/type, unit tests, verifier adversarial tests and manifest contract. Cache uv downloads but not generated data.

- [ ] Add commands and expected artifacts to the runbook. Document how to add a dataset source, rebuild a manifest, inspect quarantine rows, rotate a verifier version and reproduce one verification decision.

- [ ] Run the complete foundation gate:

~~~bash
uv lock --check
uv run ruff check src tests scripts
uv run mypy src/adaptive_math
uv run pytest tests/unit tests/contract -q --cov=adaptive_math --cov-report=term-missing
uv run python scripts/data/audit_dataset.py --manifest data/manifests/v1.json
~~~

- [ ] Record exact command outputs, platform and timestamp in docs/results/foundation-validation.md. Report coverage by module; do not turn a global percentage into the sole quality claim.

- [ ] Search for unfinished implementation markers and inspect every hit:

~~~bash
rg -n "TODO|TBD|FIXME|NotImplementedError|pass$|random\.random|np\.random" src/adaptive_math scripts tests configs
~~~

- [ ] Review all new public types against the master cross-plan contract. Update the master and dependent plans in the same commit if a type changed.

- [ ] Commit:

~~~bash
git add .github/workflows/ci.yml docs/runbooks/data-and-verifier.md docs/results/foundation-validation.md README.md tests/contract/test_no_hidden_label_leakage.py
git commit -m "test: enforce foundation quality gates"
~~~

## Foundation Exit Criteria

- [ ] All Task 1–8 tests and commands pass from a clean uv environment.
- [ ] Every public data row is traceable to a pinned source revision and content hash.
- [ ] Every verifier outcome distinguishes wrong answer from invalid input, timeout and infrastructure error.
- [ ] R0–R3 are pure, versioned, separately configurable functions with property tests.
- [ ] No Agent-facing serialization can expose the hidden reference answer.
- [ ] The next implementer can build the Agent runtime without touching data-source-specific formats.
