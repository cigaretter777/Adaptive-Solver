# AdaptiveMath-RL Agent Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个训练与产品共用、确定性可回放、预算可控的单策略数学 Agent 运行时，让模型通过严格 TOOL_CALL/FINAL 协议使用 Python 与 SymPy，并把所有异常转换成可学习的结构化观察。

**Architecture:** AgentLoop 只负责编排 ModelClient 与 MathAgentEnv；环境持有状态、预算和工具注册表；动作解析器把一个可选 think 段和唯一动作解析为结构化结果；Python 执行只通过 SandboxFusion HTTP API；离线训练环境可额外注入 HiddenVerifier，产品环境在类型与构造入口上都不能得到 reference。

**Tech Stack:** Python 3.12、Pydantic v2、httpx、SymPy、SandboxFusion、Transformers、orjson、pytest、respx、Hypothesis。

**Spec:** [AdaptiveMath-RL 产品与技术设计报告](../specs/2026-09-09-adaptive-math-rl-product-design.md)

## Global Constraints

- [ ] 开始前 Foundation Exit Criteria 必须全部通过。
- [ ] 每个 assistant turn 可有一个可选 think 段，但顶层动作只能有一个；不能同时调用工具和提交答案。
- [ ] 环境永不把 reference、verifier status 或 reward 写入下一轮模型 observation。
- [ ] 工具错误是观察，运行时基础设施错误是明确终止原因，两者不能混成空字符串。
- [ ] 所有超时同时有客户端 deadline 和服务端资源限制；客户端超时后不得自动转本地执行。
- [ ] replay 只消费记录下来的模型动作和工具结果，不重新调用模型或外部工具。

---

## File Structure

~~~text
src/adaptive_math/
├── agent/
│   ├── __init__.py
│   ├── actions.py
│   ├── parser.py
│   ├── state.py
│   ├── environment.py
│   ├── loop.py
│   ├── prompts.py
│   ├── model_client.py
│   ├── trace.py
│   └── replay.py
├── tools/
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── sympy_tool.py
│   ├── sandboxfusion.py
│   └── python_tool.py
└── verifier/
    └── hidden.py
configs/agent/default.yaml
configs/agent/direct.yaml
configs/agent/tool_budget_2.yaml
docker/compose.sandbox.yml
scripts/dev/run_agent.py
scripts/dev/replay_trace.py
tests/unit/agent/
tests/unit/tools/
tests/contract/test_sandboxfusion.py
tests/contract/test_hidden_verifier_boundary.py
tests/integration/test_agent_loop.py
tests/integration/test_trace_replay.py
tests/fixtures/agent_actions.jsonl
tests/fixtures/golden_traces/
~~~

---

### Task 1: Implement the Strict Action Schema and Parser

**Files:**

- Create: src/adaptive_math/agent/actions.py
- Create: src/adaptive_math/agent/parser.py
- Modify: src/adaptive_math/agent/__init__.py
- Create: tests/unit/agent/test_actions.py
- Create: tests/unit/agent/test_parser.py
- Create: tests/fixtures/agent_actions.jsonl

- [ ] Write schema tests for exactly these types:

~~~python
class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    arguments: dict[str, JSONValue]


class ToolAction(BaseModel):
    kind: Literal["tool_call"] = "tool_call"
    call: ToolCall


class FinalAction(BaseModel):
    kind: Literal["final"] = "final"
    answer: str = Field(min_length=1, max_length=8192)


AgentAction = Annotated[ToolAction | FinalAction, Field(discriminator="kind")]
~~~

- [ ] Add successful parser cases for an action-only turn and a turn containing one complete think block followed by one tool_call/final tag, including JSON escaping and multiline Python code.

- [ ] Add failure cases for free prose outside tags, duplicate or unclosed think tags, think after action, duplicate actions, nested action tags, unknown keys, non-object arguments, unknown root tags, empty output, incomplete JSON and payloads larger than 32768 characters.

- [ ] Define ParseResult with reasoning, action or error, raw_hash and ParseErrorCode. The codes are EMPTY, TOO_LONG, INVALID_ENVELOPE, MULTIPLE_ACTIONS, INVALID_JSON and INVALID_SCHEMA. reasoning is None when think is absent and is length-limited to 24576 characters.

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/agent/test_actions.py tests/unit/agent/test_parser.py -q
~~~

- [ ] Implement parse_action as an anchored, length-bounded parser. Strip only outer whitespace; accept grammar (think)? (tool_call|final), match exactly one complete action element, use orjson.loads, validate with Pydantic TypeAdapter, and preserve no executable object from input.

- [ ] Add a Hypothesis test that parse_action never raises for arbitrary strings up to 40000 characters.

- [ ] Run:

~~~bash
uv run pytest tests/unit/agent/test_actions.py tests/unit/agent/test_parser.py -q
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/agent tests/unit/agent tests/fixtures/agent_actions.jsonl
git commit -m "feat: add strict agent action protocol"
~~~

---

### Task 2: Define Agent State, Events and Termination Semantics

**Files:**

- Create: src/adaptive_math/agent/state.py
- Create: src/adaptive_math/agent/trace.py
- Create: tests/unit/agent/test_state.py
- Create: tests/unit/agent/test_trace.py

- [ ] Write tests for immutable sequence numbers, monotonic timestamps, usage accumulation, JSON round trip and budget exhaustion.

- [ ] Define exact enums and models:

~~~python
class TerminationReason(StrEnum):
    FINAL = "final"
    MAX_STEPS = "max_steps"
    MAX_TOOL_CALLS = "max_tool_calls"
    PYTHON_TIME_BUDGET = "python_time_budget"
    MODEL_ERROR = "model_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    CANCELLED = "cancelled"


class EventKind(StrEnum):
    MODEL_OUTPUT = "model_output"
    INVALID_ACTION = "invalid_action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL = "final"
    TERMINATION = "termination"


class Usage(BaseModel):
    steps: int = 0
    tool_calls: int = 0
    python_seconds: float = 0.0
    generated_tokens: int = 0
    invalid_actions: int = 0


class TraceEvent(BaseModel):
    sequence: int
    kind: EventKind
    monotonic_ms: int
    payload: dict[str, JSONValue]


class Trajectory(BaseModel):
    trace_id: str
    task_id: str
    events: tuple[TraceEvent, ...]
    final_answer: str | None
    termination_reason: TerminationReason
    usage: Usage
    runtime_version: str
~~~

- [ ] Model AgentState as an immutable Pydantic model containing public MathTask, Budget, Usage, events, final_answer and termination_reason. Provide transition methods that return a new state and reject transitions after termination.

- [ ] Make monotonic_ms relative to loop start; keep wall-clock time only in trace envelope metadata so replay comparison is stable.

- [ ] Run:

~~~bash
uv run pytest tests/unit/agent/test_state.py tests/unit/agent/test_trace.py -q
uv run mypy src/adaptive_math/agent
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/agent/state.py src/adaptive_math/agent/trace.py tests/unit/agent/test_state.py tests/unit/agent/test_trace.py
git commit -m "feat: define deterministic agent state transitions"
~~~

---

### Task 3: Create the Tool Protocol and Registry

**Files:**

- Create: src/adaptive_math/tools/base.py
- Create: src/adaptive_math/tools/registry.py
- Modify: src/adaptive_math/tools/__init__.py
- Create: tests/unit/tools/test_registry.py

- [ ] Write failing tests for registration, duplicate names, unknown tools, argument validation, timeout conversion and output truncation.

- [ ] Implement these stable interfaces:

~~~python
class ToolErrorCode(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    OUTPUT_LIMIT = "output_limit"
    UNAVAILABLE = "unavailable"


class ToolResult(BaseModel):
    ok: bool
    output: str
    error_code: ToolErrorCode | None = None
    latency_ms: int = Field(ge=0)
    truncated: bool = False
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class Tool(Protocol):
    name: str
    description: str
    arguments_model: type[BaseModel]

    async def execute(self, arguments: BaseModel, context: ToolContext) -> ToolResult: ...
~~~

- [ ] ToolContext contains trace_id, task_id, remaining budget and cancellation event, never a reference answer or verifier.

- [ ] Registry.execute validates arguments using the selected tool's Pydantic model, catches only declared operational exceptions, truncates UTF-8 safely to context.remaining_observation_chars, and returns deterministic error observations.

- [ ] Serialize tool descriptions from arguments_model JSON schema so system prompts and validation cannot drift.

- [ ] Run:

~~~bash
uv run pytest tests/unit/tools/test_registry.py -q
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/tools tests/unit/tools/test_registry.py
git commit -m "feat: add typed tool registry"
~~~

---

### Task 4: Add an Allowlisted SymPy Tool

**Files:**

- Create: src/adaptive_math/tools/sympy_tool.py
- Create: tests/unit/tools/test_sympy_tool.py
- Create: tests/contract/test_sympy_tool_safety.py

- [ ] Define a discriminated arguments union for operations simplify, factor, expand, solve, diff, integrate and numeric. Every operation accepts only documented string expressions and explicit variable names; solve limits variables to four.

- [ ] Write expected-result tests for factor(x^2-1), solve(x^2-4, x), diff(sin(x), x), a definite integral, and 50-digit numeric evaluation.

- [ ] Write safety tests for unknown functions, dunder names, attribute access, Python collections, import syntax, huge powers, nesting over 64 and expression length over 8192.

- [ ] Run and confirm failures:

~~~bash
uv run pytest tests/unit/tools/test_sympy_tool.py tests/contract/test_sympy_tool_safety.py -q
~~~

- [ ] Reuse the verifier worker isolation primitive, but expose a separate tool worker pool with timeout_seconds=3, memory_mb=768 and CPU limit 3 seconds. Do not parse untrusted expressions in the API process.

- [ ] Return canonical text plus optional LaTeX in metadata. Sort unordered solve results by stable string form. Mark timeouts and rejected expressions with distinct error codes.

- [ ] Run:

~~~bash
uv run pytest tests/unit/tools/test_sympy_tool.py tests/contract/test_sympy_tool_safety.py -q
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/tools/sympy_tool.py tests/unit/tools/test_sympy_tool.py tests/contract/test_sympy_tool_safety.py
git commit -m "feat: add bounded symbolic math tool"
~~~

---

### Task 5: Integrate SandboxFusion for Python Execution

**Files:**

- Create: src/adaptive_math/tools/sandboxfusion.py
- Create: src/adaptive_math/tools/python_tool.py
- Create: docker/compose.sandbox.yml
- Create: tests/unit/tools/test_sandboxfusion_client.py
- Create: tests/unit/tools/test_python_tool.py
- Create: tests/contract/test_sandboxfusion.py
- Modify: pyproject.toml

- [ ] Write HTTP client tests using respx for POST /run_code with body containing code and language=python. Cover Success, compile error, nonzero return code, timeout, malformed response, connection failure and client deadline.

- [ ] Define SandboxRunRequest and response models matching the pinned SandboxFusion contract:

~~~python
class SandboxRunRequest(BaseModel):
    code: str = Field(min_length=1, max_length=20000)
    language: Literal["python"] = "python"


class SandboxRunResult(BaseModel):
    status: str
    execution_time: float
    return_code: int
    stdout: str
    stderr: str
~~~

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/tools/test_sandboxfusion_client.py tests/unit/tools/test_python_tool.py -q
~~~

- [ ] Implement an async httpx client with base URL from ADAPTIVE_MATH_SANDBOX_URL, default http://127.0.0.1:8080, connect timeout 1 second, request timeout 6 seconds and no retries for code execution. Add GET/health probing separately.

- [ ] Map remote status, run_result and HTTP failures to ToolResult without exposing service internals. Combine stdout and stderr with labeled sections; enforce max_observation_chars after UTF-8 decoding.

- [ ] Implement PythonTool arguments as only code: str. Do not accept files, network options, Docker flags, environment variables or host paths.

- [ ] Add a runtime optional dependency set containing httpx, respx for tests, transformers, torch and accelerate; lock it before the live contract test. Keep CUDA-specific wheels in the cloud image rather than the macOS lock resolution.

- [ ] Pin SandboxFusion to an immutable image digest in compose.sandbox.yml. Bind its published port to 127.0.0.1, set read-only host mounts to none, and document the exact upstream Git SHA as an image label.

- [ ] Start the pinned service and run live contract tests:

~~~bash
docker compose -f docker/compose.sandbox.yml up -d
ADAPTIVE_MATH_RUN_SANDBOX_CONTRACT=1 uv run pytest tests/contract/test_sandboxfusion.py -q
docker compose -f docker/compose.sandbox.yml stop
~~~

The live contract must execute arithmetic, capture stderr, enforce an infinite-loop timeout and prove that a new request succeeds afterward.

- [ ] Commit:

~~~bash
git add src/adaptive_math/tools/sandboxfusion.py src/adaptive_math/tools/python_tool.py docker/compose.sandbox.yml tests/unit/tools tests/contract/test_sandboxfusion.py pyproject.toml uv.lock
git commit -m "feat: execute python through sandboxfusion"
~~~

---

### Task 6: Implement the Budgeted Math Agent Environment

**Files:**

- Create: src/adaptive_math/agent/environment.py
- Create: src/adaptive_math/verifier/hidden.py
- Create: tests/unit/agent/test_environment.py
- Create: tests/contract/test_hidden_verifier_boundary.py

- [ ] Write transition tests for reset, valid tool call, invalid action, final, max steps, max tool calls, Python time exhaustion and action after termination.

- [ ] Define StepResult:

~~~python
class Observation(BaseModel):
    kind: Literal["tool_result", "action_error", "budget_warning"]
    content: str
    remaining_steps: int
    remaining_tool_calls: int
    remaining_python_seconds: float


class StepResult(BaseModel):
    state: AgentState
    observation: Observation | None
    terminated: bool
~~~

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/agent/test_environment.py tests/contract/test_hidden_verifier_boundary.py -q
~~~

- [ ] Implement ProductMathEnv(task, budget, registry), which cannot accept ReferenceAnswer, and OfflineMathEnv(labeled_task, budget, registry, hidden_verifier), which stores the reference in a private evaluation context not serialized into state or observations.

- [ ] On invalid action, increment steps and invalid_actions, emit a concise grammar reminder, and allow retry while budget remains. On tool call, reserve a tool-call slot before execution so cancellations and failures still consume budget.

- [ ] Before a Python call, cap the remote deadline to remaining_python_seconds. Add actual execution_time from SandboxFusion to usage; terminate after emitting the result if the budget is exhausted.

- [ ] On FINAL, store only the answer. Offline evaluation happens after termination and returns an EvaluationRecord separate from Trajectory; ProductMathEnv returns no reward.

- [ ] Make the boundary contract recursively inspect public state, observation, trace JSON, system prompt and model messages for forbidden reference values and keys.

- [ ] Run:

~~~bash
uv run pytest tests/unit/agent/test_environment.py tests/contract/test_hidden_verifier_boundary.py -q
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/agent/environment.py src/adaptive_math/verifier/hidden.py tests/unit/agent/test_environment.py tests/contract/test_hidden_verifier_boundary.py
git commit -m "feat: add budgeted math agent environment"
~~~

---

### Task 7: Add Model Client, Prompt Rendering and Agent Loop

**Files:**

- Create: src/adaptive_math/agent/model_client.py
- Create: src/adaptive_math/agent/prompts.py
- Create: src/adaptive_math/agent/loop.py
- Create: configs/agent/default.yaml
- Create: configs/agent/direct.yaml
- Create: configs/agent/tool_budget_2.yaml
- Create: tests/unit/agent/test_prompts.py
- Create: tests/integration/test_agent_loop.py

- [ ] Define ModelClient as an async protocol accepting immutable ChatMessage tuples and GenerationConfig, returning ModelTurn(text, prompt_tokens, generated_tokens, finish_reason, model_id).

- [ ] Write a scripted fake client that returns a predetermined action sequence. Test direct FINAL, tool then FINAL, invalid then recovery, model exception, cancellation and max-step termination.

- [ ] Write prompt snapshot tests asserting it contains the public problem, remaining budget, JSON schemas and exact one-action rule, while excluding hidden reference and reward vocabulary.

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/agent/test_prompts.py tests/integration/test_agent_loop.py -q
~~~

- [ ] Implement render_system_prompt from the live ToolRegistry JSON schemas and render_observation as a structured tool-role message. Keep prompt version in every trajectory.

- [ ] Implement AgentLoop.run:

~~~python
async def run(
    self,
    environment: ProductMathEnv | OfflineMathEnv,
    model: ModelClient,
    generation: GenerationConfig,
    cancellation: asyncio.Event | None = None,
) -> Trajectory: ...
~~~

- [ ] Ensure each loop iteration appends raw model output before parsing, accounts generated tokens, steps the environment once, and terminates on environment stop, model error or cancellation. Never silently request another generation after a terminal state.

- [ ] Add direct.yaml with max_tool_calls=0, tool_budget_2.yaml with max_tool_calls=2, and default.yaml with max_steps=6, max_tool_calls=4, max_python_seconds=12, max_observation_chars=8000.

- [ ] Run:

~~~bash
uv run pytest tests/unit/agent tests/integration/test_agent_loop.py -q
~~~

- [ ] Commit:

~~~bash
git add src/adaptive_math/agent configs/agent tests/unit/agent tests/integration/test_agent_loop.py
git commit -m "feat: orchestrate the math agent loop"
~~~

---

### Task 8: Add Trace Persistence and Deterministic Replay

**Files:**

- Create: src/adaptive_math/agent/replay.py
- Create: scripts/dev/replay_trace.py
- Create: tests/integration/test_trace_replay.py
- Create: tests/fixtures/golden_traces/direct_correct.json
- Create: tests/fixtures/golden_traces/tool_recovery.json

- [ ] Write golden replay tests that load a trace, reconstruct state from events and assert final_answer, usage and termination_reason exactly match the envelope.

- [ ] Write tamper tests for missing sequence, changed tool output, duplicate sequence and content hash mismatch.

- [ ] Run failing tests:

~~~bash
uv run pytest tests/integration/test_trace_replay.py -q
~~~

- [ ] Define a trace envelope containing schema_version, runtime_version, prompt_version, created_at, trajectory and SHA-256 over canonical trajectory JSON. Keep created_at outside the hashed deterministic event stream.

- [ ] Implement replay without invoking ModelClient, ToolRegistry or HiddenVerifier. It must reapply recorded state transitions and reject inconsistent usage counters.

- [ ] Implement CLI:

~~~bash
uv run python scripts/dev/replay_trace.py --trace tests/fixtures/golden_traces/tool_recovery.json --verify-hash --print-events
~~~

- [ ] Run golden tests twice and compare serialized replay output hashes.

- [ ] Commit:

~~~bash
git add src/adaptive_math/agent/replay.py scripts/dev/replay_trace.py tests/integration/test_trace_replay.py tests/fixtures/golden_traces
git commit -m "feat: persist and replay agent trajectories"
~~~

---

### Task 9: Add a Local Transformers Client and Runtime CLI

**Files:**

- Create: src/adaptive_math/agent/transformers_client.py
- Create: scripts/dev/run_agent.py
- Create: tests/unit/agent/test_transformers_client.py
- Create: tests/integration/test_runtime_cli.py
- Modify: README.md

- [ ] Write client tests with a fake tokenizer/model for chat-template rendering, stop token handling, token accounting, deterministic seed and max_new_tokens.

- [ ] Write CLI integration tests using ScriptedModelClient so CI needs no model download. Assert JSON stdout conforms to Trajectory and product mode rejects --reference-answer.

- [ ] Run failing tests:

~~~bash
uv run pytest tests/unit/agent/test_transformers_client.py tests/integration/test_runtime_cli.py -q
~~~

- [ ] Implement TransformersModelClient with device/dtype configured explicitly. Use tokenizer.apply_chat_template; never hand-build Qwen control tokens. Validate that the tokenizer has a chat template and record tokenizer revision.

- [ ] Implement run_agent.py with product and offline-eval subcommands. Product accepts problem, answer type, budget and model; offline-eval additionally loads a labeled task from a manifest path, not a raw answer command-line flag.

- [ ] Run a real Qwen3-1.7B local smoke only when weights are available:

~~~bash
uv run python scripts/dev/run_agent.py product --problem "Compute 17*19." --answer-type integer --model Qwen/Qwen3-1.7B --config configs/agent/default.yaml
~~~

- [ ] Update README with action examples, sandbox startup, CLI usage, trace location and explicit warning that product self-check does not know correctness.

- [ ] Run full runtime gate:

~~~bash
uv run ruff check src/adaptive_math/agent src/adaptive_math/tools scripts/dev tests
uv run mypy src/adaptive_math/agent src/adaptive_math/tools
uv run pytest tests/unit/agent tests/unit/tools tests/integration tests/contract/test_sandboxfusion.py tests/contract/test_hidden_verifier_boundary.py -q
~~~

- [ ] Search for unsafe local execution and inspect every hit:

~~~bash
rg -n "\bexec\(|\beval\(|subprocess|os\.system|shell=True" src/adaptive_math
~~~

The only acceptable subprocess-style process control is the bounded verifier/SymPy worker implementation; PythonTool must contain none.

- [ ] Commit:

~~~bash
git add src/adaptive_math/agent/transformers_client.py scripts/dev/run_agent.py tests/unit/agent/test_transformers_client.py tests/integration/test_runtime_cli.py README.md
git commit -m "feat: expose local adaptive math runtime"
~~~

## Runtime Exit Criteria

- [ ] Scripted unit/integration suite covers direct solve, successful tool use, recovery, malformed action, timeout, budget exhaustion, model failure and cancellation.
- [ ] SandboxFusion contract passes against the pinned image and no code path executes model Python on the host.
- [ ] ProductMathEnv has no constructor, field, trace or prompt path for hidden references.
- [ ] Trace replay detects tampering and reconstructs state without external calls.
- [ ] Qwen3-1.7B smoke produces a terminal, protocol-valid trajectory or a correctly classified budget termination.
