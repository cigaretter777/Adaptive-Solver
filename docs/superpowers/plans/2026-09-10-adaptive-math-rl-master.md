# AdaptiveMath-RL Implementation Plan — Master Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前 Adaptive-Solver 原型升级为可复现、可训练、可评测、可演示的数学 Agentic RL 项目，并在最多三天云端训练窗口内产出能够支持 Agent/Post-training 校招与初级岗位面试的实验资产。

**Architecture:** 项目分成四个可独立验收但按顺序集成的工作包：基础数据与确定性验证、单策略数学 Agent 运行时、SFT/GRPO 云端训练、冻结评测与产品化服务。训练环境复用产品运行时的动作协议和工具接口，但隐藏答案验证器仅存在于数据构建、训练和离线评测链路，绝不暴露给推理时 Agent。

**Tech Stack:** Python 3.12、Pydantic v2、SymPy、math-verify、httpx、pytest、Hypothesis、Qwen3-1.7B/Qwen3-4B、Transformers、PEFT、verl-agent/verl、SandboxFusion、FastAPI、Gradio、Docker、uv。

**Spec:** [AdaptiveMath-RL 产品与技术设计报告](../specs/2026-09-09-adaptive-math-rl-product-design.md)

## Global Constraints

- [ ] 所有训练样本、轨迹和指标都必须带不可变的 task_id、dataset、split、source_hash 与 pipeline_version。
- [ ] 训练、评测和产品运行时共享动作协议与工具协议，不共享隐藏答案或训练专用 verifier handle。
- [ ] Agent 可输出的顶层动作只有 TOOL_CALL 和 FINAL；可选 thinking 段属于同一 assistant turn 的模型推理，不增加环境动作。
- [ ] Python 代码只通过独立 SandboxFusion 服务执行；主进程内严禁 exec、eval、subprocess 执行模型生成代码。
- [ ] 奖励先做 R0/R1/R2/R3 消融；任何成本系数都视为实验假设，不能写成行业标准。
- [ ] 首期只承诺 AIME/MATH 风格、可确定验证的数值或符号答案；证明题、开放题和自然语言主观题不进入首期 RL 主实验。
- [ ] 主结果必须包含未训练基础模型、SFT、GRPO-R0、GRPO-R2 四条对照线，以及至少三次可比较评测运行。
- [ ] 云端单次正式训练预算不超过 72 小时；先通过本地、单 GPU、小规模多 GPU 三层门禁再启动正式训练。
- [ ] 不以测试集调参，不把自生成题直接并入冻结测试集，不报告未经复跑验证的最好单点。
- [ ] 所有成功声明必须由命令输出、指标 JSON 和可复现配置共同支持。

---

## Plan Suite and Execution Order

### Phase 0 — Repository Baseline

执行 [Foundation Plan](./2026-09-10-adaptive-math-rl-foundation.md) 的 Task 1，建立 Python 包、测试入口、依赖锁和 Git 基线。当前目录没有 Git 元数据，因此在开始实现时先执行该任务，再允许并行开发。

验收门：

- [ ] uv sync --all-extras 成功。
- [ ] uv run pytest -q 至少能收集并运行一个 smoke test。
- [ ] git status --short 只显示执行者明确保留的工作区改动。

### Phase 1 — Trusted Offline Core

完整执行 [Foundation Plan](./2026-09-10-adaptive-math-rl-foundation.md)。先把任务协议、答案抽取、确定性 verifier、reward 和数据切分做成可信离线核心。

验收门：

- [ ] verifier 的 golden、adversarial、property tests 全部通过。
- [ ] 等价答案、集合无序、数值容差、未定义表达式、超时、恶意字符串均有明确行为。
- [ ] reward 对同一轨迹是确定性的，R2 只在答案正确时施加归一化工具成本。
- [ ] 数据 split manifest 可复建，跨 split 的精确与近似重复率均为零。

### Phase 2 — Agent Runtime

执行 [Agent Runtime Plan](./2026-09-10-adaptive-math-rl-agent-runtime.md)，建立动作解析、工具层、沙箱、预算状态机、轨迹和模型客户端边界。

验收门：

- [ ] 给定同一模型动作序列，环境产生逐字节一致的结构化轨迹和终止原因。
- [ ] malformed action、未知工具、参数错误、超时、输出截断、预算耗尽都能进入可训练的观察，而不是让进程崩溃。
- [ ] SandboxFusion 不可用时明确失败，绝不回退到本机执行模型代码。
- [ ] hidden verifier 仅由训练/离线评测 environment factory 注入。

### Phase 3 — Training and Cloud

执行 [Training & Cloud Plan](./2026-09-10-adaptive-math-rl-training-cloud.md)，完成数据物化、SFT、verl-agent 适配、GRPO 奖励桥、云端镜像和 72 小时 runbook。

验收门：

- [ ] 200 条 SFT overfit smoke 能降低 loss，并生成符合协议的 TOOL_CALL 或 FINAL。
- [ ] 32 条 RL prompt 的端到端 smoke 能完成 rollout、验证、reward、advantage、反传和 checkpoint。
- [ ] token mask 单测证明只对策略生成 token 计算训练 loss，工具观察和环境注入 token 被屏蔽。
- [ ] 正式训练前成本估算、磁盘空间、断点续训、artifact 同步和故障恢复演练全部通过。

### Phase 4 — Evaluation and Productization

执行 [Evaluation & Product Plan](./2026-09-10-adaptive-math-rl-evaluation-product.md)，冻结基准、统计分析、服务 API、演示界面和求职交付物。

验收门：

- [ ] 同一冻结评测集上得到 Base、SFT、R0、R2 的 accuracy、pass@k、invalid rate、tool cost、latency、Pareto 数据。
- [ ] 报告提供置信区间、paired comparison、失败类型分布和至少十条可回放轨迹。
- [ ] 产品 API 不接受 ground_truth 字段；线上自检只能使用无标签检查器。
- [ ] demo 可展示预算、工具调用、观察、最终答案、耗时和停止原因。

---

## Target Repository Map

~~~text
Adaptive-Solver-main/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── data/
│   ├── eval/
│   ├── sft/
│   └── grpo/
├── data/
│   ├── manifests/
│   └── README.md
├── docker/
│   ├── Dockerfile.train
│   └── compose.sandbox.yml
├── docs/
│   ├── superpowers/specs/
│   ├── superpowers/plans/
│   ├── runbooks/
│   └── results/
├── scripts/
│   ├── data/
│   ├── eval/
│   ├── cloud/
│   └── train/
├── src/adaptive_math/
│   ├── core/
│   ├── data/
│   ├── verifier/
│   ├── reward/
│   ├── agent/
│   ├── tools/
│   ├── training/
│   ├── evaluation/
│   └── serving/
├── demo/
└── tests/
    ├── unit/
    ├── integration/
    ├── contract/
    └── fixtures/
~~~

Legacy files under src/config, src/graph, src/llm, src/rl and src/router remain readable during migration. New functionality只写入 adaptive_math 包；每迁移一个旧入口先加 characterization test，再删除或改成兼容转发层，避免新旧实现长期并存。

---

## Cross-Plan Interface Contract

下面的类型名是四份计划之间的稳定边界。实现期间可以增加字段，但删除、改名或改变语义必须先更新设计报告与所有消费方测试。

~~~python
class MathTask(BaseModel):
    task_id: str
    problem: str
    answer_type: AnswerType
    dataset: str
    split: str
    source_hash: str

class ReferenceAnswer(BaseModel):
    value: str
    answer_type: AnswerType
    acceptable_forms: tuple[str, ...] = ()

class LabeledMathTask(BaseModel):
    task: MathTask
    reference: ReferenceAnswer

    def public_view(self) -> MathTask: ...

class Budget(BaseModel):
    max_steps: int
    max_tool_calls: int
    max_python_seconds: float
    max_observation_chars: int

class ToolResult(BaseModel):
    ok: bool
    output: str
    error_code: str | None
    latency_ms: int
    truncated: bool

class Trajectory(BaseModel):
    trace_id: str
    task_id: str
    events: tuple[TraceEvent, ...]
    final_answer: str | None
    termination_reason: TerminationReason
    usage: Usage
    runtime_version: str

class VerifierResult(BaseModel):
    status: VerifierStatus
    reward: float
    normalized_prediction: str | None
    normalized_reference: str | None
    details: dict[str, JSONValue]
~~~

assistant turn 的 wire format 固定为一个可选 thinking 段，后接且只接一个动作：

~~~text
<think>先计算候选值，再用工具核对。</think>
<tool_call>{"name":"python","arguments":{"code":"print(2+2)"}}</tool_call>
~~~

也允许省略 thinking：

~~~text
<tool_call>{"name":"python","arguments":{"code":"print(2+2)"}}</tool_call>
~~~

或：

~~~text
<final>{"answer":"4"}</final>
~~~

任何一次 assistant turn 只能有一个完整顶层动作；除一个可选且完整的 think 段外，动作外非空文本、多个动作、未知字段或尾随内容都判为 INVALID_ACTION。thinking 和 action token 的训练 mask 为 1；system、user、tool observation 与环境错误提示的 mask 为 0。

---

## Evidence Package Required at Project Completion

- [ ] environment.json：Python/CUDA/PyTorch/Transformers/verl/verl-agent/SandboxFusion 的确切版本与 Git SHA。
- [ ] data_manifest.json：每个数据源的许可证、revision、原始哈希、过滤数量、split 哈希。
- [ ] train_config.yaml 与 resolved_config.yaml：声明配置和运行时最终配置。
- [ ] metrics.jsonl：逐 checkpoint 训练与评测指标。
- [ ] benchmark_predictions.parquet：逐题预测、验证状态、工具成本、停止原因，不公开受限测试标签。
- [ ] ablation_table.csv：Base/SFT/R0/R1/R2/R3 可用实验的成对比较。
- [ ] traces/：成功、失败、恢复、预算耗尽、工具异常等代表性轨迹。
- [ ] model_card.md：模型用途、限制、数据、训练、评测、许可证和已知风险。
- [ ] docs/results/final-report.md：结论、置信区间、负结果、成本和复现实验命令。
- [ ] 一段 3–5 分钟 demo 录屏脚本和一页架构图。

---

## Stop/Go Gates for the Three-Day Training Budget

### Gate A — Before Renting GPUs

- [ ] 本地 unit、contract、adversarial 测试全绿。
- [ ] 公开数据下载、清洗、split 和 materialization 可在无 GPU 环境完成。
- [ ] 训练镜像能构建，锁文件无未解析依赖。
- [ ] 评测脚本能对伪模型和一份小型 Hugging Face 模型运行。

### Gate B — First Two Cloud Hours

- [ ] GPU、NCCL、flash-attention、模型权重、SandboxFusion 健康检查通过。
- [ ] SFT 与 RL 各完成一次最小前向/反向/checkpoint/恢复测试。
- [ ] rollout 吞吐、显存峰值、平均输出长度与沙箱队列等待写入 smoke metrics。
- [ ] 若预计正式配置超过 72 小时，先缩短 max response、rollout n 或训练样本数，不盲目扩卡。

### Gate C — Before Full GRPO

- [ ] SFT checkpoint 在冻结 dev 集上协议有效率至少 98%。
- [ ] sampled reward 不出现全零、全一或 NaN；group 内有足够 reward variance。
- [ ] 100 条人工抽检未发现标签泄漏、verifier exploit 或工具输出进入 loss mask。
- [ ] R0 小跑有稳定 checkpoint，随后才开启 R2 成本奖励。

### Gate D — Release

- [ ] 最终结论至少由两次独立训练或三次独立评测支持。
- [ ] 如果没有显著超过 SFT，按负结果发布，并用失败分析说明瓶颈；不得选择性隐藏。
- [ ] README 中的数字均链接到配置、日志或结果文件。

---

## Definition of Done

项目完成不等于“训练脚本跑起来”，而是以下条件同时成立：

- [ ] 可从干净环境按 README 在本地复现数据验证、Agent demo 和小型评测。
- [ ] 可按云端 runbook 在中断后恢复训练，并把 checkpoints、配置和日志同步回对象存储。
- [ ] 至少一个 GRPO 模型在冻结集上相对 SFT 展现可解释收益，或形成可信的负结果与原因定位。
- [ ] verifier、reward、action mask、budget state machine 都有针对作弊和边界条件的测试。
- [ ] 产品演示与训练系统使用同一 Agent 核心，而不是两套演示逻辑。
- [ ] 求职材料能清楚回答：解决什么问题、为什么需要 Agent、RL 优化了什么、如何避免 reward hacking、成本是多少、哪些结论尚未证实。
