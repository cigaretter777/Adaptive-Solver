# Adaptive-Solver 2.0：AdaptiveMath-RL 技术与产品设计报告

版本：1.0  
日期：2026-09-09  
状态：设计基线，尚未实施、尚无真实训练结果  
目标岗位：Agent / LLM Post-training 校招与初级岗位

## 0. 执行摘要

Adaptive-Solver 2.0 的目标不是再实现一个由规则驱动的数学工作流，也不是用多个大模型模拟“规划、反思和执行”。项目要构建一个真正可训练的数学 Agent：

> 通过监督微调和 Agentic GRPO，使一个小型开源模型学会根据数学题目的类型、难度和预算，自主选择直接推理、Python 计算、SymPy 符号求解、错误恢复和动态停止策略。

首版聚焦 AIME、MATH 一类具有确定数值或符号答案的竞赛数学问题。正式模型选择 Qwen3-4B，Qwen3-1.7B 用于低成本开发和流水线验证。训练底座采用 verl-agent，并固定其依赖的 verl 版本；项目自己实现数学环境、工具、安全沙箱、答案验证、奖励、数据治理、评测和产品服务。

首版不训练独立过程奖励模型，不做 Lean 形式化证明、不做 Web Search、不做多 Agent 辩论、不做 MCTS，也不自研 GRPO。这样可以把有限算力集中在三个可验证问题上：

1. 真实 GRPO 是否比 Base 和 SFT 提升冻结测试集上的数学正确率；
2. Policy 是否学会按题目难度和题型改变工具使用方式；
3. Adaptive Policy 是否在正确率—成本 Pareto 前沿上优于固定工具策略。

本报告既是技术设计，也是产品设计。报告中的“预期效果”均为待验证目标，不代表项目已经取得这些结果。正式对外发布只能使用真实日志、真实 checkpoint 和冻结评测结果。

## 1. 项目为什么存在

### 1.1 用户问题

普通 To-C 聊天模型在复杂数学问题上容易出现：

- 长推理链中一个局部错误导致最终答案错误；
- 能写出看似合理的解释，但无法可靠计算；
- 不知道什么时候应该调用代码或符号工具；
- 工具报错后直接失败，或者重复无效调用；
- 得到候选答案后缺少代回、交叉计算和边界检查；
- 简单题过度推理，复杂题又过早结束；
- 输出答案无法被稳定解析和验证。

传统固定工作流可以强制模型按“规划—工具—检查”执行，但行为来自代码规则而不是模型学习，难以根据题目动态调整，也无法通过训练持续改进。

### 1.2 产品目标

面向用户，产品应提供：

- 输入一道可文本表达的竞赛数学题；
- 自动选择直接推理、Python 或 SymPy；
- 工具失败后能够修改方案并继续；
- 输出结构化解答、最终答案和可展示的验证证据；
- 提供经济、平衡、深度三种预算模式；
- 对无法可靠完成的问题明确表达限制，而不是伪造确定性。

线上产品没有标准答案。产品中的“已验证”只表示候选答案通过了代入、符号化简或独立计算等自检，不能表示拥有隐藏答案的绝对正确性。

### 1.3 研究目标

核心研究命题是：

> 在确定性数学结果奖励下，预算感知的 Agentic GRPO 能否让 4B 模型学到优于 Never-Tool、Always-Tool 和 Fixed-K 的自适应工具策略？

该命题具有明确的输入、动作、环境、奖励、基线和指标，可以被实验支持或否定。

## 2. 当前项目状态与升级必要性

当前仓库适合作为概念原型，不具备真实 Agentic RL 训练能力：

- 主入口中的 Turbo 和 Max 路径仍是占位响应，而非真实模型生成，见 [src/main.py](../../../src/main.py#L142)；
- 现有 GRPO Trainer 直接把总 reward 当作 advantage，见 [src/rl/grpo_trainer.py](../../../src/rl/grpo_trainer.py#L126)；
- policy loss、value loss 和 KL 是随机数，见 [src/rl/grpo_trainer.py](../../../src/rl/grpo_trainer.py#L151)；
- 奖励包含固定“最优工具次数”和关键词匹配，无法验证数学正确性，见 [src/rl/reward_function.py](../../../src/rl/reward_function.py#L125)；
- README 声明存在 requirements.txt，但当前工作目录中没有该文件；
- 当前目录不是 Git 仓库，无法记录可复现 commit。

因此 V2 不继续修补模拟 Trainer，而采用以下迁移原则：

1. 现有版本标记为 V1 概念原型；
2. 新建清晰的 AdaptiveMath-RL 包结构；
3. 使用 verl-agent/verl 负责真实策略更新；
4. 保留可复用的状态、路由和轨迹概念，但重写接口；
5. 删除对关键词奖励、随机 loss 和模拟成功率的依赖；
6. 所有正式指标只来自可重放的真实运行。

## 3. 设计依据与项目自主假设

### 3.1 有公开依据的部分

- 数学结果奖励与 GRPO：DeepSeekMath 在数学推理背景下提出 GRPO。[DeepSeekMath](https://arxiv.org/abs/2402.03300)
- 规则化准确率和格式奖励：DeepSeek-R1 在推理任务中采用规则奖励。[DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- 动态采样：DAPO 过滤全对或全错、缺少有效梯度的 prompt group。[DAPO](https://dapo-sia.github.io/)
- 工具融合数学推理：ToRL 研究模型通过 RL 自主使用代码工具并从错误中恢复。[ToRL](https://arxiv.org/abs/2503.23383)
- 工具效率：OTC-PO 研究正确率与工具调用成本的联合优化。[OTC](https://arxiv.org/abs/2504.14870)
- 多轮 Agent RL 基础设施：verl 支持异步 rollout、多轮对话、工具调用和 AgentLoop。[verl Agentic RL](https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst)
- 长程环境与分组采样：verl-agent 提供多轮环境、GRPO、GiGPO 和 LoRA 支持。[verl-agent](https://github.com/langfengq/verl-agent)
- 主流 Agent 运行模式：模型在计划、行动、观察和调整循环中自主使用工具。[Anthropic](https://www.anthropic.com/research/trustworthy-agents)

### 3.2 项目自主设计

以下内容不是某家商业产品公开的固定实现，而是本项目需要实验验证的设计：

- 使用预算条件驱动工具选择和停止；
- 对正确轨迹应用工具成本，对错误轨迹不以节省工具为奖励；
- Python 与 SymPy 双工具的路由策略；
- 工具失败恢复、交叉验证和停止行为的测量方式；
- 面向数学等价性的类型化 Verifier；
- 训练内核和 To-C 产品外壳共享同一 Agent 协议。

### 3.3 不可声称的内容

- 不声称复现字节、阿里、腾讯的内部训练系统；
- 不声称商业 Agent 使用了本项目的具体 reward；
- 不声称接入 verl 就等于工业级训练；
- 不在实验前声称准确率必然提升；
- 不把模型说“让我检查”当作真正的自我纠错证据。

## 4. 范围与非目标

### 4.1 V1 范围

- 英文为主的 AIME/MATH 风格竞赛数学；
- 整数、有理数、数值、表达式、方程、集合、元组、区间和矩阵答案；
- 单个 Qwen3 Policy；
- Python 与 SymPy；
- 确定性结果 Verifier；
- Agent SFT；
- 同步 GRPO；
- 难度课程与动态采样；
- 本地开发、云端训练；
- 可复现评测；
- To-C 演示产品。

### 4.2 V1 非目标

- 独立 PRM；
- 开放式自然语言证明评分；
- Lean 形式化证明；
- 多 Agent 投票或辩论；
- 搜索引擎；
- MCTS；
- 异步 rollout 性能优化；
- 多模型路由；
- 7B/8B 正式全量训练；
- 自研 PPO/GRPO；
- 宣称达到前沿闭源模型水平。

## 5. 用户、场景与产品体验

### 5.1 目标用户

- 希望获得复杂竞赛数学解答的学习者；
- 研究 Tool-Integrated Reasoning 的开发者；
- 需要可重放 Agent 轨迹的 Post-training 研究者；
- 招聘场景中需要验证候选人工程与实验能力的面试官。

### 5.2 核心用户流程

~~~text
用户提交题目
  → 选择预算模式
  → 系统检查输入与支持范围
  → Agent 开始推理
  → 必要时调用 Python/SymPy
  → 根据 observation 修正
  → 生成最终结构化解答
  → 运行不含标准答案的自检
  → 返回答案、解题过程、工具证据和限制说明
~~~

### 5.3 三种预算模式

- 经济：较小 token 和工具预算，鼓励直接求解；
- 平衡：默认配置，允许有限工具和一次恢复；
- 深度：更长上下文和更多工具机会。

预算模式进入 Policy 上下文，模型学习条件化行为，而不是简单地由外层代码强制固定调用次数。

### 5.4 输出设计

产品不直接展示训练用隐藏 reward，也不把内部 raw scratchpad 原样暴露。面向用户输出：

- 问题理解；
- 可读的关键解题步骤；
- 最终答案；
- 使用过的工具及摘要；
- 已执行的自检；
- 适用条件与不确定性说明。

## 6. 总体架构

~~~text
Web / CLI
   ↓
API Gateway：鉴权、限流、取消、输入校验
   ↓
Math Agent Service
   ├── Session State
   ├── Context Builder
   ├── Action Parser
   ├── Agent Loop
   └── Trace Recorder
   ↓
Policy Server：Qwen3-4B
   ↕
Tool Runtime
   ├── Python Sandbox
   └── SymPy Service
   ↓
Serving Self-Check

训练专用旁路：
MathAgentEnv
   → Hidden Ground-truth Verifier
   → Reward Engine
   → verl-agent / verl
   → Policy Update
~~~

系统分成三个可独立理解的包：

- adaptive_math_core：协议、状态、工具、Verifier、Reward、Trace；
- adaptive_math_training：数据、SFT、GRPO、云端配置、评测；
- adaptive_math_serving：API、Session、流式输出、限流和产品指标。

训练和产品共享 core，不共享隐藏 ground truth。

### 6.1 建议代码结构

~~~text
adaptive-solver/
├── src/adaptive_math/
│   ├── core/
│   │   ├── protocol.py
│   │   ├── state.py
│   │   ├── agent_loop.py
│   │   ├── context.py
│   │   └── trace.py
│   ├── tools/
│   │   ├── base.py
│   │   ├── python_tool.py
│   │   ├── sympy_tool.py
│   │   └── sandbox_client.py
│   ├── verifier/
│   │   ├── extractor.py
│   │   ├── normalize.py
│   │   ├── typed_verifiers.py
│   │   └── adversarial_cases.py
│   ├── rewards/
│   │   ├── outcome.py
│   │   ├── invalid.py
│   │   ├── cost.py
│   │   └── registry.py
│   ├── environments/
│   │   └── math_agent_env.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── preprocess.py
│   │   ├── deduplicate.py
│   │   └── manifest.py
│   ├── training/
│   │   ├── verl_adapter.py
│   │   ├── sft_entry.py
│   │   └── grpo_entry.py
│   ├── evaluation/
│   │   ├── runner.py
│   │   ├── baselines.py
│   │   ├── metrics.py
│   │   └── report.py
│   └── serving/
│       ├── api.py
│       ├── schemas.py
│       └── session.py
├── configs/
│   ├── local/
│   └── cloud/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   └── reward_hacking/
├── data_manifests/
├── scripts/
├── docker/
├── reports/
└── pyproject.toml
~~~

上游 verl-agent/verl 作为固定依赖存在，不复制进 adaptive_math 包。这样审阅者可以清楚看到哪些代码来自框架，哪些是项目贡献。

### 6.2 关键选择及原因

| 选择 | 为什么这样设计 |
|---|---|
| 单 Policy Agent | 让工具选择、恢复和停止都能归因于同一策略 |
| Tool/Final 两类动作 | 减少格式空间和无意义的显式反思动作 |
| Python + SymPy | 同时覆盖数值/枚举与符号数学，仍控制工具规模 |
| Hidden Verifier | 训练可用标准答案，线上避免答案泄漏 |
| Rule reward 优先 | 数学结果可验证，精度高且不需要第二个 RM |
| SFT 后再 GRPO | 降低格式错误和全零 reward 风险 |
| 同步 rollout | V1 优先训练正确性和可复现，而非吞吐宣传 |
| LoRA 4B | 在模型能力、成本和多实验之间折中 |
| 1.7B 开发模型 | 用低成本发现接口和配置错误 |
| 训练/Serving 共用 core | 避免训练和产品行为不一致 |
| 本地测试、云端训练 | 把昂贵 GPU 时间留给必须使用 GPU 的步骤 |
| PRM 后置 | 三天预算先证明可靠 RLVR，再增加过程模型 |

## 7. Agent 设计

### 7.1 为什么是单 Agent

V1 只有一个可训练 Policy。Planner、Router、Reflector 不再是独立大模型：

- 规划成为 Policy 的推理行为；
- 路由成为 Policy 对工具动作的选择；
- 反思成为 Policy 对工具 observation 的后续响应；
- Critic 在 V1 中由确定性结果 Verifier 替代。

这使全部行为都能归因于同一个策略更新，避免多 Agent 带来的信用分配、延迟和部署复杂度。

### 7.2 动作协议

环境只接受两类顶层动作：TOOL_CALL 和 FINAL。为了兼容 Qwen3 thinking 并让 action span 可精确定位，一个 assistant turn 使用“可选 think 段 + 唯一动作段”的固定 wire format：

~~~text
<think>先计算候选值，再用 Python 核对。</think>
<tool_call>{"name":"python","arguments":{"code":"print(6 * 7)"}}</tool_call>
~~~

~~~text
<final>{"answer":"\\boxed{42}"}</final>
~~~

只允许一个可选且完整的 think 段，后接且只接一个动作；动作外自由文本、多个动作、未知字段或尾随内容都记为 INVALID_ACTION。SymPy 使用结构化 operation 和 expression，避免任意 Python eval。thinking 是模型生成内容，但 REPLAN、REFLECT 不定义为环境动作。

### 7.3 Agent 状态

每个 episode 保存：

- problem_id、问题文本、领域与难度；
- 隐藏标准答案；
- budget 与剩余预算；
- 消息历史；
- 模型 action token；
- 工具调用、结果和错误；
- invalid action；
- step 和 termination reason；
- policy version；
- reward breakdown。

隐藏答案只有终局 Verifier 可见。

### 7.4 上下文

V1 使用 append-only transcript，不做摘要：

- system、problem 和 tool observation 的 loss mask 为 0；
- 模型生成的 reasoning 和 action token 的 loss mask 为 1；
- 超长轨迹以 context_limit 终止，不静默截断；
- 训练、评测、Serving 使用同一个 chat template；
- 保存生成时的 token ID，禁止依赖事后重分词。

首轮 profiling 采用 8K 轨迹上限、最多 4 次工具调用、最多 6 个决策轮次；这些是容量上限，不是要求每条轨迹用满。

## 8. 工具与安全沙箱

### 8.1 Python Tool

用于枚举、数值计算、组合搜索、模拟和高精度运算。返回 stdout、stderr、exit code、执行时间和截断状态。

正式训练中，模型生成代码必须在隔离 worker 执行：

- 禁止网络；
- 只读基础文件系统；
- 独立临时目录；
- 禁止读取训练进程环境变量；
- 禁止子进程和外部命令；
- CPU、内存、输出和 wall-time 限制；
- 每个 episode 独立状态；
- 工具异常不允许杀死 rollout worker。

### 8.2 SymPy Tool

提供 allowlist 操作：

- simplify、factor、expand；
- solve、solveset；
- diff、integrate、limit；
- substitute、numeric_eval；
- polynomial roots；
- matrix operations。

禁止任意函数导入和不受控表达式执行。

### 8.3 训练 Verifier 与产品自检的隔离

- 产品工具只能计算模型提交的候选内容；
- 产品工具不能访问标准答案；
- Hidden Verifier 只在训练和离线评测终局运行；
- Hidden Verifier 的结果不返回当前 Policy，防止答案试探。

## 9. Verifier 设计

### 9.1 处理链

~~~text
Raw Output
  → Final Answer Extraction
  → Normalization
  → Answer Type Dispatch
  → Deterministic Verification
  → Structured VerifierResult
~~~

### 9.2 类型化验证

- integer：精确整数；
- rational：分数约分与十进制等价；
- real_number：绝对误差与相对误差；
- symbolic_expression：带超时的符号差化简；
- equation：解集与定义域检查；
- ordered_tuple：顺序敏感；
- unordered_set：规范化后顺序无关；
- interval：端点与开闭性；
- matrix：维度和元素；
- multiple_choice：严格选项。

随机代入只能作为辅助信号，不能单独证明恒等。涉及定义域、根式和增根时必须 fail-closed。

### 9.3 VerifierResult

结果分为：

- correct；
- incorrect；
- invalid；
- unverifiable；
- infrastructure_error。

基础设施错误和无法可靠验证的样本不进入梯度；模型格式错误可以形成小额非法行为惩罚。

### 9.4 Verifier 门禁

- 黄金测试集总体通过率至少 99%；
- 已知对抗集不允许 false positive；
- 所有解析与符号运算有超时；
- 版本变化执行完整回归；
- 随机人工抽检真实 rollout；
- 与 Math-Verify 或另一套实现交叉检查分歧样本。

## 10. Reward 设计

### 10.1 原则

- 正确性是主目标；
- 工具、反思和长推理只是手段；
- 效率只在正确答案之间比较；
- 不奖励表面数学语言；
- 不用 LLM Judge 作为 V1 主奖励；
- 所有新 reward 必须从简单基线逐项消融。

### 10.2 奖励实验

R0，纯结果奖励：

~~~text
R0 = 1[final answer correct]
~~~

R1，加入非法行为：

~~~text
R1 = R0 - μ × invalid_count
~~~

R2，加入正确轨迹的工具成本：

~~~text
R2 = 1[correct] × (1 - λ × normalized_tool_cost)
     - μ × invalid_count
~~~

R3，可选 token 成本：

~~~text
R3 = R2 - 1[correct] × η × normalized_action_tokens
~~~

约束 0 ≤ λ < 1，确保正确答案的奖励原则上高于错误答案。λ、μ、η 是待 pilot 和消融确定的参数，不写成行业固定值。

V1 不把延迟直接加入训练 reward，因为云端瞬时负载会引入噪声；延迟作为离线产品指标。

### 10.3 不直接奖励

- 出现“反思”字样；
- 调用某个指定工具；
- 写得更长；
- 故意先失败再修复；
- 多次交叉验证；
- 工具 stdout 的某些关键词。

工具价值最终通过正确性与成本体现。

### 10.4 Reward 可观测性

逐轨迹保存 answer、invalid、tool cost、token cost、总 reward、Verifier 方法和失败原因。训练面板展示：

- 正确率；
- reward mean/std；
- group reward variance；
- 全零组与全一组比例；
- invalid action rate；
- unverifiable 和 verifier timeout；
- 工具调用分布；
- action token 长度。

## 11. 数据设计

### 11.1 数据职责分离

- SFT：问题加完整 Agent 轨迹；
- RL：问题加隐藏答案，过程由当前 Policy 在线生成；
- Eval：完全冻结，训练与调参不可见。

### 11.2 SFT 数据

主要来源：

- [NuminaMath-TIR](https://huggingface.co/collections/AI-MO/numinamath)；
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)；
- [MATH](https://github.com/hendrycks/math) 训练集；
- 项目生成并执行验证的 Python/SymPy/恢复轨迹。

首版目标约 20K 条高质量轨迹，初始构成为：

- 40% 直接推理；
- 35% Python；
- 15% SymPy；
- 10% 工具失败恢复与答案复核。

数量与比例在 token 统计和行为覆盖检查后调整。不能为了凑数量保留重复、超长或未验证轨迹。

### 11.3 RL 数据

主要来源：

- [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)；
- 去重后的 MATH train；
- NuminaMath-1.5 中可被确定性验证的题目。

首轮使用约 8K 至 12K 道题，按领域、难度和答案类型分层。Policy 看不到参考解答。

### 11.4 评测数据

- 可比性：MATH-500、AIME 2024/2025、AMC；
- 高难度：Omni-MATH 的规则可验证部分；
- 严格保留：AIME 2026 和人工审核的新题/参数化题。

公开 benchmark 与严格保留集分开报告，披露潜在预训练污染。

### 11.5 数据治理

~~~text
来源登记
  → 格式标准化
  → 答案提取
  → Verifier 可执行性
  → 精确去重
  → 近似文本去重
  → 数学模板/表达式重复检查
  → 领域与难度标注
  → 固定 split
  → 内容 hash 与 manifest
~~~

每条数据记录 source、license、revision、usage、transform_version、verifier_version 和 content_hash。

## 12. 模型与训练底座

### 12.1 模型

- 开发模型：Qwen3-1.7B；
- 正式模型：Qwen3-4B；
- 微调方式：LoRA，bf16；
- 7B/8B 只作为 V2 或预算充足后的迁移验证。

Qwen3-4B 支持 thinking/non-thinking，并在官方模型说明中强调数学和 Agent 工具能力。[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

### 12.2 底座

- verl-agent：Agent 环境、分组环境和算法适配；
- verl：actor、reference、logprob、FSDP、rollout 和 checkpoint；
- vLLM 或 SGLang：真实推理；
- Ray：worker 调度；
- MLflow/W&B：训练与轨迹观测。

固定上游 commit，不直接复制上游核心。项目通过自定义环境、Reward Manager 和 Trainer Adapter 接入。

## 13. SFT 设计

### 13.1 目的

SFT 不负责把所有题教会，而是提供 RL 冷启动：

- 稳定输出 tool/final 协议；
- 能解释工具 observation；
- 学会基本错误恢复；
- 保留不调用工具的能力；
- 避免纯 RL 初期大量格式错误和全零组。

### 13.2 初始配置范围

- Qwen3-4B + LoRA；
- 1 epoch 起步；
- max sequence length 初始 8K；
- 学习率在 5e-6 至 2e-5 小范围 pilot；
- rank 32 或 64；
- 按 token 数而不是样本数控制 batch；
- 超长样本预先过滤，不在 batch 中静默截断。

配置最终值必须由小规模 loss、格式成功率和 held-out 行为共同决定。

### 13.3 SFT 门禁

进入 RL 前必须满足：

- tool/final 解析成功率不低于 98%；
- 工具 observation 能进入下一轮；
- Base 数学能力没有灾难性下降；
- Never-Tool、Python、SymPy 三类行为都存在；
- 本地黄金轨迹和云端端到端测试通过。

## 14. GRPO 与课程学习

### 14.1 Rollout

每道题由同一 policy version 采样 4 至 8 条轨迹。正式候选值 G=8，若吞吐不足再退到 G=4。每条轨迹执行真实工具，保存原始 token IDs 和 action mask。

### 14.2 同步优先

V1 使用同步 rollout，优先保证 on-policy freshness、可调试性和复现。异步只在 V2 中通过吞吐和 staleness 对照证明价值。

### 14.3 难度课程

~~~text
基础算术/GSM8K 只做冒烟
  → MATH Level 1–2
  → MATH Level 3–4
  → MATH Level 5
  → AIME/Omni-MATH 主要用于评测
~~~

维护每个难度桶的历史成功率。优先采样组内有正确也有错误的有效区间，同时保留少量过易和过难题监控边界。

### 14.4 训练稳定性

必须监控：

- 全零与全一组；
- advantage mean/std；
- KL；
- entropy；
- clip fraction；
- gradient norm；
- response length；
- 工具调用次数；
- checkpoint 间行为漂移；
- reward hacking 案例。

如果训练 reward 上升而冻结正确率不升，立即暂停并审计 Verifier、数据分布和成本项。

### 14.5 初始超参数

以 verl 官方 Qwen3-4B GRPO 示例为起点，而不是凭空设置。初始 actor learning rate 以 1e-6 附近做短程 pilot；是否启用 KL、具体 clip、batch 和长度参数由显存与稳定性试验决定。[verl Qwen3-4B GRPO 示例](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_4b_fsdp.sh)

所有偏离上游 recipe 的配置都要在实验记录中说明原因。

## 15. 本地开发与云端训练

### 15.1 本地职责

- 数据与 manifest；
- Parser、Tool、Verifier、Reward 单测；
- Sandbox 协议；
- 环境 reset/step；
- 黄金轨迹回放；
- 配置校验；
- reward hacking 测试；
- 报告与可视化。

本地 Stub Model 只允许做协议测试，不能生成正式指标。

### 15.2 云端职责

- 1.7B GPU 冒烟；
- 4B SFT；
- 真实 vLLM/SGLang rollout；
- GRPO；
- checkpoint/resume；
- 批量评测；
- 真实轨迹导出。

### 15.3 可复现元数据

每次运行记录：

- git commit；
- model/dataset revision；
- dependency lock hash；
- config hash；
- Python/PyTorch/CUDA；
- GPU 型号和数量；
- seed；
- policy version；
- start/end time。

密钥只从环境变量或 Secret Manager 注入，不能进入 Git、日志或轨迹。

### 15.4 云端门禁

1. 本地全部测试；
2. 云端 CUDA/NCCL 和依赖检查；
3. 1.7B、8 个样本、2 个更新 step；
4. 1.7B 短程趋势测试；
5. 4B SFT 小批量；
6. 4B GRPO 小批量；
7. 正式训练。

任一门禁失败都不能继续消耗正式算力。

## 16. 三天云端执行预算

三天是正式训练窗口上限，不是开发周期。镜像、数据和测试应提前准备。

推荐资源策略：

- 先用廉价 GPU 完成 1.7B 集成；
- 对 4B LoRA 先尝试 2×H100 80GB；
- 若长轨迹吞吐不足，扩展到 4×H100；
- 只有并行运行主实验种子时才考虑更多 GPU；
- 不用 H100 下载数据、编译依赖或画图。

建议时间盒：

- 0–3 小时：环境校验、1.7B 与 4B smoke；
- 3–12 小时：4B SFT 与 SFT 评测；
- 12–42 小时：GRPO R0 主训练；
- 42–62 小时：R2 核心 Adaptive 训练或主种子复现；
- 62–70 小时：冻结评测和必要消融；
- 70–72 小时：上传 checkpoint、日志、轨迹和环境报告。

这是调度预算，不是性能承诺。正式安排根据 pilot 得到的 tokens/s、trajectories/hour、峰值显存和平均轨迹长度更新。

三随机种子策略：

- 1.7B 用三个种子完成全部奖励消融；
- 4B 对核心 R0 与 R2 尽量完成三个种子；
- 若三天内不足，至少完成一个 4B 主运行，并明确把多种子验证标为未完成，不能伪装成稳定结论；
- 可在配置稳定后并行租用多组 GPU，以增加 GPU-hours 换取有限 wall time。

## 17. 评测设计

### 17.1 基线

- Base-Direct：原始 Qwen3-4B 直接回答；
- Base-Agent：原始模型允许使用工具；
- Always-Python：固定使用 Python；
- Always-SymPy：固定使用 SymPy；
- Fixed-K：固定工具预算；
- SFT-Agent；
- GRPO-R0；
- Adaptive GRPO-R2；
- GRPO-R1/R3 作为消融。

所有方法使用相同题目、最大 token、温度、采样数和工具实现。

### 17.2 主指标

- Pass@1 / exact accuracy；
- 按领域与难度的准确率；
- 工具调用次数；
- 工具有效调用率；
- invalid action rate；
- 工具失败恢复率；
- 平均 action token；
- 平均决策轮次；
- p50/p95 latency；
- 每题推理成本；
- success–cost Pareto frontier。

### 17.3 Adaptive 行为指标

- 题目难度与工具调用次数的相关性；
- Python/SymPy 与题型匹配度；
- 简单题直接回答率；
- 复杂题工具调用率；
- 工具失败后的有效重试率；
- 得到足够证据后的停止率；
- 不同预算条件下的行为单调性。

“出现反思文本”只作为定性案例，不作为能力指标。

### 17.4 统计方法

- 主要 RL 结论至少三个随机种子；
- 报告均值、标准差和置信区间；
- 对同题比较使用配对 bootstrap；
- 预先固定主指标、checkpoint 选择规则和停止条件；
- 不只挑选最好 seed；
- 失败结论也进入报告。

### 17.5 污染与泄漏

- test 不参与 SFT、RL、reward 调参和 checkpoint 选择；
- 精确与近似去重；
- 数学模板和数字替换检测；
- 公开 benchmark 与严格保留集分开；
- 答案不进入 Policy prompt；
- 工具服务不能读取 ground truth。

## 18. 产品服务设计

### 18.1 服务组件

- FastAPI/HTTP Gateway；
- Agent Service；
- vLLM/SGLang Policy Server；
- Python Sandbox Pool；
- SymPy Worker；
- Session Store；
- Trace/Metric Store；
- 简单 Web UI。

### 18.2 API

核心请求字段：

- problem；
- budget_mode；
- language；
- max_time；
- stream；
- request_id。

核心响应字段：

- final_answer；
- structured_solution；
- tools_used；
- self_checks；
- termination_reason；
- latency；
- model_version；
- trace_id；
- limitation。

### 18.3 产品可靠性

- 每个请求有 token、tool、step 和 wall-time 上限；
- 支持客户端取消；
- Tool Worker 熔断与隔离；
- Policy Server 超时后返回明确错误；
- 不因单条恶意代码影响其他请求；
- 日志脱敏；
- 相同 trace_id 可重放；
- 模型版本可灰度和回滚。

### 18.4 主流产品设计对应

本项目的单模型自主循环、结构化工具、环境反馈、状态、Guardrail 和 Trace 与主流 Agent SDK 的基础抽象一致。OpenAI Agents SDK 将 Tool、Guardrail、Session 和 Trace 作为核心组件；LangGraph 将 Agent 描述为动态决定工具与过程的循环。[OpenAI Tools](https://openai.github.io/openai-agents-python/tools/)、[OpenAI Tracing](https://openai.github.io/openai-agents-python/tracing/)、[LangGraph](https://langchain-ai.github.io/langgraph/agents/tools/)

多 Agent 是后续可选编排方式，不是 V1 完整性的必要条件。

## 19. 可观测性与实验产物

### 19.1 轨迹

每条轨迹至少包含：

- episode/problem ID；
- git/model/dataset/config revision；
- policy version 和 seed；
- 原始生成 token IDs；
- loss mask；
- 工具请求和响应；
- timeout/invalid；
- final answer；
- verifier result；
- reward breakdown；
- latency 与资源用量。

### 19.2 面板

- 训练正确性面板；
- reward 与 advantage 面板；
- 工具行为面板；
- 长度与吞吐面板；
- Verifier 健康面板；
- Eval 按难度和领域面板；
- Reward hacking 告警。

### 19.3 求职交付

- 完整 GitHub 仓库；
- 一条命令的本地测试；
- 云端训练说明；
- 锁定依赖与 Docker 镜像；
- 数据 manifest；
- LoRA checkpoint；
- 原始与汇总评测；
- W&B/MLflow 日志；
- 可浏览典型轨迹；
- 中英文 README；
- 技术报告；
- 5–10 分钟演示视频；
- 对 verl-agent/verl 的实质上游 PR 尝试。

## 20. 测试策略

### 20.1 本地单元测试

- ActionParser 合法与非法输入；
- 答案提取；
- 每种答案类型；
- SymPy timeout；
- Python 沙箱权限；
- Reward 边界；
- Env 状态转移；
- budget 终止；
- Trace 序列化；
- 配置 hash。

### 20.2 集成测试

- 固定模型输出驱动完整 tool loop；
- 工具错误后恢复；
- token mask 与消息边界；
- 同一轨迹训练/评测重放一致；
- checkpoint 保存与恢复；
- Policy version freshness；
- Tool Worker 崩溃隔离。

### 20.3 Reward hacking

- 空答案、多个 boxed、矛盾答案；
- NaN/Inf/复数；
- 浮点误差；
- 定义域和增根；
- 随机代入假阳性；
- 超长或超时表达式；
- 工具 observation 注入 final 标签；
- 文件、网络、环境变量和子进程访问；
- 重复相同工具调用；
- 输出长度与解析器边界。

## 21. 实施阶段与门禁

### Phase 0：仓库和基线

- 初始化或恢复 Git；
- 保存 V1 基线；
- 建立 pyproject、lock、CI；
- 固定 verl-agent/verl commit；
- 记录 Base-Direct 和 Base-Agent 结果。

完成条件：所有基线可重复运行且无模拟指标。

### Phase 1：数据与 Verifier

- 建立 schema、manifest 和 split；
- 实现答案类型与 Verifier；
- 建立黄金集和攻击集；
- 完成数据泄漏检查。

完成条件：Verifier 门禁通过。

### Phase 2：Math Agent Runtime

- 实现 State、Parser、AgentLoop；
- Python/SymPy Tool；
- Sandbox Worker；
- TraceRecorder；
- 本地固定轨迹集成测试。

完成条件：不依赖真实大模型也能重放完整协议。

### Phase 3：SFT

- 构造约 20K Agent 轨迹；
- 1.7B 调试；
- 4B LoRA SFT；
- 对比 Base 与 SFT。

完成条件：协议成功率、能力保持和工具行为门禁通过。

### Phase 4：真实 GRPO

- 接入 verl-agent；
- R0 纯准确率；
- 同步 rollout；
- token/action mask 审计；
- checkpoint reload；
- 短程收敛验证。

完成条件：权重真实变化，冻结评测可重复。

### Phase 5：Adaptive Reward

- R1 非法行为；
- R2 工具成本；
- R3 token 成本可选；
- λ/μ/η 敏感性；
- 固定策略对照。

完成条件：能够回答 Adaptive 假设，结论允许为否。

### Phase 6：评测与分析

- 三随机种子；
- MATH/AIME/Omni-MATH；
- Pareto、难度和题型分析；
- 失败 taxonomy；
- reward hacking 审计。

完成条件：报告可由原始记录重新生成。

### Phase 7：产品 Demo

- API、流式运行和取消；
- Session；
- UI；
- 轨迹与工具证据；
- 限流和监控；
- 部署说明。

完成条件：线上不使用 ground truth，仍能运行同一 Agent core。

## 22. 预期效果与验收

### 22.1 可以合理预期

实施完成后，项目将从“规则工作流 + 模拟 RL”升级为：

- 真实模型生成动作；
- 真实工具环境；
- 真实 token/logprob/advantage；
- 真实参数更新；
- 可验证数学奖励；
- 可重载 checkpoint；
- 可量化 Adaptive 行为；
- 可运行 To-C Demo；
- 可复现实验报告。

### 22.2 需要实验才能确认

- GRPO 是否提高 4B 的数学正确率；
- 工具策略是否真正按难度变化；
- R2 是否改善 success–cost Pareto；
- SFT 是否限制 RL 探索；
- Python 与 SymPy 是否都带来净收益；
- 三天预算是否足以获得稳定提升。

### 22.3 最低成功标准

- Verifier 黄金集 ≥99%，对抗集中无已知 false positive；
- tool/final 协议成功率 ≥98%；
- 模型参数真实更新且 checkpoint 可重载；
- Base、Base-Agent、SFT、GRPO-R0、GRPO-R2 可公平比较；
- 至少一个 RL 配置在冻结集上稳定优于对应 SFT，或给出可信的失败诊断；
- Adaptive 方法在至少一个预算区间形成更优 Pareto 点；
- 正式结果可追溯到 commit、config、dataset hash 和原始轨迹。

### 22.4 理想成功标准

- R0 提升数学 Pass@1；
- R2 在相近正确率下降低无效工具调用；
- 难题工具调用增加、简单题直接回答增加；
- 错误工具调用后出现可验证的恢复；
- 在严格保留集上仍有一致收益；
- 至少一个上游贡献被接受或得到实质 review。

## 23. 风险与缓解

| 风险 | 表现 | 缓解 |
|---|---|---|
| 全零奖励 | AIME 轨迹全部错误 | SFT、课程、动态采样 |
| 全一奖励 | 简单题无梯度 | 提高难度或学习成本 |
| Reward hacking | 错误答案得高分 | 对抗集、fail-closed、人工抽检 |
| 工具滥用 | 调用多但无收益 | R0/R2 对照、成本只作用于正确轨迹 |
| 工具回避 | 快速猜错 | 成功奖励严格主导 |
| SFT 过拟合 | 只会固定模板 | 行为混合、短 SFT、RL 探索 |
| Token drift | 训练与重放 token 不同 | 保存 token ID、统一模板 |
| 数据污染 | 公开榜单虚高 | 严格保留集、语义去重、披露 |
| Sandbox 逃逸 | 读取系统或阻塞 | 独立容器、配额、无网络 |
| 云端失败 | 三天预算浪费 | 多级 gate、checkpoint、预构建镜像 |
| 框架升级 | 接口破坏 | 固定 commit、Adapter 隔离 |
| 结果不提升 | RL 无有效收益 | 诚实报告、失败分析、缩小命题 |

## 24. 后续是否需要继续完善

需要，但必须按证据驱动，不继续堆概念。

### V2：过程监督

- 先接入现成数学 PRM 做 rerank 和错误定位；
- 再训练独立 PRM；
- 比较 outcome-only、PRM rerank 和 process reward；
- 评测 calibration、首错定位和跨模型泛化。

可参考 [PRM800K](https://github.com/openai/prm800k) 和 [Math-Shepherd](https://arxiv.org/abs/2312.08935)。

### V2：规模与系统

- 7B/8B；
- 异步 rollout；
- policy staleness；
- Tool Worker 批处理；
- context compression；
- 长尾轨迹调度；
- 吞吐、MFU、显存和容错报告。

只有测得工具等待导致明显 GPU idle 后，才引入异步。

### V3：形式化数学

- Lean；
- tactic-level 环境；
- 形式化 proof reward；
- 自然语言解答到形式证明的桥接；
- 过程级信用分配。

### V3：产品数据闭环

- 用户匿名反馈；
- 错题聚类；
- 人工复核队列；
- 难度与失败 curriculum；
- 新 Policy 离线评测、灰度、回滚；
- 隐私和数据保留策略。

## 25. 求职辨识度判断

如果只完成框架接入或 GSM8K demo，项目辨识度有限。

如果完成到 Phase 6，并公开真实证据，它可以成为 Agent/Post-training 校招或初级岗位中较有辨识度的项目，因为它同时展示：

- 数学 RLVR；
- Agent 环境设计；
- SFT 与 GRPO；
- Tool-Integrated Reasoning；
- 动态采样与稀疏奖励；
- Verifier 和 reward hacking；
- token-level 训练正确性；
- 本地—云端工程；
- 统计评测与失败分析；
- 产品部署边界。

对于核心研究岗，还需要更强原创算法或论文；对于训练 Infra 岗，还需要实际吞吐、调度、容错或框架上游贡献。本项目最匹配 Agent 算法、推理 Post-training 和应用算法工程岗位。

## 26. 最终完成定义

只有同时满足下列条件，项目才可以宣称 V2 完成：

- 不存在模拟 Policy、随机 loss 或伪造曲线；
- 从题目到工具、reward、梯度和 checkpoint 闭环真实；
- 训练与 Serving 共用 Agent core；
- Hidden Verifier 不进入线上和 Policy 上下文；
- 数据来源、许可证、split 和 hash 完整；
- Verifier、Sandbox 和 reward hacking 测试通过；
- 至少有 Base/SFT/GRPO/Adaptive 对照；
- 主要结论有多种子或明确限制；
- 所有图表可由原始结果生成；
- README 能让第三方复现；
- 未完成项和负面结果被如实披露。

## 27. 推荐的下一步

1. 评审并冻结本报告；
2. 初始化或恢复 Git，保存当前 V1；
3. 基于本报告生成逐文件、逐测试的实施计划；
4. 首先实现数据 schema 与 Verifier，而不是先租 GPU；
5. 通过本地门禁后再接入 verl-agent；
6. 完成 1.7B smoke 后确定 4B 云端资源；
7. 最后才执行三天正式训练窗口。

这一顺序把最昂贵、最不可逆的云端训练放在所有可本地验证的风险之后。
