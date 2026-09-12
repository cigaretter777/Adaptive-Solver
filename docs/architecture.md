# AdaptiveMath-RL 代码架构总览

> 生成日期：2026-09-10 · 基于 commit b3f3f25 · 257 项测试全绿
> 本文档描述 `src/adaptive_math`（新架构）的实际代码职责与数据流。
> V1 原型（`src/main.py`、`src/{config,graph,llm,rl,router}`）不参与新链路，迁移期后删除。

## 1. 分层结构

依赖方向严格自上而下：上层可 import 下层，反之禁止。

```
L6 training/   训练层    SFT 记录·tokenize/mask·rollout·奖励桥·配置
L5 data/       数据层    注册·清洗·去重·切分·manifest
L4 agent/      运行时层  动作协议·状态机·环境·AgentLoop·轨迹·模型客户端
L3 tools/      工具层    注册表·SymPy 子进程·SandboxFusion HTTP
L2 reward/     奖励层    R0–R3 纯函数
L1 verifier/   验证层    提取·规范化·类型化验证·符号 worker
L0 core/       契约层    MathTask/Budget/JSONValue·内容寻址哈希
```

## 2. 各模块职责

### L0 core/ — 全局契约（所有层共享）

| 文件 | 职责 |
|---|---|
| `types.py` | `MathTask`（公开、frozen、含 pipeline_version）、`ReferenceAnswer`（隐藏答案）、`LabeledMathTask.public_view()`（唯一脱敏出口）、`Budget`（步数/工具/Python 秒/观察字符四限额）、递归 `JSONValue`（拒绝 NaN/Inf，metadata 存储递归冻结） |
| `hashing.py` | `canonical_text`（NFKC+行尾+空白折叠，不做代数化简）、`make_task_id`（dataset:SHA-256 前 20 位，内容寻址）、`make_source_hash`（原始记录字节 SHA-256） |

### L1 verifier/ — 确定性答案验证（隐藏答案的唯一消费者）

| 文件 | 职责 |
|---|---|
| `extractor.py` | 双模式提取：`extract()` 严格模式（模型输出：<final> → \boxed → $ 组，多候选=AMBIGUOUS）；`extract_solution_answer()` 解答模式（数据集解题文本：尾窗扫描，最后 \boxed → 散文锚点（倒序）→ 全文单数学段）。均有界（32K/嵌套 64）、永不抛异常 |
| `normalizer.py` | `normalize_surface`：仅表示保持清理（NFKC、Unicode 减号、frac 拼写、一层外围定界符），幂等，不做代数化简；千分位在类型化解析层处理（避免集合记法误伤） |
| `numeric.py` | 有界精确解析（Decimal/Fraction，永不 float）：整数/有理/实数（atol+rtol 容差）/集合（无序去重）/元组（有序）/区间（开闭端点）；max_digits=256、max_items=128、max_depth=8 |
| `symbolic.py` | 符号等价比较（只在 worker 进程内运行）：math-verify `parse_latex_cached` 解析 → 限制（8192 字符/4096 节点/32 自由符号/未知函数拒绝）→ 证据顺序：canonical 相等 → 域感知化简差（singularities 对比，防 (x²-1)/(x-1)=x+1 假阳性）→ task_id 种子的 5 点数值交叉验证 |
| `worker.py` | spawn 隔离进程 + 双超时：请求级 SIGPROF CPU 定时器（worker 存活）+ 父进程墙钟 deadline（杀死重建）；macOS 上 RLIMIT_AS best-effort |
| `service.py` | `verify_answer()` 唯一公开入口：按 answer_type 分发到 7 个类型化验证器；六状态（CORRECT/INCORRECT/INVALID_PREDICTION/INVALID_REFERENCE/TIMEOUT/INTERNAL_ERROR）；reward 仅 CORRECT=1.0；acceptable_forms 只能升级不能降级 |
| `hidden.py` | `HiddenVerifier`：离线训练/评测专用包装，产品环境在类型上拿不到 |

### L2 reward/ — 纯函数奖励（只消费 VerifierResult，不重新解析答案）

| 文件 | 职责 |
|---|---|
| `types.py` | `RewardContext`（verifier 结果+工具/Python/invalid/token 用量+Budget）、`RewardConfig`（全字段显式必填+config_hash）、`RewardBreakdown`（total+components+version+hash） |
| `functions.py` | R0=correct；R1=−invalid 惩罚(封顶)；R2=correct×(1−工具/Python 归一化成本)−invalid；R3=R2−token 成本。成本仅作用于正确轨迹；零分母→零成本；clip [−1,1]。四个 YAML 配置与代码预设由测试锁定一致 |

### L3 tools/ — 有界工具执行

| 文件 | 职责 |
|---|---|
| `base.py` | `Tool` Protocol、`ToolResult`（ok/output/error_code/latency/truncated）、`ToolContext`（trace_id+剩余预算，永不含答案）、六错误码 |
| `registry.py` | 注册/重名拒绝/参数 Pydantic 校验/UTF-8 安全截断到剩余观察预算/描述从 arguments_model JSON schema 生成（prompt 与校验不漂移） |
| `sympy_tool.py` | 白名单操作（simplify/factor/expand/solve/diff/integrate/numeric），专用子进程池（3s/768MB），solve 结果稳定排序 |
| `sandboxfusion.py` | 外部 Python 执行服务的 fail-closed HTTP 客户端（`ADAPTIVE_MATH_SANDBOX_URL`，连接 1s/请求 6s、无重试、健康检查独立） |
| `python_tool.py` | 只接受 `code:str` 的门面；主进程零 exec/eval/subprocess |

### L4 agent/ — 训练与产品共用的运行时

| 文件 | 职责 |
|---|---|
| `actions.py` | `ToolCall`（名称 pattern 锁定）/`ToolAction`/`FinalAction`，extra=forbid，非可执行数据 |
| `parser.py` | `parse_action`：锚定有界解析「(think)? (tool_call|final)」，六错误码；Hypothesis 证明永不抛异常 |
| `state.py` | 不可变 `AgentState`（事件序列号/单调时钟/用量累计/预算判定/终止转移），终止后拒绝任何转移 |
| `trace.py` | `Trajectory` 契约（trace_id/task_id/events/final_answer/termination_reason/usage/runtime_version） |
| `environment.py` | `ProductMathEnv`（构造器/字段/序列化层面都无法接触答案）与 `OfflineMathEnv`（隐藏 reference 存于私有评估上下文，`evaluate()` 仅在终止后可用）；invalid 动作消耗步数并给语法提醒；工具槽先扣再执行；Python deadline 受剩余秒数封顶 |
| `loop.py` | `AgentLoop.run`：生成→记录原始输出→解析→环境 step→终止判定；模型异常/取消→明确终止原因 |
| `prompts.py` | 从公开任务+剩余预算+活工具 JSON schema 渲染 system/user；prompt_version 进每条轨迹 |
| `model_client.py` | 唯一模型边界：`ModelClient` Protocol（async generate）+ ChatMessage/GenerationConfig/ModelTurn（含真实 token 计数） |
| `transformers_client.py` | 本地 HF 实现：延迟导入 torch、强制 chat_template、temperature=0 走贪心、stop_strings 截断、token 计数如实返回 |
| `replay.py` | `TraceEnvelope`（schema/runtime/prompt 版本+created_at 在哈希外）+ 内容寻址哈希 + `replay()` 离线重放（不调模型/工具/verifier），篡改即拒 |

### L5 data/ — 可追溯数据管线

| 文件 | 职责 |
|---|---|
| `sources.py` | `SourceSpec`（name/uri/40 位不可变 revision/license/intended_use/citation/loader/answer_type）+ 注册表 + 加载器（HF datasets 按 revision 钉死；DAPO 布局适配器；synthetic 用于测试） |
| `canonicalize.py` | 记录→`LabeledMathTask`：确定性排序、解答/答案双列提取、**reference 自检**（verify 自身，不过关即隔离）、五类机器可读隔离原因 |
| `deduplicate.py` | 精确（canonical SHA-256）+ 近似（字符 5-gram MinHash，种子化 hashfunc，阈值 0.90）；跨 split 簇评测成员胜出 |
| `split.py` | 种子洗牌 + 精确计数切分 train/sft_dev/rl_dev（90/5/5）；eval 源全部进 frozen_eval；重标 split 字段 |
| `manifest.py` | `DataManifest`：逐源 revision/resolved/许可/计数 + 逐 split 文件哈希/task_ids 哈希 + dedup/隔离统计 |

### L6 training/ — SFT 与 GRPO 桥接（verl 只作固定版本后端）

| 文件 | 职责 |
|---|---|
| `upstreams.py` + `third_party/manifest.json` | verl-agent/verl/SandboxFusion 的 40 位 SHA 钉死校验，禁止静默前移 |
| `sft_records.py` | `SFTTrajectoryRecord`：角色序列校验（system,user,(assistant,tool*)*,assistant-FINAL）、每个 assistant 过生产 parser、final 与动作一致、四行为桶（DIRECT/PYTHON/SYMPY/RECOVERY） |
| `sft_builder.py` | 生产轨迹→SFT 示范：行为分类（工具名+恢复检测）、工具调用经生产注册表重放、拒绝参考串泄漏 |
| `sft_io.py` | Parquet 持久化（zstd）+ 结构审计 |
| `teacher_rollout.py` | 教师采样：同一 AgentLoop+OfflineMathEnv 跑候选轨迹，只保留 verifier-CORRECT |
| `tokenization.py` | chat-template 前缀差分定位 assistant span；labels 仅 assistant token，其余 IGNORE_INDEX；超长拒绝不截断；tokenizer revision 强制 |
| `collator.py` | 确定性 padding（长度不齐报错不静默）、revision 一致性校验 |
| `rollout_records.py` | `RolloutStep/RolloutTrajectory`：保存生成时 token IDs+mask；组内 policy_version 一致性校验 |
| `verl_environment.py` | `MathRolloutManager`：后端中立的向量化 reset/step/build_text_obs（每环境独立 OfflineMathEnv，终局才给 reward）；未来 pinned verl-agent 适配器的核心 |
| `reward_bridge.py` | `reward_for_trajectory`（生产 reward 函数薄适配）+ `group_advantages`（(r−mean)/(std+ε)，全零/全一组标记 ineffective→零 advantage） |
| `config.py` | `SFTConfig`：frozen、revision 40-hex 强制、manifest 64-hex 强制、LoRA/批量/种子全显式 |

## 3. 脚本入口

| 脚本 | 工作 |
|---|---|
| `scripts/data/build_dataset.py` | 注册表→加载→规范化→自检→去重→切分→Parquet+manifest（--dry-run/--sample-per-source） |
| `scripts/data/audit_dataset.py` | 哈希/schema/计数/跨 split 泄漏/许可/reference 重验证，任一失败 exit≠0 |
| `scripts/data/build_sft_trajectories.py` | labeled tasks + 运行时 traces → SFT Parquet（经生产注册表重放工具调用） |
| `scripts/data/audit_sft_trajectories.py` | SFT 工件结构与覆盖审计（单源>60% 或协议<100% 即失败） |
| `scripts/data/generate_teacher_traces.py` | 教师模型采样 verifier-correct 公开轨迹（GPU） |
| `scripts/train/run_sft.py` | SFT LoRA 训练：占位哈希拒训、manifest 哈希门禁、--set 覆盖留痕、原子写 resolved_config/environment/metrics/adapter |
| `scripts/dev/run_agent.py` | 产品/离线 CLI：真模型或脚本化模型跑单题，输出 Trajectory JSON |
| `scripts/dev/replay_trace.py` | 轨迹哈希校验+事件重放 |
| `scripts/setup/pin_upstreams.py` | 上游后端 SHA 解析/校验/diff（--update 显式前移） |

## 4. 三条主数据流

```
① 数据管线（已完成，v1 已物化）
HF pinned revision → sources → canonicalize(+quarantine) → dedup(精确+MinHash)
  → split(种子) → Parquet×4 + v1.json manifest → audit ✅
  产物：100,804 任务（train 87,620/sft_dev 4,868/rl_dev 4,868/frozen_eval 3,448）

② SFT 管线（本地物化代码就绪，正式数据未跑）
pinned raw source → build_source_sft（DIRECT：来源解答→think+FINAL；
  TIR：fenced Python 经生产 ToolRegistry 重放）→ trace JSONL + SFT Parquet
  → teacher_rollout（GPU，RECOVERY/补量 ⬜未跑）
  → build_sft_record → sft_v1.parquet → audit（≥18k 覆盖门禁）
  → run_sft（tokenize+mask → Trainer+LoRA → adapter+COMPLETE）⬜未跑

③ GRPO 管线（云端真实入口与固定上游适配器就绪，尚未 GPU smoke）
rl_dev prompts → MathRolloutManager.reset/step（向量化 OfflineMathEnv）
  → run_grpo.py + configs/grpo（输入 SHA 门禁、Hydra resolved config、上游 SHA 校验）
  → verl_agent_adapter（对固定 checkout 的幂等 `adaptive_math` 环境注册补丁）
  → pinned verl-agent / Ray actor 后端
  → 终局 HiddenVerifier → reward_bridge(R0/R2，逐项 reward info) → group_advantages
  → 策略更新（仅 assistant token，mask 已由 rollout_records 保存）

④ 云端可复现层（脚本与运行手册就绪，尚未真实镜像构建）

  immutable CUDA base digest + Dockerfile.train + pinned verl-agent checkout
  → compose.sandbox.yml（独立 SandboxFusion）
  → preflight / launch_sft / launch_grpo / resume / artifact sync / health stop
  → tmux-backed cloud runbook
```

## 5. 关键架构不变量（由契约测试钉死）

1. **答案隔离**：`ProductMathEnv` 无任何路径接触 reference；泄漏契约测试递归扫描公开序列化（answer/reference/ground_truth/expected_output/verifier_handle 键即失败）
2. **单一动作协议**：训练/评测/产品共用同一 parser 与 prompt；SFT 记录的每个 assistant 消息必须过生产 parser
3. **零本地执行**：模型生成的 Python 只经 SandboxFusion HTTP；主进程无 exec/eval/subprocess（唯一例外：verifier/SymPy 的有界 worker 进程）
4. **确定性**：同输入→逐字节一致轨迹/奖励/切分；种子重建树一致；reward/verifier 全纯函数
5. **fail-closed**：验证/提取/工具永不猜测——不可靠即枚举状态隔离，错误答案与基础设施故障不混淆
6. **版本钉死**：模型/数据/上游后端全部 40-hex revision；每次运行写 resolved_config+environment.json

## 6. 当前缺口（截至本文档生成）

| 缺口 | 所属 | 性质 |
|---|---|---|
| `run_grpo.py` + `configs/grpo/*` + profiling | Training Task 7 | 本地代码+GPU 门控测试 |
| 云端脚本/runbook/requirements.cuda.lock（AutoDL 适配） | Training Task 8 | 本地代码 |
| `evaluation/`（registry/runner/统计/报告） | Phase 4 Task 1–4 | 本地代码 |
| `serving/` + demo | Phase 4 Task 5–7 | 本地代码 |
| 教师轨迹、SFT/GRPO 真跑、冻结评测 | — | GPU（Kaggle/AutoDL） |
| 真模型冒烟（Runtime exit criteria） | — | GPU 或 Mac MPS |
