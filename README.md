# 自适应复杂任务求解器

> 基于langgraph状态机编排与 GRPO 强化学习的智能任务求解器

## AdaptiveMath-RL 2.0 开发状态

本仓库正在升级为 **AdaptiveMath-RL 2.0**：通过 SFT + Agentic GRPO 训练一个可自主选择直接推理 / Python / SymPy 工具的真实数学 Agent。V1 规则工作流与模拟 GRPO 代码保留在 `src/` 下作为原型基线，新功能写入 `src/adaptive_math`。

实施文档：

- [产品与技术设计报告](docs/superpowers/specs/2026-09-09-adaptive-math-rl-product-design.md)
- [Master Roadmap](docs/superpowers/plans/2026-09-10-adaptive-math-rl-master.md)
- [Foundation Plan](docs/superpowers/plans/2026-09-10-adaptive-math-rl-foundation.md)
- [Agent Runtime Plan](docs/superpowers/plans/2026-09-10-adaptive-math-rl-agent-runtime.md)
- [Training & Cloud Plan](docs/superpowers/plans/2026-09-10-adaptive-math-rl-training-cloud.md)
- [Evaluation & Product Plan](docs/superpowers/plans/2026-09-10-adaptive-math-rl-evaluation-product.md)

### 环境与测试

- Python 3.12（由 uv 管理项目环境）
- 安装依赖：`uv sync --dev`
- 运行测试：`uv run pytest`
- 代码检查：`uv run ruff check src tests scripts`
- 类型检查：`uv run mypy`

### 数据管线

- 数据源治理：`configs/data/sources.yaml`（固定 revision + license + 用途 + 引用）
- 构建（可复现、确定性）：

```bash
uv run python scripts/data/build_dataset.py \
  --registry configs/data/sources.yaml \
  --output-dir data/processed/v1 \
  --manifest data/manifests/v1.json \
  --seed 20260910
```

- 审计（hash 不匹配 / split 泄漏 / 重复 task_id / 不可验证 reference 都会非零退出）：

```bash
uv run python scripts/data/audit_dataset.py --manifest data/manifests/v1.json
```

- 下载数据不入库（`data/processed`、`data/raw` 被忽略）；manifest 与文档受版本管理
- 操作手册：[docs/runbooks/data-and-verifier.md](docs/runbooks/data-and-verifier.md)
- 门禁报告：[docs/results/foundation-validation.md](docs/results/foundation-validation.md)

### Agent Runtime（V2）

- 运行时执行严格的 `<tool_call>` / `<final>` 协议，轨迹可 hash 校验、可离线 Replay。
- 产品路径没有参考答案；仅离线评估环境在终止后通过私有验证边界计算正确性。
- 本地无权重 smoke（不下载模型）：

```bash
uv run python scripts/dev/run_agent.py \
  --problem "Compute 17 * 19." --answer-type integer \
  --config configs/agent/direct.yaml \
  --scripted-action '<final>{"answer":"323"}</final>'
```

- 真实本地模型路径会使用 tokenizer 的 chat template；需要在云镜像或本机额外安装 `transformers`、`torch` 后传入 `--model`。Python 工具执行必须连接 Linux SandboxFusion，绝不会回退到宿主机执行。
- 回放：`uv run python scripts/dev/replay_trace.py --trace tests/fixtures/golden_traces/direct_correct.json --verify-hash --print-events`

## 项目概述

    为解决通用大模型(如GPT-4、Qwen-3)在处理简单任务时性能过剩、在处理复杂任务时能力不足的问题,构建了一个融合动态路由、状态机编排与强化学习微调的智能 Agent 系统，旨在解决通用模型在复杂逻辑任务中成本高且准确率低的痛点。

### 核心问题

1. **成本问题**：无论任务难易都使用高性能模型（如 GPT-4o、Qwen-Max），导致推理成本居高不下
2. **能力局限**：传统 Agent 流程固定，缺乏自我修正能力，遇到工具调用失败时容易中断
3. **领域适配**：通用模型在特定垂直领域（如数学证明、逻辑推理）表现不够优秀

### 解决方案

通过三大核心技术解决上述问题：

| 技术 | 作用 
|------|------|
| **动态路由网关** | 实现 Context-Aware 路由策略，根据任务复杂度智能选择模型 |
| **状态机编排** | 基于 LangGraph 构建可回退 Agent 工作流,提升复杂任务完成率|
| **在线强化学习** | 使用 GRPO 算法进行领域特定优化 | 
---

## 项目结构

```
adaptive-solver/
├── src/                           # 源代码目录
│   ├── main.py                    # 程序主入口
│   │
│   ├── config/                    # 配置管理模块
│   │   └── settings.py            # 使用 Pydantic Settings 管理配置
│   │
│   ├── router/                    # 动态路由模块 ⭐
│   │   ├── complexity_analyzer.py # 任务复杂度分析器
│   │   └── router.py              # 路由决策器
│   │
│   ├── graph/                     # 状态机编排模块 ⭐
│   │   ├── state.py               # 状态定义和管理
│   │   ├── nodes/                 # 状态节点
│   │   │   ├── planner.py         # 规划节点
│   │   │   ├── tool_caller.py     # 工具调用节点
│   │   │   ├── critic.py          # 反思/检查节点
│   │   │   └── executor.py        # 执行节点
│   │   └── workflow.py            # 状态图定义
│   │
│   ├── llm/                       # LLM 模型封装
│   │   └── providers.py            # 阿里云 Qwen 模型提供商
│   │
│   ├── rl/                        # 强化学习模块 ⭐
│   │   ├── data/
│   │   │   └── gsm8k_loader.py     # GSM8K 数据集加载器
│   │   ├── trajectory_collector.py # 轨迹数据收集器
│   │   ├── reward_function.py      # 多维度奖励函数
│   │   ├── grpo_trainer.py         # GRPO 训练器
│   │   └── training_visualizer.py  # 训练可视化工具
│   │
│
├── tests/                         # 测试模块
│   ├── test_router.py              # 路由模块测试
│   ├── test_graph.py               # 状态机测试
│   └── test_rl.py                  # 强化学习测试
│
├── examples/                      # 示例代码
│   ├── simple_task.py             # 简单任务示例
│   ├── complex_task.py            # 复杂任务示例
│   ├── router_demo.py             # 路由演示
│   └── rl_demo.py                 # 强化学习演示
│
├── autodl_train.py                # AutoDL 训练脚本 
├── training_config.json            # 训练配置文件 
└── requirements.txt                # 依赖列表
```

---

## 示例执行流程

### 1. 简单任务执行流程

```
用户输入 "计算 1+1"
    ↓
AdaptiveSolver.solve(query)
    ↓
Router.route(query)
    ↓
ComplexityAnalyzer.analyze(query)
    - Token 长度: 4
    - 任务类型: CALCULATION
    - 复杂度: SIMPLE (0.95 置信度)
    ↓
Router._make_decision()
    → RouteDecision.DIRECT_TURBO
    ↓
AdaptiveSolver._solve_with_turbo()
    → 调用 Qwen Turbo API
    → 返回答案
```

### 2. 复杂任务执行流程

```
用户输入 "证明：任意两个连续整数的乘积是偶数"
    ↓
AdaptiveSolver.solve(query)
    ↓
Router.route(query)
    ↓
ComplexityAnalyzer.analyze(query)
    - Token 长度: 20
    - 任务类型: PROOF
    - 复杂度: MEDIUM/COMPLEX
    ↓
Router._make_decision()
    → RouteDecision.WORKFLOW
    ↓
AdaptiveSolver._solve_with_workflow()
    ↓
Workflow.run(query)
    ↓
┌─────────────────────────────────────┐
│          状态机执行流程             │
└─────────────────────────────────────┘
    ↓
State: {"status": "planning"}
    ↓
Planner(state)
    - 生成计划: "1.明确命题 2.选择证明方法 3.执行证明 4.验证结论"
    - 生成子任务: ["明确命题", "选择证明方法", "执行证明", "验证结论"]
    - State: {"status": "executing", "subtasks": [...], "current_index": 0}
    ↓
State: {"status": "executing", current_index": 0}
    ↓
ToolCaller(state)
    - 当前子任务: "明确命题"
    - 选择工具: logic_checker
    - 执行工具: "命题：两个连续整数 n, n+1，必有一个是偶数"
    - 记录结果，current_index += 1
    ↓
State: {"status": "executing", current_index = 4 (全部完成)}
    ↓
Critic(state)
    - 检查: 所有工具结果无错误
    - 决策: approved
    - State: {"status": "reviewing"}
    ↓
Executor(state)
    - 整合结果: "## 证明\n\n**命题**：...\n\n**证明过程**：..."
    - State: {"status": "completed"}
    ↓
返回答案
```

### 3. Self-Correction 触发流程

```
ToolCaller 执行失败
    ↓
Critic(state)
    - 检查: tool_results 包含错误 "未找到可计算的数字"
    - 判断: contains_error = True
    - 决策: rejected
    - 检查: error_count < max_retries
    - State: {"status": "planning", "error_count": 1}
    ↓
Planner(state)
    - 重新规划
    - 生成新的计划
    - State: {"status": "executing", "error_count": 0}
    ↓
继续执行...
```

---

## GRPO 训练具体过程

### 训练流程图

```
┌─────────────────────────────────────────────────────────┐
│                    GRPO 训练流程                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  1. 数据准备阶段                                       │
│  ├─ 加载 GSM8K 数据集 (7473 条)                    │
│  ├─ 格式化为训练数据 (query + expected_answer)      │
│  ├─ 模拟生成初始轨迹（或使用状态机生成）           │
│  └─ 收集到训练数据集                                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  2. 奖励计算阶段                                       │
│  对每条轨迹计算多维度奖励：                           │
│  ┌──────────────────────────────────────────────────┐ │
│  │ • 准确性奖励 (Accuracy Reward)                   │ │
│  │   - 检查答案是否正确                             │ │
│  │   - 验证计算结果                                 │ │
│  │   - 检查证明逻辑                                 │ │
│  │                                                  │ │
│  │ • 效率奖励 (Efficiency Reward)                   │ │
│  │   - 执行步数（越少越好）                       │ │
│  │   - 工具调用次数                                 │ │
│  │                                                  │ │
│  │ • 安全性奖励 (Safety Reward)                     │ │
│  │   - 错误处理                                     │ │
│  │   - 回退次数                                     │ │
│  │   - 状态一致性                                   │ │
│  │                                                  │ │
│  │ • 一致性奖励 (Consistency Reward)               │ │
│  │   - 结果与查询的关联度                           │ │
│  │   - 关键信息匹配度                               │ │
│  └──────────────────────────────────────────────────┘ │
│                                                          │
│  总奖励 = Σ(奖励_i × 权重_i)                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  3. 优势值计算 (Advantage Calculation)              │
│                                                          │
│  使用 GAE (Generalized Advantage Estimation):          │
│                                                          │
│  A_t = Σₖ≥ₜ (γλ)ᵏ⁻ᵏ * (rₜ + γV(sₜ₊₁) - V(sₜ))        │
│                                                          │
│  其中：                                                 │
│  • γ (gamma) = 0.99 - 折扣因子                         │
│  • λ (lambda) = 0.95 - GAE 参数                        │
│  • V(s) = 价值函数估计                                 │
│  • r = 奖励                                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  4. GRPO 策略更新                                     │
│                                                          │
│  对每个小组 (Group) 的样本：                          │
│  ┌──────────────────────────────────────────────────┐ │
│  │ 1. 生成多个候选答案（Group Size = 4）            │ │
│  │                                                   │ │
│  │ 2. 计算相对优势：                               │ │
│  │   A_i - mean(A_group)                             │ │
│  │                                                   │ │
│  │ 3. 计算策略损失：                                │ │
│  │   L_policy = -mean(advantage * log_ratio)        │ │
│  │                                                   │ │
│  │ 4. KL 散度惩罚：                                │ │
│  │   L_kl = β * KL(π_old || π_new)                 │ │
│  │                                                   │ │
│  │ 5. 总损失：                                      │ │
│  │   L_total = L_policy + L_kl                       │ │
│  │                                                   │ │
│  │ 6. 反向传播更新模型参数                          │ │
│  └──────────────────────────────────────────────────┘ │
│                                                          │
│  超参数：                                               │
│  • learning_rate = 5e-6                               │
│  • kl_penalty = 0.1                                   │
│  • clip_range = 0.2                                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  5. 评估与可视化                                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │ • 计算训练指标（平均奖励、成功率、步数）       │ │
│  │ • 绘制训练曲线                                   │ │
│  │ • 生成完成率对比图（用于简历）                 │ │
│  │ • 保存检查点                                      │ │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
                    训练完成
```
