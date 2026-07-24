# RLVR

> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · [← 演进总览 §2](../reports/deepseek-version-lineage-20260625.md#2-版本时间线与关系) · [← R1 梗概](./r1.md) · [Raschka §RLVR/GRPO](../reports/raschka-technical-deepseek-v3-v32-highlights.md#rlvr--grpo-要点326)
> **论文**：[DeepSeek-R1 arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

## 核心结论摘要

- **RLVR**（Reinforcement Learning with Verifiable Rewards）用规则/验证器给奖励，无需神经 RM。
- **GRPO** 在组内做相对 advantage 优化，去掉 critic 网络。
- R1 后训练核心；适合数学、代码等可自动判对错任务。
- 与 PPO+神经 RM 的 RLHF 路线形成对照（见 GRPO vs PPO 图）。

---

## 一句话

**RLVR** = 在强化学习里 **不用神经 reward model**，而对 **可程序/符号验证** 的任务（数学答案、代码单测、格式规则）直接给 **0/1 或规则分** 作奖励；DeepSeek 用 **GRPO** 做优化（无 critic 的组内相对 advantage）。**R1** = V3-Base **架构不变** + RLVR 后训练。

---

## 和 RLHF / GRPO 的关系

<img src="../figures/rl/grpo-vs-ppo.svg" alt="PPO vs GRPO：RLHF 神经 RM + Critic 与 RLVR 验证器 + 组内 baseline 对比" width="920"/>

[图示详情](../figures/rl/grpo-vs-ppo.svg)

| 路线 | 奖励从哪来 | 优化算法 | 典型场景 |
|------|-----------|----------|----------|
| **RLHF** | 人类偏好训练的 **神经 RM** | PPO（需 critic） | 开放域对齐 |
| **GRPO** | 任意标量奖励 | 组内采样 $G$ 条，相对 baseline 算 advantage；**无 critic** | 省显存、易扩展 |
| **RLVR + GRPO** | **规则 / 验证器**（计算器、sympy、单测、格式检查） | 同上 GRPO | 数学、代码、可判定推理 |

> 同一条 prompt 采样 $G$ 次 rollout：每条用 verifier 打分（对/错、格式、语言）→ 组内减均值得 advantage → 更新 policy（R1 上即 V3-Base 权重）。

**RLVR 省掉什么**：不训、不依赖 **reward model**，减轻 **reward hacking**（模型讨好 RM 而非真做对题）。

**RLVR 局限**：只适合 **答案可验证** 的短程任务；开放域写作、主观 helpfulness 仍需 RM 或 LLM-as-judge（V3.2 后训练即 **RLVR + 生成式 RM 混合**）。

---

## DeepSeek-R1 里的 RLVR

<a id="grpo"></a>

### 算法：GRPO

- 同一 prompt 生成 **$G$ 条** 完整回答（R1 一阶段约 **16 rollout / 题**）
- 每条算规则奖励 → **组内相对** advantage（无 value network）
- 配合 **KL 到 reference**、clip 等稳定训练

### R1 奖励

| 奖励 | 作用 | 阶段 |
|------|------|------|
| **Accuracy / verifier** | 数学、代码等 **对错**（sympy、单测等） | R1-Zero、R1 一阶段 RL |
| **Format** | 思考/答案分隔、`` 等结构 | R1-Zero 起 |
| **Language consistency** | 惩罚中英混杂，鼓励与问题同语言 | R1 二阶段 RL 起 |

### 两条产物

| 模型 | 路径 | 要点 |
|------|------|------|
| **R1-Zero** | V3-Base → **纯 GRPO + RLVR**（无 SFT 冷启动） | 推理能力 **自发涌现**（长度增长、自反思）；可读性差 |
| **R1** | 冷启动 SFT → RL → 拒绝采样 SFT → RL | 在 R1-Zero 能力上补 **可读性、通用任务、安全** |

详见 [R1 四阶段训练 pipeline](../material/papers/deepseek-r1/training-pipeline.md)（含 Dev-1→R1 与 Table 3 指标）。

---

## 在 DeepSeek 系列中的位置

| 版本 | 与 RLVR 关系 |
|------|-------------|
| **V3** | Base；**无** RLVR |
| **R1** | **RLVR + GRPO** 主路径；架构 **同 V3** |
| **V3.1 / Terminus** | Hybrid 对话；训练 pipeline 不同，非 R1 专用推理模型 |
| **V3.2** | 继承 R1 系 GRPO 经验 + **生成式 RM**（开放域）+ DeepSeekMath V2 过程奖励；[Raschka 对比](../reports/raschka-technical-deepseek-v3-v32-highlights.md#rlvr--grpo-要点326) |

<img src="../figures/rl/rlvr-posttrain-branch.svg" alt="V3-Base 后训练分叉：R1 专用推理 vs V3.1 Hybrid vs V3.2" width="920"/>

[图示详情](../figures/rl/rlvr-posttrain-branch.svg)

---

## 为何 R1 不改架构

RLVR 只改 **后训练**（采样、奖励、策略梯度），不动 MLA / MoE 结构。因此 R1 与 V3 **权重形状、KV cache 格式、推理引擎配置一致**——差异在 **行为分布**（更长 CoT、更强推理）。

---

## 延伸

| 资源 | 说明 |
|------|------|
| [DeepSeek-R1](./r1.md) | R1 一页纸梗概 |
| [DeepSeek-R1 训练 Pipeline](../material/papers/deepseek-r1/training-pipeline.md) | 四阶段 + R1-Zero 详解 |
| [Raschka 全文解析](../reports/raschka-technical-deepseek-v3-v32.md) §3.2 | RLVR vs PPO vs GRPO 对照表 |
| [GRPO 长程任务局限](../rl/optimize.md) | 社区讨论：GRPO 与长程任务局限 |

**论文**：R1 [2501.12948](https://arxiv.org/abs/2501.12948) · V3 [2412.19437](https://arxiv.org/abs/2412.19437)
