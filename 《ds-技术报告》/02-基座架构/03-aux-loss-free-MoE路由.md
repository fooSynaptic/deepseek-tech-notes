# aux-loss-free MoE 路由逻辑

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §3.3](../01-总览/01-版本演进总览.md#33-deepseek-v3) · [← MoE 线导读](../01-总览/07-MoE线导读.md) · [← DeepSeekMoE 上游](05-DeepSeekMoE.md) · [← V3 梗概](01-V3基座.md) · [版本目录](../01-总览/02-版本梗概索引.md)
> **论文**：[DeepSeek-V3 arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §2.1 · [Megatron MoE aux loss free](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)

## 核心结论摘要

- V3 用 **动态 expert bias** 做负载均衡，**去掉 aux loss 主均衡**。
- Sigmoid 路由替代 V2 softmax 系；Table 5 给出 ablation。
- 避免 aux loss 与主 loss 权衡，简化 MoE 训练调参。
- V4 Hash MoE 为更激进的下游路由方案。

---

## 一句话

**aux-loss-free** = MoE **不用（或几乎不用）辅助损失** 逼专家均匀，而是在每个 routed expert 上维护一个 **可学习的 routing bias $b_i$**，按 batch 负载 **动态加减 $b_i$** 来调 top-$K$ 路由；**门控权重仍用原始 affinity $s_{i,t}$**，避免 aux loss 伤害主任务 loss。

V3 起标配；V4 系仍沿用同一思路。

---

## 为什么 MoE 要「负载均衡」

| 问题 | 后果 |
|------|------|
| 少数 expert 过热、多数闲置 | **routing collapse**（Shazeer et al., 2017） |
| Expert Parallel（EP）下 token 分布不均 | 部分 GPU 空转、部分打满 → **算力浪费** |

因此几乎所有 MoE 都要某种 **load balancing**——关键是 **用什么机制**，以及会不会 **牺牲模型质量**。

---

## 传统做法：auxiliary loss 及其矛盾

经典 Switch / GShard 路线在总 loss 里加 **辅助损失** $L_{\mathrm{Bal}}$，惩罚专家负载不均（token 级或 sequence 级）。

| 优点 | 缺点 |
|------|------|
| 实现简单、梯度直接推路由 | **aux loss 过大** → 损害主任务表现（Wang et al., 2024a） |
| 易与训练框架集成 | **aux loss 过小** → 均衡不够，EP 效率差 |
| | **sequence-wise** aux 强制 **每条序列内** 专家均匀 → 抑制 **按领域 specialization** |

DeepSeek-V3 论文 Table 5：纯 aux-loss 基线在多数 benchmark 上 **不如** aux-loss-free。

---

## aux-loss-free 核心逻辑

### 1. 路由分数 vs 门控值

DeepSeek-V3 MoE 对 routed experts 用 **sigmoid affinity** $s_{i,t}$，再对选中专家做归一化得到门控 $g_{i,t}$（与 V2 的 softmax 路由不同，见 V3 论文 §2.1）。

**aux-loss-free 只改「谁进 top-$K$」**，不改门控乘到 FFN 输出的数值：

| 量 | 是否加 bias $b_i$ | 作用 |
|----|-------------------|------|
| **Top-$K$ 选择** | ✅ 用 $s_{i,t} + b_i$ 排序 | 决定 token 去哪个 expert |
| **门控 $g_{i,t}$** | ❌ 仍用原始 $s_{i,t}$ | 乘 expert 输出，参与前向/反传主 loss |

这样 bias 是 **纯路由调度旋钮**，不直接扭曲 expert 输出的幅度。

### 2. 每个训练 step 末尾更新 $b_i$

监控 **整个 batch**（一步训练）上各 expert 的负载：

<img src="figures/aux-loss-free-bias-update.svg" alt="过载 b_i -= γ；欠载 b_i += γ" width="520"/>

[图示详情](figures/aux-loss-free-bias-update.svg)

$\gamma$ = **bias update speed**（V3 预训练：前 14.3T tokens 取 **0.001**，论文 §3.2）。

**无 aux loss 梯度** 参与均衡；均衡靠 **启发式反馈** 调 bias，主 loss 只负责「学得好不好」。

### 3. 公式

专家 $i$ 在 token $t$ 被激活，当且仅当 $s_{i,t} + b_i$ 落在 top-$K_r$ 集合内；否则 $g'_{i,t}=0$。门控 $g_{i,t}$ 仍由 $s_{i,t}$ 经 top-$K$ 内归一化得到。

---

## 与传统 aux loss 对比

| 维度 | **auxiliary loss** | **aux-loss-free（+ bias）** |
|------|-------------------|------------------------------|
| 均衡信号 | 加在 **loss** 上，反传进 router | **不改 loss**；改 routing bias |
| 均衡粒度 | 常见 **sequence-wise**（每序列内均匀） | 默认 **batch-wise**（一步内整体均匀） |
| 专家 specialization | sequence 内被压平 | 允许 **不同领域** 走不同专家 |
| 超参 | aux loss 系数 $\alpha$ 难调 | $\gamma$（bias 步长） |
| 与主任务冲突 | 大 $\alpha$ 伤性能 | 论文 ablation：**多数 benchmark 更优** |

### 表：V3 论文 Table 5 摘要

**Small MoE**

| Benchmark | Aux-Loss-Based | Aux-Loss-Free |
|-----------|----------------|---------------|
| Pile-test (BPB↓) | 0.727 | **0.724** |
| BBH (EM) | 37.3 | **39.3** |
| MMLU (EM) | 51.0 | **51.8** |
| DROP (F1) | 38.1 | **39.0** |
| GSM8K (EM) | 27.1 | **29.6** |
| MATH (EM) | 10.9 | **11.1** |

**Large MoE**

| Benchmark | Aux-Loss-Based | Aux-Loss-Free |
|-----------|----------------|---------------|
| Pile-test (BPB↓) | 0.656 | **0.652** |
| BBH (EM) | 66.7 | **67.9** |
| HumanEval (Pass@1) | 40.2 | **46.3** |
| GSM8K (EM) | 70.7 | **74.5** |
| MATH (EM) | 37.2 | **39.6** |

论文结论：去掉纯 aux loss、改用 aux-loss-free 后，**大多数评测一致更好**。

---

## 补充：sequence-wise auxiliary loss 仍在

aux-loss-free **不是**完全不管极端倾斜。V3 还保留 **互补的 sequence-wise balance loss** $L_{\mathrm{Bal}}$，防止 **单条序列内** 专家极度失衡——但 **主均衡机制** 仍是 bias 方案。

> **详解**：[序列均衡损失](04-序列均衡损失.md) — $f_i$、$P_i$ 如何在每条序列上统计、$f_i P_i$ 如何反传拉平负载、与 $b_i$ 的分工。

| 机制 | 作用域 | 角色 |
|------|--------|------|
| **aux-loss-free bias** | batch-wise | **主负载均衡**（$b_i \pm \gamma$，无梯度） |
| **sequence-wise aux loss** $L_{\mathrm{Bal}}$ | **单序列** | **兜底**，$\alpha$ 极小；$\sum_i f_i P_i$ 反传 router |

论文 §4.5.3：batch-wise（含 aux-loss-free）比 sequence-wise 更灵活，专家更易 **按领域分化**；1B/3B 实验上 batch-wise aux 与 aux-loss-free 的 val loss 可打平。sequence-wise 单独作主均衡时会 **压平序列内 specialization**（val loss 略差）。

---

## 推理时是什么

- $b_i$ 在训练结束后 **固定**（推理不再按 step 更新）。
- 路由仍按 **训练结束时的 $s+b$** 选 expert；与训练一致。
- 与 **Index Share / DSA** 等 **无关**——纯 MoE router 训练策略。

---

## 工程对应

| 概念 | Megatron 配置 |
|------|----------------|
| aux-loss-free | `--moe-router-enable-expert-bias` |
| bias 步长 $\gamma$ | `--moe-router-bias-update-rate`（如 `1e-3`） |
| 传统 aux loss | `--moe-router-load-balancing-type aux_loss` / `seq_aux_loss` |

DeepSeek-V3 细粒度 MoE + sigmoid router 在 Megatron[侧常配合 **FlexDispatcher + DeepEP** 做 token dispatch](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)。

---

## MoE 线位置

| 方向 | 文档 |
|------|------|
| **本节点（③ aux-loss-free）** | [MoE 线导读 §1](../01-总览/07-MoE线导读.md#1-演进链ffn--路由) |
| **上游 ② DeepSeekMoE** | [DeepSeekMoE](05-DeepSeekMoE.md) |
| **并列 ④ $L_{\mathrm{Bal}}$** | [序列均衡损失](04-序列均衡损失.md) |
| **下游 ⑤ Hash MoE** | [Hash MoE + FP4](../04-版本代际/06-Hash-MoE-FP4.md) · [DeepSeek-V4](../04-版本代际/03-V4.md) |

---

## 在 DeepSeek 系列中的位置

| 版本 | MoE 路由 |
|------|----------|
| **V3 / R1 / V3.1 / V3.2** | DeepSeekMoE + **aux-loss-free**（256 routed + shared，top-8） |
| **V4** | 论文仍写 auxiliary-loss-free；另改 score 函数（如 Sqrt(Softplus)）等 |

演进总览表中「**aux-loss-free 路由**」即指此项，不是 Engram 或 DSA 的 index。

---

## 参考

- DeepSeek-V3：[arXiv:2412.19437](https://arxiv.org/abs/2412.19437) — §2.1 Auxiliary-Loss-Free Load Balancing；Eq. 17–20 sequence-wise $L_{\mathrm{Bal}}$；§4.5.2 Table 5；§4.5.3 batch vs sequence
- 方法先例：Wang et al., 2024a（V3 论文引用）
- 本地：[序列均衡损失](04-序列均衡损失.md) · [DeepSeek-V3 梗概§DeepSeekMoE](01-V3基座.md) · [Raschka V3 导读](../08-外部解读/01-Raschka要点速读.md)