# mHC

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §3.7](../01-总览/01-版本演进总览.md#37-deepseek-v4) · [← 算法线导读](../01-总览/05-算法线导读.md) · [← V4 梗概](03-V4.md) · [HC 基础](04b-Hyper-Connections.md) · [Raschka §8 mHC](../08-外部解读/01-Raschka要点速读.md#结论表-7--8)
> **论文**：[mHC arXiv:2512.24880](https://arxiv.org/abs/2512.24880)（2025-12-31）· **落地**：[V4 arXiv:2606.19348](https://arxiv.org/abs/2606.19348)

## 核心结论摘要

- **mHC** 在 Hyper-Connections 上加 **双随机流形约束**（Sinkhorn–Knopp）。
- 恢复 HC 恒等映射稳定性，避免多路残差流训练发散。
- V4 落地；含 §3 流形推导与 Birkhoff 多面体背景。
- 前置阅读：[Hyper-Connections](04b-Hyper-Connections.md)。

---

## 一句话

**mHC** 在 **[Hyper-Connections（HC）](04b-Hyper-Connections.md)** 的多路残差流之上，用 **Sinkhorn–Knopp** 把残差混合矩阵投影到 **双随机流形**（[Birkhoff 多面体](qa/mhc-birkhoff-polytope.md)），使层间信号混合等价于 **凸组合 / 加权平均**，从而 **恢复恒等映射的稳定性**；**DeepSeek-V4** 将其作为残差路径的标准组件。

> **答疑**：[HC（Hyper-Connections）基础](04b-Hyper-Connections.md) — $n$ 路并行残差流 + pre/post/comb 混合；裸 HC 破坏恒等映射，mHC 用双随机约束修复

---

## 1. 残差路径演进

| 阶段 | 做法 | 特点 |
|------|------|------|
| **标准残差** | $x_{l+1} = x_l + F_l(x_l)$ | 恒等路径清晰，深层可训 |
| **Hyper-Connections (HC)** | 扩宽残差流为 $n$ 条并行流，可学习混合 | 容量↑，但 **破坏恒等映射** → 训练易爆（[HC 专文](04b-Hyper-Connections.md)） |
| **mHC** | HC + **流形约束**（双随机混合矩阵） | 保留 HC 表达力，**抑制信号放大** |

[Raschka 扩展阅读 §8 · 表 8-1](../08-外部解读/02-Raschka全文解析.md#表-8-1-transformer-模块演进) 归纳的 Transformer 模块演进链（节选）（[要点速读](../08-外部解读/01-Raschka要点速读.md#结论表-7--8)）：

| 模块 | 演进 |
|------|------|
| Attention | GQA → [sliding window（SWA）](qa/v4-swa-sliding-window.md#行业链) → **MLA** → **DSA** → **[CSA/HCA](05-CSA-HCA混合压缩注意力.md)** |
| FFN | GeLU → SwiGLU → **MoE** |
| **残差** | 恒等残差（ResNet）→ **HC** → **mHC** |

> **扩展阅读**：[Raschka 全文 §8 — Transformer 模块演进表](../08-外部解读/02-Raschka全文解析.md#表-8-1-transformer-模块演进) · [要点速读 §7–8](../08-外部解读/01-Raschka要点速读.md#结论表-7--8)

---

## 算法线位置

| 方向 | 文档 |
|------|------|
| **本节点（④ mHC）** | [算法线导读 §1](../01-总览/05-算法线导读.md#1-演进链attention--残差) · [HC 基础](04b-Hyper-Connections.md) · [§3 双随机流形](#3-mhc-核心双随机流形约束) |
| **同代 Attention** | [CSA / HCA](05-CSA-HCA混合压缩注意力.md) · [DeepSeek-V4](03-V4.md) |
| **Attention 上游** | [MLA 低秩注意力](../02-基座架构/02-MLA低秩注意力.md) · [DSA 稀疏注意力](../05-DSA稀疏注意力/02-DSA梗概.md) |

---

## 2. Hyper-Connections基础

<a id="hc-basics"></a>

标准残差 → **$n$ 路并行残差流** + **pre / post / comb** 混合 → 子层 Attention/FFN；裸 HC **破坏恒等映射**，深层可出现 **~3000×** 信号放大。

> **专文**：[Hyper-Connections（HC）— 多路残差流 · 混合矩阵 · 不稳定原因](04b-Hyper-Connections.md)

**本节摘要**

| 要点 | 说明 |
|------|------|
| **扩展率** | 常见 **$n{=}4$**（V4 / mHC 实验） |
| **混合** | $H^{\mathrm{pre}}$ 汇聚入子层；$H^{\mathrm{post}}$、$H^{\mathrm{comb}}$ 写回 $n$ 路 |
| **动机** | 加宽残差 **信息高速公路** |
| **问题** | $\prod_l H_l^{\mathrm{res}}$ 无约束 → 训练易爆 → **mHC**（§3） |

---

## 3. mHC 核心：双随机流形约束

<a id="3-mhc-核心双随机流形约束"></a>

mHC 把 [HC](04b-Hyper-Connections.md) 的可学习混合矩阵 **投影** 到 **双随机流形**（Birkhoff 多面体 $\mathcal{B}_n$）：非负、行和列均为 1 → 对 $n$ 条残差流做 **凸组合**，抑制无约束 HC 的 **~3000×** 级信号放大，恢复深层 **近似恒等映射** 的稳定性。

### 3.1 双随机矩阵定义

<a id="31-双随机矩阵定义"></a>

设 $H \in \mathbb{R}^{n \times n}$。若同时满足：

$$
H_{ij} \ge 0 \quad \forall i,j; \qquad
\sum_{j=1}^{n} H_{ij} = 1 \;\; \forall i; \qquad
\sum_{i=1}^{n} H_{ij} = 1 \;\; \forall j
$$

则称 $H$ 为 **双随机矩阵**（doubly stochastic matrix）。

<img src="figures/mhc/mhc-doubly-stochastic-matrix.svg" alt="双随机矩阵示例 n=3：H_ij 非负，每行每列和为 1" width="920"/>

[图示详情](figures/mhc/mhc-doubly-stochastic-matrix.svg)

| 性质 | 含义 |
|------|------|
| **行随机** | 每一行是 $n$ 个非负数的 **概率分布** |
| **列随机** | 每一列之和也为 1（比「仅行随机」更严） |
| **凸组合** | 对向量 $x$，$Hx$ 的每个分量都是 $x$ 各分量的 **加权平均** |

**极端点**：**置换矩阵** $P_\pi$（每行每列恰有一个 1）是双随机矩阵；其余双随机矩阵可写成置换矩阵的 **凸组合**（Birkhoff–von Neumann 定理）。

### 3.2 Birkhoff 多面体 = 「双随机流形」

<a id="32-birkhoff-多面体"></a>

所有 $n \times n$ 双随机矩阵的集合记为 **Birkhoff 多面体** $\mathcal{B}_n$：

$$
\mathcal{B}_n = \mathrm{conv}\,\{\, P_\pi \mid \pi \in S_n \,\}
$$

即 $n$ 阶 **置换矩阵** 的 **凸包**。mHC 论文把 $\mathcal{B}_n$ 称为残差混合权重所在的 **特定流形**（manifold）；工程语境下常口语化为 **双随机流形**。

> **答疑**：[Birkhoff 多面体（置换矩阵的凸包）](qa/mhc-birkhoff-polytope.md) — 双随机矩阵全体 = $\mathrm{conv}\{P_\pi\}$；mHC 把混合矩阵投影到此集合

训练时 HC 产生 **无约束 logits** $\tilde{H}$；mHC 用 **Sinkhorn–Knopp**（§3.4）把 $\tilde{H}$ **投影** 到 $\mathcal{B}_n$ 的近似点 $H$。参数在 $\mathbb{R}^{n^2}$ 上优化，但 **前向使用的混合矩阵** 始终落在（或逼近）双随机集合——这就是 **Manifold-Constrained** 的含义。

### 3.3 为何 mHC 需要这个约束

<a id="33-为何-mhc-需要这个约束"></a>

[HC](04b-Hyper-Connections.md) 把每层残差扩成 $n$ 条并行流，用可学习矩阵 $H_l^{\mathrm{res}}$ 混合。若 **不约束** $H_l^{\mathrm{res}}$：

- 复合映射 $\prod_{l=1}^{L} H_l^{\mathrm{res}}$ 的谱范数可 **远大于 1**
- 特征范数在深层 **指数放大**
- 训练在万步级 **发散**

标准残差 $x_{l+1} = x_l + F_l(x_l)$ 有清晰 **恒等路径**；无约束 HC **破坏了** 这条保范数通道。

设 $n$ 条残差流向量堆叠为 $X \in \mathbb{R}^{n \times d}$，混合后第 $i$ 行：

$$
(X')_i = \sum_{j=1}^{n} H_{ij} X_j, \qquad H_{ij} \ge 0,\; \sum_j H_{ij} = 1
$$

即 $(X')_i$ 是 $\{X_j\}$ 的 **凸组合**。由 Jensen / 三角不等式：

$$
\|(X')_i\| \le \sum_j H_{ij} \|X_j\| \le \max_j \|X_j\|
$$

**单层混合不会把范数放大到超过输入流的最大范数**；多层复合时，信号仍被 **凸组合** 反复「平均」，近似 **保范数恒等映射**。

| 混合矩阵类型 | 对范数的影响 | 深层复合 |
|--------------|--------------|----------|
| 无约束 $H$ | 可任意缩放 / 旋转 | 易指数放大 |
| **双随机** $H$ | 凸组合，不增最大流范数 | 近似稳定 |

### 3.4 Sinkhorn–Knopp 投影

<a id="34-sinkhornknoop-投影"></a>

给定非负矩阵 $K$（通常由 logits 经 $\exp$ 得到），交替 **行归一化** 与 **列归一化**：

$$
K \leftarrow \mathrm{diag}(r)^{-1} K \mathrm{diag}(c)^{-1}
$$

迭代若干步（vLLM 中 `sinkhorn_repeat`；数值稳定项 `hc_sinkhorn_eps`）后，$K$ 逼近双随机 $H \in \mathcal{B}_n$。

mHC 对 **三组** 混合矩阵分别投影：

| 矩阵 | 作用 |
|------|------|
| $H^{\mathrm{pre}}$ | $n$ 条残差流 → 子层输入（流间汇聚） |
| $H^{\mathrm{post}}$ | 子层输出写回主流 |
| $H^{\mathrm{comb}}$ | 子层输出与残差流之间的组合混合 |

单层读写顺序见 [§4 单层数据流](#4-单层数据流推理)。

### 3.5 与标准残差、HC 的对照

<a id="35-与标准残差hc-的对照"></a>

| 路径 | 混合结构 | 恒等映射 |
|------|----------|----------|
| **标准残差** | 单流 $x_{l+1} = x_l + F(x_l)$ | 显式 $+x_l$ |
| **HC** | 多流 + 任意 $H_l^{\mathrm{res}}$ | **无保证** |
| **mHC** | 多流 + $H_l^{\mathrm{res}} \in \mathcal{B}_n$ | **凸组合 ≈ 加权恒等** |

mHC **保留** HC 的多流表达力（$n$ 条并行高速公路），**只** 把混合权重限制在 Birkhoff 多面体上——这是「Manifold-Constrained」相对裸 HC 的 **最小必要约束**。

| 符号（直觉） | 作用 |
|--------------|------|
| **$H^{\mathrm{pre}}$** | 多条残差流 → **子层输入**（流间加权汇聚） |
| **$H^{\mathrm{post}}$** | 子层输出 → 写回某条主流 |
| **$H^{\mathrm{comb}}$** | 子层输出与残差流之间的 **组合混合** |

---

## 4. 单层数据流

<a id="4-单层数据流推理"></a>

与 vLLM `MHCPreOp` / `MHCPostOp` 对齐的直觉：

**Pre（进子层前）**

1. 对 $n$ 条残差流做 RMSNorm（及可选 scale/bias）。
2. 由归一化后的流 + 子层分支特征算 **mix logits**。
3. Sinkhorn 得到 **pre_mix**；**layer_input** $= \sum_i \mathrm{pre\_mix}_i \cdot \mathrm{residual}_i$。

**Post（出子层后）**

$$\mathrm{out}_j = \mathrm{post\_mix}_j \cdot x + \sum_i \mathrm{comb\_mix}_{ij} \cdot \mathrm{residual}_i$$

即：子层输出与各路残差按 **post / comb** 双随机权重写回 $n$ 条流。

> **infra 注**：mHC 额外引入 **多流读写** 与 Sinkhorn 迭代；论文通过 **算子融合** 控制开销（见 §5）。

---

## 5. 训练与系统工程

mHC 论文除算法外强调 **大规模可训可部署**：

| 手段 | 目的 |
|------|------|
| **TileLang / 自定义 kernel** | Sinkhorn 迭代与 pre/post 融合，减少 HBM 往返 |
| **选择性重计算** | 不存巨大中间激活，换算力 |
| **DualPipe 通信重叠** | 掩盖流水线 stage 延迟 |
| **重排 Norm 与混合顺序** | 降低内存带宽 |

论文报告：扩展率 $n{=}4$ 时，相对标准 Transformer 训练时间开销约 **6.7%**（基础设施优化后）。

---

## 6. 在 DeepSeek-V4 中的应用

| 维度 | 说明 |
|------|------|
| **角色** | V4 **残差路径**组件，与 CSA/HCA、Hash MoE、Muon 等 **同期引入** |
| **相对 V3.2** | V3.2 **无 mHC**；mHC 独立论文（2512.24880）先发表，后在 **V4（2606.19348）** 与百万 token 架构一并落地 |
| **Ablation** | V4 多变量同改，**难以单独剥离 mHC 贡献**；Index Share 类纯 infra 补丁更易做对照 |
| **推理** | Engram 等演示代码将 Attention/MoE/**mHC** 作为标准块 mock；生产引擎（如 vLLM）提供 `mhc_pre` / `mhc_post` 自定义算子 |

详见 [DeepSeek-V4](03-V4.md) 与 [演进总览 §3.7](../01-总览/01-版本演进总览.md#37-deepseek-v4)。

---

## 7. 与 attention / MoE 线的关系

| 线 | mHC 是否涉及 |
|----|-------------|
| **MLA / DSA / CSA** | **否** — 改的是 **注意力与 KV**，不是残差拓扑 |
| **MoE 路由** | **正交** — mHC 在子层 **前后** 混合残差流，不替代 expert 选择 |
| **ESS / Index Share** | **正交** — 后者是 V3.2 **KV / indexer** infra |

mHC 回答的是：**把模型做深、做宽残差流时如何不失稳**，而非长上下文 cache 或稀疏注意力本身。

---

## 8. 上下游

| 方向 | 文档 |
|------|------|
| 前置概念 | [HC 基础](04b-Hyper-Connections.md) · [§3 双随机流形](#3-mhc-核心双随机流形约束) · [Raschka 全文 §8](../08-外部解读/02-Raschka全文解析.md#8-appendix-mhcmanifold-constrained-hyper-connections) |
| 同代 V4 组件 | [CSA / HCA](05-CSA-HCA混合压缩注意力.md) · [Muon](07-Muon优化器.md) · [DeepSeek-V4](03-V4.md)（Hash MoE） |
| 对比（无 mHC 的上一档） | [DeepSeek-V3.2](02-V3.2-DSA.md) |

---

## 参考

1. Xie et al. *mHC: Manifold-Constrained Hyper-Connections.* arXiv:2512.24880, 2025.
2. DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence.* arXiv:2606.19348, 2026.
3. Raschka. *From DeepSeek V3 to V3.2* — [§8 mHC 附录](../08-外部解读/02-Raschka全文解析.md#8-appendix-mhcmanifold-constrained-hyper-connections).
4. Birkhoff (1946) — 双随机矩阵与置换矩阵凸包（Birkhoff–von Neumann 定理）。
5. Sinkhorn & Knopp (1967) — 行列交替归一化算法。
6. 实现参考：vLLM `model_executor/layers/mhc.py`（`MHCPreOp` / `MHCPostOp` + TileLang Sinkhorn kernel）.