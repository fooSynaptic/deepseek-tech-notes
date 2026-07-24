# Hyper-Connections

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← mHC 主文档](04-mHC流形约束超连接.md) · [← 算法线导读](../01-总览/05-算法线导读.md) · [mHC §3 双随机流形](04-mHC流形约束超连接.md#3-mhc-核心双随机流形约束) · [V4 梗概](03-V4.md) · [Raschka §8](../08-外部解读/01-Raschka要点速读.md#结论表-7--8)
> **论文**：[mHC arXiv:2512.24880](https://arxiv.org/abs/2512.24880) §2（HC 为 mHC 前置）· **落地**：[V4 arXiv:2606.19348](https://arxiv.org/abs/2606.19348)

## 核心结论摘要

- **HC**：$n$ 路并行残差流 + pre/post/comb 变换。
- 比单路残差更强表达，但训练稳定性挑战更大。
- V4 **mHC** 在其上加双随机流形约束后落地。
- 含公式与与标准残差的对照说明。

---

## 一句话

**Hyper-Connections（HC）** 把 Transformer 每层 **单条残差流** 扩成 **$n$ 条并行残差流**，用可学习矩阵 **pre / post / comb** 在流与子层（Attention / FFN）之间混合；动机是 **加宽残差信息高速公路**，但无约束时 **破坏恒等映射** → 深层信号可 **指数放大**，需 [mHC](04-mHC流形约束超连接.md) 的 **双随机流形** 约束后才可大规模训练。

---

## 算法线位置

| 方向 | 文档 |
|------|------|
| **本文（HC 基础）** | mHC 节点 **前置概念**；算法线 ④ 的上游 |
| **下游（约束版）** | [mHC](04-mHC流形约束超连接.md)（含 [§3 双随机流形](04-mHC流形约束超连接.md#3-mhc-核心双随机流形约束)） |
| **正交模块** | [CSA/HCA](05-CSA-HCA混合压缩注意力.md)（Attention / KV）· [MoE 线](../01-总览/07-MoE线导读.md)（FFN 路由） |

---

## 1. 标准残差 vs HC

### 1.1 标准 Transformer 残差

一层子模块（Attention 或 FFN）的典型写法：

$$
x_{l+1} = x_l + F_l(x_l)
$$

| 性质 | 含义 |
|------|------|
| **单流** | 隐状态 $x_l \in \mathbb{R}^d$ 只有 **一条** 载体 |
| **恒等路径** | 即使 $F_l \approx 0$，仍有 $x_{l+1} \approx x_l$ |
| **深度可训** | 梯度可沿 $+x_l$ **直通** 浅层 |

### 1.2 HC 改了什么

HC **不替换** Attention / FFN 内部算子，而是改 **残差拓扑**：

- 把每层隐状态扩成 **$n$ 条并行残差流**（扩展率 **$n$**，mHC / V4 常见 **$n{=}4$**）；
- 子层 **输入** 由 $n$ 条流 **加权汇聚** 得到；
- 子层 **输出** 再 **写回** $n$ 条流（可与原流 **组合混合**）。

直觉：**$n$ 条并行「信息高速公路」**，层间连接比单流残差 **更丰富**。

<img src="figures/mhc/hyper-connections.svg" alt="HC 残差拓扑：单流 vs n 路并行流 + H^pre / F_l / H^post+H^comb" width="920"/>

[图示详情](figures/mhc/hyper-connections.svg)

---

## 2. 多路残差流

记第 $l$ 层、第 $t$ 个 token 处 $n$ 条残差流为：

$$
\mathbf{r}_l = (r_{l,1}, \ldots, r_{l,n}), \quad r_{l,i} \in \mathbb{R}^d
$$

| 符号 | 含义 |
|------|------|
| **$n$** | 扩展率（parallel stream count） |
| **$r_{l,i}$** | 第 $i$ 条残差流上的隐状态切片（实现上常为 $d$ 维向量或 $d/n$ 维分片，依实现而定） |
| **初始化** | 通常由 embedding 或上一层输出 **复制 / 投影** 到 $n$ 路 |

相对单流 $x_l$，HC 在 **宽度** 上多了 $n$ 路，参数与激活的 **读写模式** 从「一条链」变成「$n$ 路网状混合」。

---

## 3. 三组混合矩阵

mHC 论文对每层子模块（Attention 或 FFN）使用 **三组** 可学习混合（裸 HC 为 **无约束实矩阵**；mHC 再投影到双随机流形）：

| 矩阵 | 形状（直觉） | 作用 |
|------|--------------|------|
| **$H^{\mathrm{pre}}$** | $1 \times n$（或列向量） | $n$ 条残差流 → **汇聚成子层输入** $\tilde{x}_l = \sum_i H^{\mathrm{pre}}_i \, r_{l,i}$ |
| **$H^{\mathrm{post}}$** | $n \times 1$ | 子层输出 $F_l(\tilde{x}_l)$ → **写入某条主流** 的权重 |
| **$H^{\mathrm{comb}}$** | $n \times n$ | 子层输出与 **各路残差** 的 **组合混合**（写回 $n$ 条流） |

另有层间或块级 **$H_l^{\mathrm{res}}$**（$n \times n$）在 **相邻层 / 子层之间** 混合 $n$ 条流；深层复合 $\prod_l H_l^{\mathrm{res}}$ 是 **不稳定性的主要来源**（见 §5）。

---

## 4. 单层数据流

与 [mHC 主文档 §4](04-mHC流形约束超连接.md#4-单层数据流推理) 及 vLLM `MHCPreOp` / `MHCPostOp` 对齐：

**Pre（进 Attention / FFN 前）**

1. 对 $n$ 条残差流做 Norm（如 RMSNorm）。
2. 由归一化后的流（及可选分支特征）算 **mix logits** → 得 **$H^{\mathrm{pre}}$**。
3. 子层输入：$\tilde{x}_l = \sum_i H^{\mathrm{pre}}_i \cdot r_{l,i}$。

**子层**

$$
y_l = F_l(\tilde{x}_l) \quad (F_l = \text{Attention 或 FFN/MoE})
$$

**Post（出子层后）**

$$
r_{l+1,j} = H^{\mathrm{post}}_j \cdot y_l + \sum_i H^{\mathrm{comb}}_{ji} \cdot r_{l,i}
$$

即：子层输出与各路残差按 **post / comb** 权重写回第 $l{+}1$ 层的 $n$ 条流。

| 阶段 | 单流残差 | HC |
|------|----------|-----|
| 输入 | $x_l$ 直连 $F_l$ | $n$ 路 **先混合** 再进 $F_l$ |
| 输出 | $x_l + F_l(x_l)$ | $F_l$ 输出 **分流写回** $n$ 路 |
| 恒等性 | $F_l{=}0 \Rightarrow x_{l+1}{=}x_l$ | 一般 **不成立**（混合矩阵任意） |

---

## 5. 为何裸 HC 不稳定

| 问题 | 机制 |
|------|------|
| **恒等映射被破坏** | 即使 $F_l \approx 0$，$H^{\mathrm{pre/post/comb}}$ 仍可对 $n$ 路特征 **重分配、放大** |
| **深层复合放大** | $\prod_l H_l^{\mathrm{res}}$ 若谱半径 $>1$，范数可 **指数增长** |
| **论文观测** | 无约束 HC 可出现 **~3000×** 量级信号放大，训练 **万步级崩溃** |

标准残差把「子层增量」与「恒等捷径」 **解耦**；HC 把二者都变成 **可学习线性混合**，表达力↑，但 **缺少保范数结构**。

**mHC 的修复**：把上述混合矩阵 **投影到双随机流形** $\mathcal{B}_n$，使混合变为 **凸组合 / 加权平均** → 见 [mHC §3](04-mHC流形约束超连接.md#3-mhc-核心双随机流形约束)。

---

## 6. HC 与 mHC、V4 的关系

| | **HC** | **mHC** |
|--|--------|---------|
| 残差拓扑 | $n$ 路并行 + 可学习混合 | **同 HC 拓扑** |
| 混合矩阵 | 无约束（或弱约束） | **Sinkhorn–Knopp** → 双随机 |
| 训练 | 大规模易不稳 | V4 / 论文报告可训可部署 |
| 文档 | **本文** | [mHC](04-mHC流形约束超连接.md) |

| 版本 | HC / mHC |
|------|----------|
| V3 / V3.2 | **无** HC、mHC |
| 独立论文 2512.24880 | 提出 HC + mHC |
| **V4** | **mHC** 作为残差路径标准组件（与 CSA/HCA 等 **正交**） |

---

## 7. 与 Attention / MoE 的边界

| 模块 | HC 是否涉及 |
|------|-------------|
| **MLA / DSA / CSA / HCA** | **否** — 改 Attention 与 KV，不改残差拓扑 |
| **MoE 路由** | **否** — HC 在子层 **前后** 混合残差流；expert 选择在 FFN **内部** |
| **ESS / KV layout** | **否** — infra 管 cache，不管残差宽化 |

HC 回答：**残差路径加宽时如何连**，不回答长上下文或稀疏注意力本身。

---

## 8. 上下游

| 方向 | 文档 |
|------|------|
| **下游** | [mHC](04-mHC流形约束超连接.md)（[§3 双随机流形](04-mHC流形约束超连接.md#3-mhc-核心双随机流形约束)） |
| **同代 V4** | [DeepSeek-V4](03-V4.md) · [CSA/HCA](05-CSA-HCA混合压缩注意力.md) |
| **外部解读** | [Raschka §8](../08-外部解读/02-Raschka全文解析.md#8-appendix-mhcmanifold-constrained-hyper-connections) |

---

## 参考

1. Xie et al. *mHC: Manifold-Constrained Hyper-Connections.* arXiv:2512.24880, 2025（§2 Hyper-Connections）.
2. DeepSeek-AI. *DeepSeek-V4.* arXiv:2606.19348, 2026.
3. Raschka. *From DeepSeek V3 to V3.2* — [§8 mHC 附录](../08-外部解读/02-Raschka全文解析.md#8-appendix-mhcmanifold-constrained-hyper-connections).