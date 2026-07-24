# Hash MoE + FP4

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §3.7](../01-总览/01-版本演进总览.md#37-deepseek-v4) · [← MoE 线导读](../01-总览/07-MoE线导读.md) · [V4 梗概](03-V4.md) · [上游 DeepSeekMoE](../02-基座架构/05-DeepSeekMoE.md) · [上游 aux-loss-free](../02-基座架构/03-aux-loss-free-MoE路由.md) · [$L_{\mathrm{Bal}}$](../02-基座架构/04-序列均衡损失.md) · [centroid vs gate-weight 答疑](../02-基座架构/qa/moe-centroid-vs-gate-weight.md)
> **论文**：[DeepSeek-V4 arXiv:2606.19348](https://arxiv.org/abs/2606.19348)

## 核心结论摘要

- V4 前几层用 **Hash 路由**替代 learned gate，降低路由开销。
- Routed expert 权重 **FP4 + QAT**，进一步压推理带宽。
- MoE 线 ⑤；与 V3 aux-loss-free 形成演进对照。
- Shared expert 仍保留，Hash 仅作用于部分浅层 routed。

---

## 一句话

**Hash MoE** 是 V4 在 **MoE 线 ⑤** 对 FFN 的改动：**前几层** 把稠密 FFN 换成 **Hash-routed MoE**（token → expert 由 **确定性 hash** 决定，不再走 V3 的 centroid / sigmoid 亲和度路由）；**FP4 MoE** 则对 **routed expert 权重** 做 **FP4 量化 + QAT**，在继承 DeepSeekMoE 256/8 + shared 骨架的前提下压显存与带宽。

---

## MoE 线位置

| 方向 | 文档 |
|------|------|
| **本节点（⑤ Hash MoE + FP4）** | [§1 Hash 路由](#hash-moe-routing) · [§2 FP4 量化](#fp4-moe-quant) |
| **MoE 线 hub** | [MoE 线导读 §1](../01-总览/07-MoE线导读.md#1-演进链ffn--路由) |
| **上游 ②–④** | [DeepSeekMoE](../02-基座架构/05-DeepSeekMoE.md) · [aux-loss-free MoE 路由](../02-基座架构/03-aux-loss-free-MoE路由.md) · [序列均衡损失](../02-基座架构/04-序列均衡损失.md) |
| **版本总览** | [DeepSeek-V4](03-V4.md)（两个规格、Attention、训练） |
| **正交** | [CSA/HCA](05-CSA-HCA混合压缩注意力.md) · [mHC](04-mHC流形约束超连接.md) — 不改 expert 选择逻辑本身 |

---

## 1. Hash MoE：路由从「学出来」到「算出来」

<a id="hash-moe-routing"></a>

### 1.1 V3 及以前：centroid / affinity 路由

DeepSeekMoE（V2→V3）用 **可学习 expert 向量** $e_i$ 与 token hidden $u$ 做亲和度 $u^\top e_i$，再 sigmoid + top-$K_r$ + 动态 bias $b_i$（[aux-loss-free](../02-基座架构/03-aux-loss-free-MoE路由.md)）。语义上是 **[「token 匹配 expert 原型」](../02-基座架构/qa/moe-centroid-vs-gate-weight.md)**。

### 1.2 V4 Hash 路由

<a id="hash-moe-routing-diagram"></a>

V4 **前几层** 改为 **Hash-routed MoE**：

<img src="figures/v4/hash-moe-routing.svg" alt="V3 gate affinity uTe_i 与 V4 hash 查询对比；id 确定后共用 EP scatter、Routed FFN、Gather+shared" width="920"/>

浅层 MoE 只把路由从 **gate 亲和**（$u^\top e_i$ + sigmoid + top-$K_r$）换成 **hash 查表**（$\varphi(x_t,t)\bmod N_r$）；expert id 确定后仍走同一套 EP scatter → Routed FFN → Gather+shared。

[图示详情](figures/v4/hash-moe-routing.svg)

| 维度 | **V3 aux-loss-free** | **V4 Hash MoE（前几层）** |
|------|----------------------|---------------------------|
| Expert 选择 | $u^\top e_i$ + sigmoid + top-$K_r$ | **确定性 hash**（token / 位置 → expert id） |
| 可学习路由参数 | $e_i$、$b_i$ 等 | **无** centroid 式语义匹配 |
| 负载均衡 | $b_i$ + $L_{\mathrm{Bal}}$ | hash 设计 + 仍可有 shared / 池化均衡 |
| 动机 | 语义特化 + 均衡 | **省路由算力与参数**；浅层更偏通用变换 |

> **读法**：Hash MoE **不是**换掉整个 V4 的 MoE 栈，而是 **部分层** 离开 centroid 路由；更深层仍可在 DeepSeekMoE **256 routed / 8 active + shared** 框架内沿用 V3 族路由 为准）。

> **答疑**：[为何只改浅层、深层仍用 centroid？](qa/hash-moe-shallow-vs-deep.md) — 浅层通用变换 vs 深层语义特化；静态 hash vs 动态 $b_i$ / EP 均衡

### 1.3 仍继承什么

<a id="still-inherited"></a>

- **DeepSeekMoE 形态**：细粒度 routed pool + **shared experts** 恒激活（[DeepSeekMoE](../02-基座架构/05-DeepSeekMoE.md)）
- **MoE 并行 / EP**：routed gather-scatter 推理路径与 V3 同族（[答疑：EP 与 gather/scatter](qa/moe-expert-parallel-ep.md)）
- **与 Attention 正交**：Hash 只改 **FFN 专家选择**；[CSA/HCA](05-CSA-HCA混合压缩注意力.md) 改 Attention/KV

> **答疑**：[Expert Parallel（EP）与 gather/scatter](qa/moe-expert-parallel-ep.md) — routed 分片、scatter/gather 与 V3 同族；Hash 只改 expert id，不改 EP 骨架

---

## 2. FP4 MoE：routed expert 低比特权重

<a id="fp4-moe-quant"></a>

| 对象 | 精度 | 说明 |
|------|------|------|
| **Routed expert 权重** | **FP4** + **QAT** | 训练时量化感知，部署减 HBM / 带宽 |
| **Shared expert** | 通常高于 FP4（与 V3 FP8 训推栈衔接） | 恒激活路径对误差更敏感 |
| **Router（非 Hash 层）** | 与 V3 族一致 | sigmoid + bias 等仍可能存在于深层 |

FP4 MoE 与 V3 的 [FP8 动态量化](../02-基座架构/06-V3-FP8动态量化.md) **同族目标**（压 FFN 内存），但 **比特更 aggressive**，且 **只针对 routed expert 块**。

---

## 3. 在 V4 两个规格中的角色

| 规格 | MoE 侧关注点 |
|------|-------------|
| **V4-Pro**（1.6T / 49B act） | Hash 前几层 + FP4 routed；能力上限 |
| **V4-Flash**（284B / 13B act） | 同族机制，激活参数更小 |

MoE 改动与 CSA/HCA、mHC、Muon **同期打包**进 V4，**[难以单独 ablation](03-V4.md#定位)** Hash vs FP4 vs Attention。

---

## 4. 演进链小结

<img src="figures/hash-moe-evolution-chain.svg" alt="稠密 FFN → DeepSeekMoE → aux-loss-free → Hash MoE + FP4" width="920"/>

[图示详情](figures/hash-moe-evolution-chain.svg)

| 边 | 关系 |
|----|------|
| ③ → ⑤ | **继承** MoE 池化与 shared；**前几层** 路由机制换 Hash |
| Hash ⊥ centroid | 浅层 **不算** $u^\top e_i$；深层可仍用 V3 族路由 |
| FP4 ⊥ Hash | 量化对象（权重）与 路由函数（选 expert）**正交**，V4 同时上 |

---

## 5. 推理 infra 关注点

- **EP + FP4 kernel**：routed expert 权重 4bit 存取；需与 shared 高精度路径 **分 kernel**（[EP 答疑](qa/moe-expert-parallel-ep.md#5-v4-叠加fp4--ep)）。
- **Hash 层**：路由 **无 GEMM 打分**，但 expert id 仍走 gather/scatter；负载由 hash 函数 **静态近似均衡**。
- **Checkpoint 兼容**：V3 sigmoid-router 权重 **不可** 直接灌入 Hash 层逻辑。

---

## 6. 上下游

| 方向 | 文档 |
|------|------|
| MoE 线 hub | [MoE 线导读](../01-总览/07-MoE线导读.md) |
| 版本总览 | [DeepSeek-V4](03-V4.md) |
| 路由对照 | [MoE 路由：gate-weight 还是 expert centroid？](../02-基座架构/qa/moe-centroid-vs-gate-weight.md) §4.4 |
| FP8 前代 | [V3 FP8 动态量化](../02-基座架构/06-V3-FP8动态量化.md) |

---

## 参考

- DeepSeek-V4：[arXiv:2606.19348](https://arxiv.org/abs/2606.19348)
- HuggingFace：[deepseek-v4 collection](https://huggingface.co/collections/deepseek-ai/deepseek-v4)