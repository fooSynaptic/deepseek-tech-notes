# CSA / HCA 混合压缩注意力

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §3.7](../01-总览/01-版本演进总览.md#37-deepseek-v4) · [← 算法线导读](../01-总览/05-算法线导读.md) · [← 基础设施线导读](../01-总览/06-基础设施线导读.md) · [V4 梗概](03-V4.md) · [上游 DSA](../05-DSA稀疏注意力/02-DSA梗概.md) · [上游 MLA](../02-基座架构/02-MLA低秩注意力.md) · [并列 mHC](04-mHC流形约束超连接.md) · [Lightning Indexer 详解](../05-DSA稀疏注意力/04-Lightning-Indexer详解.md)
> **论文**：[DeepSeek-V4 arXiv:2606.19348](https://arxiv.org/abs/2606.19348)

## 核心结论摘要

- **CSA** 4:1 稀疏压缩 + top-k；**HCA** 128:1 dense 压缩。
- V4 算法线核心：混合两种压缩注意力应对 1M context。
- 配合异构 KV layout、SWA、Indexer、Tail buffer。
- 上游继承 DSA indexer 思想，下游衔接 V4 KV / HiSparse。

---

## 一句话

**CSA**（Compressed Sparse Attention）与 **HCA**（Heavily Compressed Attention）是 V4 的 **双档 KV 压缩注意力**：先把历史 token **按块压成更短的 KV 序列**，再在压缩序列上分别做 **稀疏 top-$k$**（CSA，$m{=}4$）与 **dense attend**（HCA，$m'{=}128$）；二者与 [SWA](qa/v4-swa-sliding-window.md)、[Indexer KV](qa/v4-indexer-kv.md)、[tail buffer](qa/v4-tail-buffer.md) 共同构成 **百万 token** 的算法侧主因。

> **答疑**：[SWA（滑动窗口精确局部）](qa/v4-swa-sliding-window.md) — 最近窗口内未压缩 K/V，进 State 池，eviction 与 prefix 策略独立于 CSA/HCA
> **答疑**：[Indexer KV](qa/v4-indexer-kv.md) — CSA 路径上对全长历史廉价打分，再 top-$k$ 选 C4 entry
> **答疑**：[Tail buffer](qa/v4-tail-buffer.md) — 未凑满 4/128 块的 token 尾，凑满后才压入 Classical 池

---

## 算法线位置

| 方向 | 文档 |
|------|------|
| **本节点（③ CSA / HCA）** | [算法线导读 §1](../01-总览/05-算法线导读.md#1-演进链attention--残差) |
| **上游 ② DSA** | [DSA 稀疏注意力](../05-DSA稀疏注意力/02-DSA梗概.md) · [DSA 逻辑详解](../05-DSA稀疏注意力/03-DSA逻辑详解.md) |
| **上游 ① MLA** | [MLA 低秩注意力](../02-基座架构/02-MLA低秩注意力.md)（V4 不再单一 per-token latent） |
| **同代 V4 其他组件** | [DeepSeek-V4](03-V4.md)（两个规格、MoE） · [Muon](07-Muon优化器.md) · [mHC](04-mHC流形约束超连接.md)（残差路径，与 Attention **正交**） |
| **infra 依赖** | [KV layout](../06-推理基础设施/05-V4-KV-Layout.md) · [HiSparse](../06-推理基础设施/06-V4-HiSparse.md) · [磁盘 prefix](../06-推理基础设施/07-V4-磁盘Prefix-Cache.md) |

---

## 1. 两个机制对照

| 维度 | **CSA** | **HCA** |
|------|---------|---------|
| 全称 | **C**ompressed **S**parse **A**ttention | **H**eavily **C**ompressed **A**ttention |
| 压缩比 | 每 **$m{=}4$** token → **1** 条 KV entry | 每 **$m'{=}128$** token → **1** 条 entry |
| 压缩后序列长 | $\approx L/4$ | $\approx L/128$（极短） |
| 注意力模式 | 对压缩 entry 做 **DSA 式 top-$k$** 稀疏读 | 序列已足够短 → **dense attention** |
| 典型角色 | **远距、高选择性** 历史（indexer 挑块） | **全局粗粒度** 摘要（全 entry 参与） |
| Cache 别名 | **C4** 压缩层（infra 文档常用） | HCA 128:1 entry |
| 1M context 量级 | ~250K 条 C4 entry（再经 top-$k$ 只读子集） | ~8K 条 HCA entry |

> **读法**：CSA 延续 DSA「**先选再看**」；HCA 把「看」的成本压到足够低，直接 **全 attend** 仍可行。

---

## 2. CSA：块压缩 + 稀疏注意力

<a id="csa-compressed-sparse"></a>

### 2.1 流程

1. Prefill / decode 过程中，每凑满 **4 个 token** 的块，经可学习压缩算子合成 **1 条 CSA KV entry**（不再 per-token 缓存全长 latent）。
2. 当前 query 经 **lightning indexer**（思想延续 [DSA](../05-DSA稀疏注意力/02-DSA梗概.md) / [Lightning Indexer](../05-DSA稀疏注意力/04-Lightning-Indexer详解.md)）在 **$\sim L/4$ 条压缩 entry** 上打分。
3. 取 **top-$k$** 条 entry 做 attention；复杂度相对全长 MLA 的 $O(L^2)$ 降为 **$O(Lk)$** 量级（$k$ 为压缩 entry 数，$k \ll L/4$）。

### 2.2 相对 DSA 的变化

| 维度 | **DSA（V3.2）** | **CSA（V4）** |
|------|----------------|---------------|
| KV 粒度 | **每 token** 一条 MLA latent | **每 4 token** 一条压缩 entry |
| Indexer 作用对象 | 全长 latent 序列 | **更短的压缩序列** |
| MLA 结构 | Core MLA **不变** | V4 **新注意力栈**，非 V3.2 MLA 直扩 |

DSA 的「indexer + top-$k$ + 主 attention」范式 **延续**；V4 在 indexer 之前增加 **固定 stride 块压缩**，进一步压 cache 体积与 indexer 扫描长度。

---

## 3. HCA：重压缩 + 短序列 dense

<a id="hca-heavily-compressed"></a>

### 3.1 流程

1. 每 **128 token** 合成 **1 条 HCA KV entry**（压缩比远高于 CSA）。
2. 序列长度 $\approx L/128$：1M context 仅 **~8K entry**，对当前 query 做 **标准 dense attention**（无需 top-$k$）。
3. 提供 **全局、低分辨率** 的历史摘要，与 CSA 的 **局部高精度稀疏块** 互补。

### 3.2 为何不用稀疏

HCA 的设计前提是：128:1 压缩后 entry 数 **足够少**，dense matmul 的 FLOPs 与带宽仍低于「对 $L/4$ 条 CSA entry 做 sparse 路径 + indexer」；相当于用 **更粗粒度** 换 **更简单的读模式**。

---

## 4. V4 内如何「混合」

<a id="v4-mixed-attention"></a>

V4 **不是**「全层 CSA 或全层 HCA」二选一，而是在同一模型中 **组合多种 attention 路径**：

| 对象 | 与 CSA/HCA 关系 |
|------|-----------------|
| **CSA C4 entry** | 远距稀疏主路径；HiSparse offload 的 **inactive 块** 主要指此层 |
| **HCA entry** | 全局 dense 摘要路径 |
| **SWA** | 滑动窗口 **精确局部** state；与压缩 entry **独立 eviction** |
| **Indexer KV** | CSA 路径的 lightning indexer 向量；维度与主 entry 不同 |
| **Tail buffer** | 不足 4（CSA）或 128（HCA）token 的 **未压缩尾**；凑满后再入 Classical 池 |

<img src="../01-总览/figures/v4/v4-hetero-kv.svg" alt="DeepSeek-V4 异构 KV：CSA 4:1、HCA 128:1、SWA、Indexer、Tail buffer 与 HiSparse offload" width="920"/>

[图示详情](../01-总览/figures/v4/v4-hetero-kv.svg) · [演进总览 §3.7](../01-总览/01-版本演进总览.md#37-deepseek-v4)

**内存布局**→ **[V4 KV Layout 专文](../06-推理基础设施/05-V4-KV-Layout.md)**，算法对象与引擎池化的分界见该文 §1。

---

## 5. 相对 V3.2 的效率

V4 把 **算力** 与 **KV 体积** 同时压下来，CSA/HCA 是算法侧主因；infra 侧另需 [HiSparse](../06-推理基础设施/06-V4-HiSparse.md) / [磁盘 prefix](../06-推理基础设施/07-V4-磁盘Prefix-Cache.md) 才能「跑得动」1M。

| 模型 | 单 token FLOPs @ 1M（相对 V3.2） | 累计 KV cache @ 1M |
|------|----------------------------------|-------------------|
| V4-Pro | 27% | 10% |
| V4-Flash | 10% | 7% |

详见 [DeepSeek-V4 梗概§1M context 效率](03-V4.md#1m-context-效率相对-v32)。

---

## 6. 训练侧：渐进式上下文

V4 训练采用 **渐进拉长上下文**（[DeepSeek-V4 梗概§训练要点](03-V4.md#训练要点)）：

- 4K **dense** 基座 → 16K → 64K **引入稀疏** → 1M
- CSA/HCA 与稀疏 indexer 在 **中长上下文阶段** 才全面启用；短上下文阶段可近似 dense 行为，便于稳定收敛。

---

## 7. 与 infra 线的关系

| infra 专题 | 依赖 CSA/HCA 的方式 |
|------------|---------------------|
| [KV layout](../06-推理基础设施/05-V4-KV-Layout.md) | Classical 池存 CSA/HCA 压缩块；State 池存 tail + SWA |
| [HiSparse](../06-推理基础设施/06-V4-HiSparse.md) | 针对 **C4 inactive entry** 的 CPU offload |
| [磁盘 prefix cache](../06-推理基础设施/07-V4-磁盘Prefix-Cache.md) | CSA/HCA 压缩 entry **可直接落盘**；SWA 三档策略另计 |
| [ESS](../06-推理基础设施/01-ESS概念.md) | **不可直迁** — ESS 面向 V3.2 per-token latent |

完整 **基础设施线** 见 [基础设施线导读](../01-总览/06-基础设施线导读.md#1-演进链kv--offload)。

---

## 8. 演进链小结

<img src="figures/csa-hca-evolution-chain.svg" alt="MLA → DSA → CSA → HCA + V4 异构 KV 演进链" width="920"/>

[图示详情](figures/csa-hca-evolution-chain.svg)

| 边 | 关系 |
|----|------|
| MLA → DSA | MLA **结构不变**；在 latent 序列上加稀疏选择 |
| DSA → CSA | 「先选再看」**延续**；先 **块压缩** 再 indexer |
| CSA ⊥ HCA | **互补档位**：4:1 稀疏精细 vs 128:1 dense 粗摘要 |
| CSA/HCA ⊥ mHC | 前者改 **Attention / KV**；mHC 改 **残差拓扑** |

---

## 9. 上下游

| 方向 | 文档 |
|------|------|
| 版本总览 | [DeepSeek-V4](03-V4.md)（两个规格、MoE、训练要点） · [Muon](07-Muon优化器.md) |
| 上游 | [DeepSeek-V3.2](02-V3.2-DSA.md) · [DSA 稀疏注意力](../05-DSA稀疏注意力/02-DSA梗概.md) |
| 并行（V3.2 infra） | [Index Share 梗概](../05-DSA稀疏注意力/05-Index-Share梗概.md) — 纯 infra，与 V4 路线 **互补** |
| 同代残差 | [mHC](04-mHC流形约束超连接.md) |
| 部署解读 | [Together.ai — Serving DeepSeek-V4](https://www.together.ai/blog/serving-deepseek-v4-why-million-token-context-is-an-inference-systems-problem) |

---

## 参考

- 论文：[arXiv:2606.19348](https://arxiv.org/abs/2606.19348)
- HuggingFace：[deepseek-v4 collection](https://huggingface.co/collections/deepseek-ai/deepseek-v4)