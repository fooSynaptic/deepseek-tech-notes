# ESS：Latent-Cache Offload

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §5.2](../01-总览/01-版本演进总览.md#52-v32indexer-cache--latent-cache-分离--ess) · [← 基础设施线导读](../01-总览/06-基础设施线导读.md) · [← DSA](../05-DSA稀疏注意力/02-DSA梗概.md) · [并列 Index Share](../05-DSA稀疏注意力/05-Index-Share梗概.md) · [ESS 论文梗概](02-ESS论文梗概.md)
> **论文**：[ESS arXiv:2512.10576](https://arxiv.org/abs/2512.10576) — *An Offload-Centric Latent-Cache Management Architecture for DeepSeek-V3.2-Exp*（Chen et al., 2025）
> **论文梗概**：[ESS 论文梗概](02-ESS论文梗概.md)

## 核心结论摘要

- **ESS** 将 MLA **Latent-Cache** offload 到 CPU，扩展有效 KV 容量。
- 百度百舸针对 DeepSeek-V3.2 的 infra 方案，**与 DSA 算法正交**。
- 与 V4 异构 KV / HiSparse 是不同代际的 offload 设计。
- 论文梗概见 [ess-paper-highlights](02-ESS论文梗概.md)。

---

## 一句话

**ESS** 是面向 **V3.2 DSA** 的 **KV offload 架构**：**Indexer-Cache**（~16.8%）**常驻 GPU**；**Latent-Cache**（~83.2%）**卸载到 CPU DRAM**，按 top-$k$ 稀疏访问 **prefetch 热 entry 回 GPU**。不改模型权重，依赖 DSA 把主 attention 限制在 $k{=}2048$ 个 latent 上。

**归属**：ESS 为 **百度百舸（Baige AI）** 提出（[arXiv:2512.10576](https://arxiv.org/abs/2512.10576)），优化 **Latent-Cache 显存**；与 **清华 + 智谱** 的 [Index Share / IndexCache](../05-DSA稀疏注意力/05-Index-Share梗概.md)（优化 **indexer 计算**）**正交**，可同开。

---

## 为什么 V3.2 需要 ESS

| 问题 | 说明 |
|------|------|
| **128K latent 线性涨** | 即使用 MLA 压缩，全长 Latent-Cache 仍占 HBM 大头 |
| **V3 通用 offload 不好用** | 同质 MLA latent 整条搬移，PCIe 小块传输带宽差 |
| **DSA 带来的机会** | Core MLA **每步只读 $k$ 个** latent entry → offload 粒度可变成 **稀疏 entry**，而非全长序列 |

DSA 先把 cache **[拆成 Indexer + Latent 两类](../05-DSA稀疏注意力/02-DSA梗概.md)**，ESS 专门管 **Latent 那一侧的 CPU/GPU 分层**。

---

## 双 Cache 与 ESS 分工

<img src="../05-DSA稀疏注意力/figures/ess-dual-cache.svg" alt="MLA Decode 一步: Indexer 选 top-2048 位置 vs Latent-Cache 升维并做 Core MLA" width="920"/>

[图示详情](../05-DSA稀疏注意力/figures/ess-dual-cache.svg) · 图源 [Lightning Indexer · Decode 一步](../05-DSA稀疏注意力/04-Lightning-Indexer详解.md#decode-一步分工)

**读图要点（单层、第 $t$ 个 decode token）**

| 组件 | 存什么 | 算什么 | 不算什么 |
|------|--------|--------|----------|
| **Indexer-Cache** | 全长 $L$ 个 **轻量 indexer 向量**（~16.8%，GPU 常驻） | 对 $j{=}1..L$ 打分 → **Top-2048 下标 $I$** | 不读 $c_j^{KV}$、不做 MLA softmax |
| **Latent-Cache** | 每位置 **$c_j^{KV}$ [512]** + $k^R$（~83.2%，ESS 可 offload） | **仅 $j \in I$**：prefetch → $W^{UK}/W^{UV}$ 升维 → **Core MLA** → $u_t$ | 不对全长 $L$ 做稠密 attention |

Indexer 回答「**看哪 2048 个位置**」；Latent-Cache 回答「**这些位置的 MLA K/V 是多少、怎么加权**」。详见 [MLA 前向流程图](../02-基座架构/02-MLA低秩注意力.md#forward-flow)。

| Cache | 占比 | ESS 策略 | 原因 |
|-------|------|----------|------|
| **Indexer-Cache** | ~16.8% | **GPU 常驻，不 offload** | 每 decode step 要对全长跑 indexer |
| **Latent-Cache** | ~83.2% | **CPU offload + GPU LRU 热池** | 主 attention 只 touch top-$k$；相邻 step index **重叠率高** |

DSA 异构 cache 同时支撑 **ESS（搬 latent）** 与 **[Index Share](../05-DSA稀疏注意力/05-Index-Share梗概.md)（省 indexer 计算）**。

---

## ESS 核心机制

| 组件 | 作用 |
|------|------|
| **Latent-Cache → CPU** | 冷 latent entry 放主机 DRAM，释放 GPU HBM |
| **Sparse Memory Pool（GPU）** | LRU 维护 **热** latent 子集；miss 时从 CPU **prefetch** |
| **FlashTrans + UVA** | 优化大量 **656B 级小块** PCIe 传输 由 ~0.8 GB/s 提升至 ~37 GB/s） |
| **Layer-wise overlap** | 计算与传输 **流水线**，掩盖 prefetch 延迟 |

**局部性依据**：DSA 每步选出的 top-$k$ index 集合在相邻 decode step 间 **高度相似** → 多数需要的 latent 已在 GPU 热池，少量 miss 再拉取。

---

## 与 Index Share / V4 的关系

| | **ESS** | **Index Share** | **V4 HiSparse** |
|--|---------|-----------------|-----------------|
| 改权重 | 否 | 否 | 是（新模型） |
| 省什么 | **Latent** 显存（offload） | **Indexer** 算力（跨层复用 index） | 异构压缩 cache + inactive entry offload |
| 适用 | V3.2 DSA | V3.2 / GLM-5 DSA | V4 CSA/HCA |
| 叠加 | 与 Index Share **正交可同开** | 与 ESS **正交** | **非** V3.2 ESS 的简单放大 |

> V4 的 KV-offload 围绕 **[CSA/HCA/SWA 异构 layout](../04-版本代际/05-CSA-HCA混合压缩注意力.md)** 重做，不是把 ESS 直接扩到 1M · [DeepSeek-V4](../04-版本代际/03-V4.md)。

---

## 论文收益

详见 **[论文梗概 §Table 2](02-ESS论文梗概.md#table-2-吞吐与-otps核心结果)**。

| 上下文 | 吞吐提升 |
|--------|------------------|
| 32K | +69.4%（MTP=2，batch 52→160，Ratio 1.0→0.21） |
| 128K | 最高 +123%（MTP=2，batch 13→54，Ratio 1.0→0.1） |

---

## 基础设施线位置

| 方向 | 文档 |
|------|------|
| **本节点（④ ESS offload）** | [基础设施线导读 §1](../01-总览/06-基础设施线导读.md#1-演进链kv--offload) |
| **前置 ② 异构 cache** | [DSA稀疏注意力§异构 KV](../05-DSA稀疏注意力/02-DSA梗概.md#异构-kv-cache) |
| **并列 ③ Index Share** | [Index Share 梗概](../05-DSA稀疏注意力/05-Index-Share梗概.md)（indexer 算力，可同开） |
| **下游 ⑤ V4 infra** | [DeepSeek-V4 梗概§推理 infra](../04-版本代际/03-V4.md#推理-infra-关注点) · [KV layout](05-V4-KV-Layout.md) · [HiSparse](06-V4-HiSparse.md) · [磁盘 prefix](07-V4-磁盘Prefix-Cache.md)（**非** ESS 简单放大） |

---

## 在版本线中的位置

<img src="../05-DSA稀疏注意力/figures/ess-kv-lineage-tree.svg" alt="KV-offload 演进：V3 同质 MLA → V3.2 双 Cache → ESS / Index Share → V4" width="640"/>

[图示详情](../05-DSA稀疏注意力/figures/ess-kv-lineage-tree.svg)

**前置**：[DSA](../05-DSA稀疏注意力/02-DSA梗概.md)（必须先有双 cache 结构）
**并列**：[Index Share](../05-DSA稀疏注意力/05-Index-Share梗概.md)

---

## 延伸

| 资源 | 说明 |
|------|------|
| **[ESS 论文梗概](02-ESS论文梗概.md)** | **论文梗概**：Fig.1–9、Table 1–2 逐图逐表解读 |
| [DSA逻辑详解](../05-DSA稀疏注意力/03-DSA逻辑详解.md) §4 | 异构 Cache 设计含义 |
| [Index Share逻辑详解](../05-DSA稀疏注意力/06-Index-Share逻辑.md) | 与 ESS 正交性 |
| [DeepSeek-V3.2](../04-版本代际/02-V3.2-DSA.md) | V3.2 梗概 |
| [演进总览 §5.4](../01-总览/01-版本演进总览.md#54-三代-offload-对比) | V3 / V3.2 ESS / V4 三代 offload 对比 |

**论文**：[arXiv:2512.10576](https://arxiv.org/abs/2512.10576)