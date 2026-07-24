# V4 KV Layout：Classical + State 双池

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §5.3 KV layout](../01-总览/01-版本演进总览.md#v4-kv-layout) · [← 基础设施线导读](../01-总览/06-基础设施线导读.md) · [CSA/HCA 算法专文](../04-版本代际/05-CSA-HCA混合压缩注意力.md) · [V4 梗概 §推理 infra](../04-版本代际/03-V4.md#推理-infra-关注点) · [并列 HiSparse](06-V4-HiSparse.md) · [并列 磁盘 Prefix Cache](07-V4-磁盘Prefix-Cache.md) · [上游 ESS](01-ESS概念.md)
> **论文**：[arXiv:2606.19348](https://arxiv.org/abs/2606.19348) §3.5.1 — *DeepSeek-V4*
> **演进总览 §5.3** 只保留梗概；**layout 细节以本文为准**。

## 核心结论摘要

- V4 KV 分为 **Classical + State 双池**，不再单一 MLA latent。
- 需同时管理 CSA/HCA 压缩 entry、SWA、Indexer KV、Tail buffer。
- 1M context 下 FLOPs 与累计 cache 相对稠密 attention 大幅节省。
- 推理引擎需围绕异构 cache 重新设计内存层级。

---

## 一句话

V4 推理引擎不再用「每层每 token 一条同质 latent」，而是把 **CSA/HCA 压缩 entry** 放进按 $\mathrm{lcm}(m,m')$ 对齐的 **Classical KV cache**，把 **[SWA](qa/v4-swa-sliding-window.md)**（**S**liding **W**indow **A**ttention，滑动窗口注意力）+ 未凑满压缩块的 [tail buffer](qa/v4-tail-buffer.md) 放进每请求固定大小的 **State cache**——两类池子尺寸、生命周期、读写模式不同，是后续 HiSparse offload 与磁盘 prefix 的**内存布局前提**。

> **答疑**：[SWA（Sliding Window Attention）](qa/v4-swa-sliding-window.md) — 最近约 128 token 的精确未压缩 K/V，与 CSA/HCA 分池，eviction 与 prefix 策略独立

---

## 为什么 V3/V3.2 的单一 layout 不够

| 代际 | Cache 形态 | 问题 |
|------|-----------|------|
| V3 / V3.1 | 同质 MLA latent，每层每 token 同 shape | 1M token 下 HBM 与带宽双瓶颈 |
| V3.2 ESS | Indexer-Cache + Latent-Cache **两类**，但仍是 **每 token 一条** | ESS 只 offload Latent；layout 仍是 DSA 的 per-token 流 |
| **V4** | CSA 4:1、HCA 128:1、SWA、Indexer、tail **五类对象** | 压缩比 $m{=}4$、$m'{=}128$ 不同 → 块对齐、尾缓冲、SWA 窗口必须 **分池管理**（见下） |

> **答疑**：[为何要分池？块对齐与尾缓冲](qa/v4-kv-dual-pool-alignment.md) — Classical 只存满块不可变 entry；State 存 tail + SWA；$\mathrm{lcm}(4,128){=}128$ 统一两种压缩边界

算法侧五类 entry 见 [CSA/HCA 专文 §4](../04-版本代际/05-CSA-HCA混合压缩注意力.md#v4-mixed-attention) · [V4 梗概](../04-版本代际/03-V4.md#推理-infra-关注点) 与 [演进总览 §5.3](../01-总览/01-版本演进总览.md#53-v4异构-kv--hisparse--磁盘-prefix-cache) 表格；**本文只讲引擎如何把它们落进内存**。

<img src="../01-总览/figures/v4/v4-hetero-kv.svg" alt="V4 异构 KV：Classical KV cache + per-request state blocks (paper sec 3.5.1)" width="920"/>

[图示详情](../01-总览/figures/v4/v4-hetero-kv.svg)

---

## Classical KV cache

**服务对象**：已凑满压缩块的 **CSA entry**（stride $m{=}4$）与 **HCA entry**（stride $m'{=}128$）。

| 项 | 说明 |
|----|------|
| **对齐粒度** | 按 $\mathrm{lcm}(m,m') = \mathrm{lcm}(4,128) = 128$ **token** 对齐的压缩块 |
| **为何 lcm** | CSA 每 4 token 一条、HCA 每 128 token 一条；同一物理块需同时满足两种压缩器的块边界，避免跨层/跨类型碎片 |
| **生命周期** | 随 prefix 增长 append；压缩完成后 entry **不可变**，适合 **[prefix 共享与落盘](07-V4-磁盘Prefix-Cache.md)** |
| **读模式** | CSA：top-$k$ 稀疏读；HCA：对 ~$L/128$ 条 entry **dense attend**（1M context ≈ 8K entry） |

Together.ai 部署解读强调：V4 的难点不在 attention kernel 本身，而在 **多种 cache 对象的 batching、eviction 与 prefix 复用**（[Serving DeepSeek-V4](https://www.together.ai/blog/serving-deepseek-v4-why-million-token-context-is-an-inference-systems-problem) §「multiple KV-cache layouts」）。

---

## State cache

**服务对象**：

| 组件 | 含义 |
|------|------|
| **SWA** | Sliding Window Attention 的 **精确局部** K/V；窗口约 128 token，独立 eviction |
| **Tail buffer** | 不足 $m$（CSA）或 $m'$（HCA）个 token 的 **未压缩尾**；等待凑满后再写入 Classical 池 |
| **Indexer KV**（若与 state 同池实现） | CSA lightning indexer 的轻量向量；与主 attention entry 维度不同 |

| 项 | 说明 |
|----|------|
| **分配方式** | **每请求固定大小块**（per-request state block），便于 batch 内不同序列长度对齐 |
| **生命周期** | SWA 随 decode **滑动更新**；tail 在凑满前 **可变** |
| **与 Classical 的分工** | Classical 存「已定型」压缩历史；State 存「仍在形成中」的局部与尾 |

论文 §3.5.1 的 **State cache** 是 SWA + tail 的归宿；部署上 SWA 往往占 **per-token 体积大头**（Together 早期 bring-up：全量 SWA 时 per-token KV 甚至略高于 V3 路径），因此 cache **策略**（存多少、何时 evict）比压缩算法本身更决定并发容量——见 [HiSparse 专文](06-V4-HiSparse.md)。

---

## 与 ESS / HiSparse 的关系

| | **ESS（V3.2）** | **V4 KV layout（本文）** | **HiSparse** |
|--|----------------|-------------------------|--------------|
| 层级 | 算法已分裂 Indexer/Latent；ESS 管 **offload 策略** | **定义** Classical vs State **物理布局** | 在 layout 之上把 **inactive C4** 搬出 GPU |
| 可否叠加 | — | layout 是 HiSparse / 磁盘 prefix 的 **前置** | 依赖 CSA 4:1 entry 的 **稀疏激活** |

> V4 **不是**把 ESS 的 Latent-Cache 直接放大到 1M；而是 **换一套异构压缩 layout** 后再做分层内存（[ESS §与 V4](01-ESS概念.md#与-index-share--v4-的关系)）。

---

## 基础设施线位置

| 方向 | 文档 |
|------|------|
| **本专题（layout 基础）** | [演进总览 §5.3](../01-总览/01-版本演进总览.md#v4-kv-layout) |
| **infra 线 ⑤ 子项** | [基础设施线导读 §1](../01-总览/06-基础设施线导读.md#1-演进链kv--offload) |
| **并列 offload** | [HiSparse](06-V4-HiSparse.md) · [磁盘 Prefix Cache](07-V4-磁盘Prefix-Cache.md) |
| **上游 ④ ESS** | [ESS Latent offload](01-ESS概念.md) |
| **算法依赖** | [CSA / HCA](../04-版本代际/05-CSA-HCA混合压缩注意力.md) · [V4 梗概](../04-版本代际/03-V4.md) · [算法线 §③](../01-总览/05-算法线导读.md#1-演进链attention--残差) |

---

## 延伸

| 资源 | 说明 |
|------|------|
| [演进总览 §5.4](../01-总览/01-版本演进总览.md#54-三代-offload-对比) | V3 / V3.2 ESS / V4 三代 offload 对照 |
| [Together.ai — Serving V4](https://www.together.ai/blog/serving-deepseek-v4-why-million-token-context-is-an-inference-systems-problem) | 多 layout 并存的 serving 含义 |
| [V4 异构 KV 总览图](../01-总览/figures/v4/v4-hetero-kv.svg) | Classical + HiSparse 分层示意图 |

**论文**：[arXiv:2606.19348](https://arxiv.org/abs/2606.19348) §3.5.1