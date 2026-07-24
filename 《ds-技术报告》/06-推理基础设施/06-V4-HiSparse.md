# V4 HiSparse：inactive C4 entry CPU offload

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §5.3 HiSparse](../01-总览/01-版本演进总览.md#v4-hisparse) · [← 基础设施线导读](../01-总览/06-基础设施线导读.md) · [CSA/HCA 算法专文](../04-版本代际/05-CSA-HCA混合压缩注意力.md) · [V4 梗概 §推理 infra](../04-版本代际/03-V4.md#推理-infra-关注点) · [前置 KV layout](05-V4-KV-Layout.md) · [并列 磁盘 Prefix Cache](07-V4-磁盘Prefix-Cache.md) · [上游 ESS](01-ESS概念.md) · [正交 DSpark](04-DSpark投机解码.md)
> **部署参考**：[Together.ai — Serving DeepSeek-V4](https://www.together.ai/blog/serving-deepseek-v4-why-million-token-context-is-an-inference-systems-problem)（2026-05，HGX B200 bring-up）
> **演进总览 §5.3** 只保留梗概；**HiSparse 机制与数据以本文为准**。

## 核心结论摘要

- **HiSparse** 将 inactive CSA 块 **CPU offload**，约 **3× KV 容量**。
- 与 DSpark 投机解码 **正交**，可叠加。
- 针对 V4 异构压缩 cache，不同于 V3.2 ESS 的 latent offload。
- 属于 V4 基础设施线 ⑤ 节点。

---

## 一句话

**HiSparse** 是面向 V4 **CSA 4:1 压缩层（C4）** 的 **GPU/CPU 分层 cache 策略**：decode 每步仅 **top-$k$ 稀疏激活** 少量压缩 entry，其余 **inactive** entry **offload 到 CPU pinned memory**；GPU 只保留 **active 热工作集**，从而在单节点上把可服务 token 容量从约 **1.2M 提升到 ~3.7M**（约 **3×**，B200 部署实测口径）。

---

## 为何需要 HiSparse

| 维度 | **ESS（V3.2）** | **HiSparse（V4）** |
|------|----------------|------------------|
| Cache 结构 | 同质 Latent-Cache（[per-token MLA） | **CSA/HCA/SWA/Indexer/tail** 异构](05-V4-KV-Layout.md) |
| Offload 粒度 | 按 **latent entry**；依赖 DSA top-$k$ **时间局部性** | 按 **C4 压缩 entry**；依赖 CSA **空间稀疏激活** |
| Indexer | GPU 常驻 | V4 indexer 仍参与 CSA 路径；layout 见 [DeepSeek-V4](../04-版本代际/03-V4.md) |
| 可否直接迁移 | — | **否** — 须先实现 §3.5.1 双池 layout |

V3.2 上 **Index Share + ESS 可同开**；V4 则需要 **HiSparse + 定制 layout + prefix 策略**（[演进总览 §5.4](../01-总览/01-版本演进总览.md#54-三代-offload-对比)）。

---

## 核心机制

| 组件 | 作用 |
|------|------|
| **C4 压缩 entry** | CSA stride-4：每 4 token 一条 KV；1M context 约 250K 条（再经 top-$k$ 只读子集） |
| **Active 集** | 当前 decode step **indexer 选中的** ~128 条 CSA entry + 必要 HCA/SWA 局部 |
| **Inactive 集** | 全长 prefix 中 **本步未参与 attention** 的 C4 entry |
| **CPU pinned pool** | Inactive entry 驻留 **主机 pinned memory**；GPU miss 时 **prefetch** 回 HBM |
| **GPU 热池** | LRU（或类似策略）维护 active 工作集；与 ESS Sparse Memory Pool 思想类似，但 **entry 形态为压缩块** |

<img src="../01-总览/figures/v4/v4-hetero-kv.svg" alt="HiSparse：inactive C4 entries offload 到 CPU pinned memory" width="920"/>

[图示详情](../01-总览/figures/v4/v4-hetero-kv.svg) · 图下半区标注 HiSparse offload

**局部性依据**：CSA 每步 top-$k$ 选中的压缩块在相邻 decode step 间 **重叠率高**（类比 ESS 的 index 时间相似度）；多数所需 entry 已在 GPU，少量 cold entry 从 CPU 拉回。

---

## 部署数据

| 指标 | 数值 | 说明 |
|------|------|------|
| 平台 | NVIDIA **HGX B200** 单节点 | Together 早期 V4 bring-up |
| 优化前容量 | ~**1.2M tokens** | 默认 cache 策略下总 KV 预算 |
| HiSparse + cache 策略后 | ~**3.7M tokens** | 约 **3×**；主要释放来自 **inactive C4 offload** + SWA 复用策略 |
| SWA 注意 | 全量 SWA 时 per-token KV 可 **高于** V3 路径 | Together 称早期瓶颈常在 **SWA state** 而非 CSA/HCA 压缩体本身 |

Together 文中同时提到：通过 **只保留最可能被复用的 SWA state**，在不改权重的情况下提升总容量——这与 **C4 CPU offload** 互补，共同构成 V4 serving 的 **cache policy** 层（非单一 knob）。

---

## 与磁盘 Prefix Cache / DSpark 的关系

| 技术 | 关系 |
|------|------|
| [KV layout](05-V4-KV-Layout.md) | **前置**：须先分 Classical / State 池，HiSparse 主要动 **Classical 中 C4 部分** |
| [磁盘 Prefix Cache](07-V4-磁盘Prefix-Cache.md) | **互补**：压缩 entry 可 **跨请求落盘**；HiSparse 管 **单请求内 GPU↔CPU 热冷分层** |
| [DSpark](04-DSpark投机解码.md) | **正交**：DSpark 优化 decode **步吞吐**；HiSparse 优化 **KV 驻留容量** |

---

## 与 ESS 对照

| 维度 | V3.2 (ESS) | V4 (HiSparse) |
|------|------------|---------------|
| Offload 对象 | 仅 **Latent-Cache** | **Inactive C4 压缩 entry** |
| 局部性 | top-$k$ index 时间相似 | CSA **稀疏激活** + SWA 复用 |
| 传输优化 | FlashTrans / UVA | 分层内存池 +（引擎）PD 分离 |
| 与算法耦合 | 中（依赖 DSA top-$k$） | **高**（依赖压缩比 $m{=}4$、$m'{=}128$） |

完整表：[演进总览 §5.4](../01-总览/01-版本演进总览.md#54-三代-offload-对比)。

---

## 基础设施线位置

| 方向 | 文档 |
|------|------|
| **本节点（⑤ HiSparse）** | [基础设施线导读 §1](../01-总览/06-基础设施线导读.md#1-演进链kv--offload) |
| **前置 layout** | [V4 KV Layout](05-V4-KV-Layout.md) |
| **并列 prefix** | [V4 磁盘 Prefix Cache](07-V4-磁盘Prefix-Cache.md) |
| **上游 ④ ESS** | [ESS Latent offload](01-ESS概念.md)（**非**简单放大） |
| **V4 总览** | [DeepSeek-V4 梗概§推理 infra](../04-版本代际/03-V4.md#推理-infra-关注点) |

---

## 延伸

| 资源 | 说明 |
|------|------|
| [Together.ai — Serving V4](https://www.together.ai/blog/serving-deepseek-v4-why-million-token-context-is-an-inference-systems-problem) | 1.2M→3.7M、SWA 瓶颈、多 layout serving |
| [ESS 概念](01-ESS概念.md) | V3.2 offload 基线对照 |
| [演进总览 §5.3](../01-总览/01-版本演进总览.md#v4-hisparse) | 总览反向链入口 |

**论文背景**：[arXiv:2606.19348](https://arxiv.org/abs/2606.19348)（算法侧 CSA/HCA；HiSparse 为 **社区/部署层命名**，与 Together cache policy 实践一致）