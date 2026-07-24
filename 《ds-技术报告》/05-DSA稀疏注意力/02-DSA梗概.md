# DSA稀疏注意力

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §3.6](../01-总览/01-版本演进总览.md#36-deepseek-v32--v32-exp) · [← 算法线导读](../01-总览/05-算法线导读.md) · [← 基础设施线导读](../01-总览/06-基础设施线导读.md) · [← V3.2 梗概](../04-版本代际/02-V3.2-DSA.md) · [上游 MLA](../02-基座架构/02-MLA低秩注意力.md) · [下游 CSA/HCA](../04-版本代际/05-CSA-HCA混合压缩注意力.md) · [下游 V4 梗概](../04-版本代际/03-V4.md) · [下游 Index Share](05-Index-Share梗概.md) · [下游 ESS](../06-推理基础设施/01-ESS概念.md) · [Lightning Indexer 详解](04-Lightning-Indexer详解.md) · [完整逻辑](03-DSA逻辑详解.md) · [Raschka §4 DSA](../08-外部解读/01-Raschka要点速读.md#与本地文档映射)
> **论文**：[DeepSeek-V3.2 arXiv:2512.02556](https://arxiv.org/pdf/2512.02556) · Exp：[DeepSeek-V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2)

## 核心结论摘要

- DSA 在 **MLA 不变**前提下，用 indexer 选 top-k=2048 再做 Core MLA。
- 复杂度 **O(L²) → O(Lk)**；k 个位置是学到的、内容相关的，非固定滑动窗口。
- 三阶段：Lightning Indexer → Top-k Selector → Core MLA。
- V3.2 唯一架构改动；ESS / IndexCache 为叠在其上的 infra 补丁。

---

## 一句话

**DSA** 在 **MLA 不变** 的前提下，把长上下文注意力从「对全长 $L$ 做稠密 MLA」改成 **先 indexer 扫全长选 top-$k$，再只对 $k$ 个 latent entry 做 MLA**；主路径复杂度 **$O(L^2) \to O(Lk)$**（$k{=}2048$，$k \ll L$）。**DeepSeek-V3.2-Exp**（2025-09，DeepSeek 官方实验版）验证稀疏不损精度；**DeepSeek-V3.2**（2025-12）为正式版。

> **逻辑详解**：[DSA逻辑详解](03-DSA逻辑详解.md) · [Lightning Indexer](04-Lightning-Indexer详解.md) · [系列导读](01-系列导读.md)

---

## 技术归属

| 组件 | 机构 | 说明 |
|------|------|------|
| **DSA** | **DeepSeek** | 稀疏注意力算法；V3.2 **唯一架构改动** |
| **V3.2-Exp / V3.2** | **DeepSeek** | 官方模型 release（Exp 铺生态，正式版完整后训练） |
| **ESS** | 百度百舸 | 针对 DeepSeek V3.2 的 **Latent-Cache offload**；**非** DSA 发明方 |
| **IndexCache** | 清华 + 智谱 | 跨层 index 复用；**非** DSA 发明方 |

> 易混点：ESS 论文标题写 *for DeepSeek-V3.2-Exp*，指的是**优化对象**是 DeepSeek 模型，不是百度发布了 V3.2-Exp。

---

## 流程图

<img src="figures/dsa-pipeline.svg" alt="DSA：Lightning Indexer → Top-k → Core MLA；Indexer-Cache 与 Latent-Cache" width="920"/>

[图示详情](figures/dsa-pipeline.svg) · [系列目录](01-系列导读.md)

---

## 三阶段

| 阶段 | 做什么 | 复杂度 | 说明 |
|------|--------|--------|------|
| **① [Lightning Indexer](04-Lightning-Indexer详解.md)** | **当前** $q_t$ 对全长历史的 indexer **key** $k_s$ 打分（$I_{t,s}$） | $O(L^2)$ 量级，常数极小 | 决定「看谁」；**[walkthrough](04-Lightning-Indexer详解.md#decode-forward-walkthrough)** |
| **② Top-$k$ Selector** | 取分数最高的 **$k{=}2048$** 个位置 | $O(L \log k)$ | 得到 index 集合 $I$ |
| **③ Core MLA** | **仅对** $I$ 中 entry 做标准 MLA attention | $O(Lk)$ | 精度敏感主算子 |

<img src="figures/dsa-three-stage.svg" alt="DSA 每层三阶段：Query + 全长历史 → Lightning Indexer → Top-k → Core MLA ← Latent-Cache" width="920"/>

[图示详情](figures/dsa-three-stage.svg)

[**与滑动窗口的区别**：DSA 的 $k$ 个位置是 **学到的](../08-外部解读/01-Raschka要点速读.md)、[内容相关** 的](../08-外部解读/01-Raschka要点速读.md)、[不是固定局部窗口](../08-外部解读/01-Raschka要点速读.md)。

> **Lightning Indexer 专题**：[Lightning Indexer 详解](04-Lightning-Indexer详解.md)

---

## 异构 KV Cache

DSA 把 cache **拆成两类**（为 ESS offload、Index Share 铺路）：

| Cache | 作用 | 占比（ESS 论文） | GPU 常驻 |
|-------|------|------------------|----------|
| **Indexer-Cache** | indexer 打分、选 top-$k$ | ~16.8% | **是**（每步全扫） |
| **Latent-Cache** | 核心 MLA 的 latent KV | ~83.2% | 可 offload（[ESS](../06-推理基础设施/01-ESS概念.md)） |

主 attention 只读 **被选中的 $k$ 个** latent entry → Latent-Cache 适合稀疏访问与 CPU 分层。

---

## 基础设施线位置

| 方向 | 文档 |
|------|------|
| **本节点（② Indexer/Latent 异构）** | [基础设施线导读 §1](../01-总览/06-基础设施线导读.md#1-演进链kv--offload) |
| **上游 ① 同质 MLA KV** | [MLA 低秩注意力](../02-基座架构/02-MLA低秩注意力.md) |
| **下游 ③ Index Share** | [Index Share 梗概](05-Index-Share梗概.md)（indexer 算力，**并列**） |
| **下游 ④ ESS** | [ESS Latent offload](../06-推理基础设施/01-ESS概念.md)（Latent offload，**并列**） |
| **下游 ⑤ V4 infra** | [DeepSeek-V4 梗概§推理 infra](../04-版本代际/03-V4.md#推理-infra-关注点) · [KV layout](../06-推理基础设施/05-V4-KV-Layout.md) · [HiSparse](../06-推理基础设施/06-V4-HiSparse.md) · [磁盘 prefix](../06-推理基础设施/07-V4-磁盘Prefix-Cache.md) |

---

## 算法线位置

| 方向 | 文档 |
|------|------|
| **本节点（② DSA）** | [算法线导读 §1](../01-总览/05-算法线导读.md#1-演进链attention--残差) |
| **上游 ① MLA** | [MLA 低秩注意力](../02-基座架构/02-MLA低秩注意力.md) |
| **下游 ③ CSA/HCA** | [CSA / HCA](../04-版本代际/05-CSA-HCA混合压缩注意力.md) · [DeepSeek-V4](../04-版本代际/03-V4.md) |

---

## 在版本线中的位置

| 版本 | DSA |
|------|-----|
| V3 / V3.1-T | 稠密 MLA，**无 DSA** |
| **V3.2-Exp** | 在 Terminus 上 **续训加 DSA**；benchmark 平淡，**铺推理生态** |
| **V3.2** | 架构同 Exp；完整后训练成品 |
| **Index Share** | **不改 DSA 算法**；跨层复用 top-$k$ index，减 indexer 重复计算（[Index Share 梗概](05-Index-Share梗概.md)） |
| **V4** | CSA/HCA 等 **下一代**稀疏/压缩注意力 |

相对 V3.1-Terminus，V3.2 **唯一架构改动**即为 DSA；MoE、MLA latent 格式、参数量均不变。

---

## 与 MLA 的关系

[- **MLA**：K/V 压入 latent 再缓存](../02-基座架构/02-MLA低秩注意力.md)
- **DSA**：在 MLA latent **序列**上增加「选哪些位置参与 attention」
- V3.1 **Hybrid**（Prefill MHA / Decode MQA）仍是 DSA 的前置（[DeepSeek-V3.1](../04-版本代际/01-V3.1-Terminus.md)）

---

## 推理 infra

| 组件 | 作用 |
|------|------|
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | indexer logit kernel |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | sparse MLA paged kernel |
| [IndexCache](https://github.com/THUDM/IndexCache) | Index Share 跨层 index 复用 |
| [ESS](../06-推理基础设施/01-ESS概念.md) | Latent-Cache CPU offload |

---

## 延伸

| 资源 | 说明 |
|------|------|
| [Lightning Indexer 详解](04-Lightning-Indexer详解.md) | **Lightning Indexer** 公式、Indexer-Cache、与滑动窗对比 |
| [DSA逻辑详解](03-DSA逻辑详解.md) | 完整机制、与 Hybrid/ESS/Engram 关系 |
| [Index Share逻辑详解](06-Index-Share逻辑.md) | Index Share `FFFS` 模式 |
| [DeepSeek-V3.2](../04-版本代际/02-V3.2-DSA.md) | V3.2 版本梗概 |
| [Raschka DSA 解读](../08-外部解读/01-Raschka要点速读.md) | 第三方梳理 |

**论文**：V3.2 [2512.02556](https://arxiv.org/pdf/2512.02556) · ESS [2512.10576](https://arxiv.org/abs/2512.10576)