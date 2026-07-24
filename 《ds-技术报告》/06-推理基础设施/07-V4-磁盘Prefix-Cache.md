# V4 磁盘 Prefix Cache

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §5.3 磁盘 Prefix Cache](../01-总览/01-版本演进总览.md#v4-disk-prefix-cache) · [← 基础设施线导读](../01-总览/06-基础设施线导读.md) · [CSA/HCA 算法专文](../04-版本代际/05-CSA-HCA混合压缩注意力.md) · [V4 梗概 §推理 infra](../04-版本代际/03-V4.md#推理-infra-关注点) · [前置 KV layout](05-V4-KV-Layout.md) · [并列 HiSparse](06-V4-HiSparse.md) · [上游 ESS](01-ESS概念.md)
> **论文**：[arXiv:2606.19348](https://arxiv.org/abs/2606.19348) §3.5.2
> **部署参考**：[Together.ai — Prefix caching becomes a storage policy](https://www.together.ai/blog/serving-deepseek-v4-why-million-token-context-is-an-inference-systems-problem#prefix-caching-becomes-a-storage-policy)
> **演进总览 §5.3** 只保留梗概；**prefix 三档策略以本文为准**。

## 核心结论摘要

- 将 CSA/HCA 历史块 **落盘**，配合 **SWA 三档策略** 管理长前缀。
- 面向 Agentic Coding 等超长 prefix 场景（100K–1M token）。
- 与内存侧 HiSparse、算法侧 CSA/HCA 协同。
- 见 V4 论文 §3.5.2 与演进总览 §5.3。

---

## 一句话

V4 的 prefix 不再等于「整条同质 KV 共享一次 prefill」：引擎须按 **cache 类型** 决定存什么——**CSA/HCA 压缩 entry 可直接落盘**（紧凑、immutable），**SWA 精确局部 state 体积约为压缩 entry 的 ~8×**，论文给出 **Full / Periodic Checkpointing / Zero** 三档 SWA 策略，在 **存储带宽 vs 命中重算** 之间取舍。

---

## 从「共享 prefix」到「共享哪几类 cache」

传统 prefix cache：**相同 token 前缀 → 复用同一份 KV**，跳过 prefill。

[V4 一条共享 prefix 实际包含](05-V4-KV-Layout.md)：

| Cache 类型 | 是否适合落盘 | 原因 |
|------------|-------------|------|
| **CSA 压缩 entry** | ✅ 优先 | 4:1 紧凑；块对齐后 **不可变** |
| **HCA 压缩 entry** | ✅ 优先 | 128:1 极紧凑；1M prefix ≈ 8K entry |
| **Tail buffer** | ⚠️ 随 prefix 推进变化 | 未凑满块；通常随请求走 State 池 |
| **Indexer KV** | 视引擎实现 | 可与 CSA 路径一并序列化 |
| **SWA state** | ⚠️ **体积大** | 精确局部 K/V；长约 **压缩 entry 的 8×** |

Together 总结：**Prefix caching becomes a storage policy** — 存 CSA/HCA、**如何**存 SWA、各类型 **独立 eviction**。

---

## CSA / HCA：压缩 entry 落盘

| 项 | 说明 |
|----|------|
| **收益** | 多请求共享 repo / system prompt / tool trace 等 **长公共前缀** 时，**免重复 prefill** 压缩路径 |
| **格式** | Classical KV cache 中已对齐的 **immutable 压缩块**（$\mathrm{lcm}(4,128)$ 对齐，见 [V4 KV Layout](05-V4-KV-Layout.md)） |
| **与 HiSparse** | 磁盘层 = **跨请求 / 跨会话** 持久；HiSparse = **单请求内** GPU↔CPU 热冷（[V4 HiSparse](06-V4-HiSparse.md)） |
| **命中路径** | 加载 CSA/HCA 块 → 仅需补全 **tail + SWA**（策略见下） |

---

## SWA 三档策略

SWA 保存最近 $n_{\text{win}}$（约 128）token 的 **精确** attention state。对 **长 prefix**，全量 SWA 的 **存储与写带宽** 迅速成为瓶颈。

| 策略 | 行为 | 优点 | 代价 |
|------|------|------|------|
| **Full** | 落盘 / 驻留 **完整 SWA cache** | 命中后 **零重算**，复用最简单 | footprint **最大**；Together 早期 bring-up 采用此策略以降低工程复杂度 |
| **Periodic Checkpointing** | 每 $K$ token 存一份 SWA **检查点**；命中时在相邻检查点间 **重算 gap** | 存储介于 Full 与 Zero 之间 | 命中需 **部分 prefill 重放** |
| **Zero（Recompute on hit）** | **只存** CSA/HCA（+ tail 元数据）；命中 prefix 时 **按窗口重算 SWA** | 磁盘最省；适合 **极长 prefix、高复用、SWA 窗口相对短** | 重算上界 ≈ $n_{\text{win}} \times L_{\text layers}}$ token 量级 |

**Zero 策略数量级**：$n_{\text{win}}{=}128$、$L_{\text layers}}{=}61$ → 约 **8K tokens** 重算，相对 **1M prefix** 往往可接受。

```
命中 1M 公共前缀（Zero SWA）：
 磁盘/共享池：CSA + HCA 压缩块（~L/4 + L/128 条）
 重算：最后 128×61 ≈ 8K token 的 SWA 路径
 vs 全量 prefill 1M：大幅省算力，略增命中延迟
```

---

## 策略选型

| workload | 倾向策略 | 理由 |
|----------|---------|------|
| Agent / coding（长共享 repo） | 优先 **CSA/HCA 落盘** + SWA **Periodic 或 Zero** | 前缀极长、重复率高；SWA Full 占满 KV 预算 |
| 短 chat、低 prefix 复用 | SWA **Full** 或不做磁盘 prefix | 重算省下的存储不值 setup 成本 |
| 高并发、磁盘/PCIe 带宽紧 | **Zero** + HiSparse GPU 热池 | 磁盘只存压缩体；SWA 命中时再算 |

Together 当前（2026-05）：**Full SWA** + 激进 cache eviction → 1.2M→3.7M tokens；说明 **策略与 HiSparse 强耦合**，无单一默认最优。

---

## 与 ESS / V3 prefix cache

| | V3 / V3.2 | V4 磁盘 Prefix |
|--|-----------|----------------|
| 共享对象 | 同质 MLA latent 条 | **分类型**：CSA/HCA vs SWA |
| ESS 角色 | Latent CPU offload，**非**磁盘 prefix 专论 | V4 需 **新** tiering（GPU / CPU pinned / **磁盘**） |
| Index Share | 与 prefix **正交**（省 indexer 算力） | V4 仍可能有 indexer 复用，但 **不在** §3.5.2 核心 |

---

## 基础设施线位置

| 方向 | 文档 |
|------|------|
| **本专题（§3.5.2）** | [演进总览 §5.3](../01-总览/01-版本演进总览.md#v4-disk-prefix-cache) |
| **infra 线 ⑤ 子项** | [基础设施线导读 §1](../01-总览/06-基础设施线导读.md#1-演进链kv--offload) |
| **前置 layout** | [V4 KV Layout](05-V4-KV-Layout.md) |
| **并列 HiSparse** | [V4 HiSparse](06-V4-HiSparse.md) |
| **V4 总览** | [DeepSeek-V4 梗概§推理 infra](../04-版本代际/03-V4.md#推理-infra-关注点) |

---

## 延伸

| 资源 | 说明 |
|------|------|
| [Together.ai — Prefix caching](https://www.together.ai/blog/serving-deepseek-v4-why-million-token-context-is-an-inference-systems-problem) | 三档 SWA 工程解读与 Full 策略取舍 |
| [演进总览 §5.4](../01-总览/01-版本演进总览.md#54-三代-offload-对比) | 磁盘 prefix 在三代 offload 表中的位置 |
| [V4 异构 KV 总览图](../01-总览/figures/v4/v4-hetero-kv.svg) | 异构 cache 总览图 |

**论文**：[arXiv:2606.19348](https://arxiv.org/abs/2606.19348) §3.5.2