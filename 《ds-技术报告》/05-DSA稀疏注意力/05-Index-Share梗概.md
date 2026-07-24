# Index Share梗概

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §4](../01-总览/01-版本演进总览.md#4-index-shareindexcache) · [← 基础设施线导读](../01-总览/06-基础设施线导读.md) · [← 版本目录](../01-总览/02-版本梗概索引.md) · [上游 DSA §异构 KV](02-DSA梗概.md#异构-kv-cache) · [并列 ESS](../06-推理基础设施/01-ESS概念.md) · [逻辑详解](06-Index-Share逻辑.md)

## 核心结论摘要

- **IndexCache** 跨层复用 DSA indexer 的 top-k index，纯 **推理 infra 补丁**。
- 不改模型权重；社区有时称「V3.3」但非 DeepSeek 官方版本号。
- 清华 + 智谱工作；与 ESS 正交，可叠加。
- 逻辑详解见 [Index Share 系列](06-Index-Share逻辑.md)。

---

## 定位

社区昵称 **Index Share** / 「**V3.3**」；正式名 **IndexCache**（清华 + Z.ai，2026-03）。**不是新模型**，而是面向 **[DSA](02-DSA梗概.md)** 架构的 **纯推理 infra 补丁**：零额外显存，在 V3.2 / GLM-5 等模型上即插即用。

典型体现「**infra 归 infra，算法归算法**」——算法仍是 V3.2 的 **[DSA](02-DSA梗概.md)**，系统侧利用跨层冗余砍掉冗余 indexer 计算。

> **逻辑详解**：[Index Share逻辑详解](06-Index-Share逻辑.md) · [上游 DSA §异构 KV](02-DSA梗概.md#异构-kv-cache)

---

## 基础设施线位置

| 方向 | 文档 |
|------|------|
| **本节点（③ Index Share）** | [基础设施线导读 §1](../01-总览/06-基础设施线导读.md#1-演进链kv--offload) |
| **前置 ② 异构 cache** | [DSA稀疏注意力§异构 KV](02-DSA梗概.md#异构-kv-cache) |
| **并列 ④ ESS** | [ESS Latent offload](../06-推理基础设施/01-ESS概念.md)（Latent offload，可同开） |
| **下游 ⑤ V4** | [CSA / HCA](../04-版本代际/05-CSA-HCA混合压缩注意力.md) · [DeepSeek-V4](../04-版本代际/03-V4.md)（V4 自带 CSA indexer，路线互补） |

---

## 技术归属

### 结论摘要

| # | 要点 |
|---|------|
| 1 | **IndexCache / Index Share**（跨层索引复用）**不是** DeepSeek 原创，**也不是**百度百舸原创 |
| 2 | **DSA 稀疏注意力**（含 **Lightning Indexer**）为 **DeepSeek 自研**，是被优化的模型侧基底 |
| 3 | **IndexCache** 由 **清华大学计算机系 + 智谱 AI（Z.ai）** 联合提出（[arXiv:2603.12201](https://arxiv.org/abs/2603.12201)） |
| 4 | **百度百舸（Baige AI）** 在 IndexCache 上主要是**工程集成与云侧落地**；其**自研**的同类 infra 是 **[ESS](../06-推理基础设施/01-ESS概念.md)**（Latent-Cache offload），与 IndexCache **正交** |

### 各方分工

| 机构 | 角色 | 代表工作 |
|------|------|----------|
| **DeepSeek** | 造出带 Lightning Indexer 的 **DSA 模型**（**被优化对象**） | DSA、[arXiv:2512.02556](https://arxiv.org/pdf/2512.02556)；每层独立 top-$k$，$O(L^2)$ indexer + $O(Lk)$ Core MLA |
| **清华 + 智谱（Z.ai）** | 提出 **IndexCache / index-share** 跨层索引复用（**优化算法本体**） | Full (F) 层算索引并 **index-cache**；Shared (S) 层 **index-share** 复用；`FFFS` 等模式 |
| **百度百舸** | **ESS** 原创 + IndexCache **训推引擎适配/部署**（**落地方**） | [ESS arXiv:2512.10576](https://arxiv.org/abs/2512.10576)（Latent-Cache offload）；百舸云侧集成 IndexCache、KV 缓存、并行调度等 |

### IndexCache 论文中的两个名词

（勿与 DSA 里的 **Indexer-Cache** 存储块混淆，见下节。）

| 论文术语 | 含义 |
|----------|------|
| **index-cache** | F（Full）层算出 top-$k$ 后，把**索引集合**缓存起来 |
| **index-share** | S（Shared）层**不跑 indexer**，直接复用最近 F 层的缓存索引 |

相邻层索引重叠可达约 **70%–100%**；典型 `FFFS` 下约 **75%** 层可跳过 indexer 计算。

### DeepSeek 侧：DSA 为何需要这类补丁

- DSA 每层独立 lightning indexer，长上下文 Prefill 时 indexer 可占主导耗时；
- IndexCache **不改 DSA 结构**，只在调度上减少「重复跑 indexer」的次数。

### 百舸侧：两套不同技术

| 技术 | 归属 | 优化对象 |
|------|------|----------|
| **IndexCache** | 清华 + 智谱 **发明**；百舸 **集成部署** | DSA **indexer 计算**（跨层复用 top-$k$ 下标） |
| **ESS** | 百舸 **原创**（Chen et al., 2025） | DSA **Latent-Cache 显存**（CPU offload + 热池） |

### 易混淆：三类「Cache」

| 名称 | 谁的概念 | 是什么 |
|------|----------|--------|
| **Indexer-Cache** | DSA / ESS 文档 | GPU 常驻的 **indexer 向量 KV**（~16.8%），每步参与打分 |
| **Latent-Cache** | DSA / ESS 文档 | **MLA latent KV**（~83.2%），ESS 可 offload |
| **index-cache**（IndexCache 论文） | 清华 + 智谱 | F 层产出的 **top-$k$ 下标缓存**（计算复用，非 KV 存储块） |
| **DeepSeek API 硬盘上下文缓存** | DeepSeek 业务 | 请求级 **前缀 KV** 复用，与 IndexCache **无关** |

> **逻辑详解**：[Index Share逻辑详解](06-Index-Share逻辑.md) §1

---

[DSA](02-DSA梗概.md) 每层独立跑 lightning indexer，复杂度 $O(L^2)$。长上下文 prefill 时 indexer 成为显著瓶颈。观察：**相邻层选出的 top-$k$ index 高度相似**。

## 机制

> **逻辑详解**：[Index Share逻辑详解](06-Index-Share逻辑.md) · [DSA 前置](02-DSA梗概.md)

层划分为两类：

| 类型 | 行为 |
|------|------|
| **Full (F)** | 保留 indexer，正常计算 top-$k$ |
| **Shared (S)** | 不跑 indexer，复用最近 F 层的 cached indices |

典型模式 `FFFS` 重复：每 4 层留 1 个 F 层，**去掉 75% indexer 计算**。

部署模式：

- **Training-free**：校准集上贪心选保留哪些层的 indexer
- **Training-aware**：多层蒸馏，让 F 层 indexer 拟合覆盖层的平均 attention 分布

## 效果

| 指标 | 加速 |
|------|------|
| Prefill（TTFT） | **1.82×** |
| Decode 吞吐 | **1.48×** |
| 精度损失 | 可忽略 |

## 推理 infra 关注点

- 实现：SGLang / vLLM patch，核心为一个 `if/else` 分支
- **零额外 GPU 显存**
- 仅适用于 **[DSA](02-DSA梗概.md) 系**（DeepSeek-V3.2、GLM-5 等）
- 与 **[ESS](../06-推理基础设施/01-ESS概念.md)** Latent-Cache offload 正交，可叠加

## 与 V4 对比

| 属性 | Index Share | V4 |
|------|------------|-----|
| 权重 | 不变 | 全量重训 |
| 改动量 | 极小 | 架构级 |
| 上下文 | 128K 系优化 | 原生 1M |
| Ablation | 干净（纯 infra） | 多变量纠缠 |

## 参考

- 论文：[arXiv:2603.12201](https://arxiv.org/abs/2603.12201)
- 代码：[THUDM/IndexCache](https://github.com/THUDM/IndexCache)
- 前置版本：[DeepSeek-V3.2](../04-版本代际/02-V3.2-DSA.md)