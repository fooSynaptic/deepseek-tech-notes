# 投机解码与 DSpark

> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · [← 演进总览 §2 时间线](../reports/deepseek-version-lineage-20260625.md#2-版本时间线与关系) · [§3.3 V3 MTP](../reports/deepseek-version-lineage-20260625.md#33-deepseek-v3) · [§3.7 V4 + DSpark](../reports/deepseek-version-lineage-20260625.md#37-deepseek-v4) · [§6 推理栈](../reports/deepseek-version-lineage-20260625.md#6-推理技术栈对照) · [V3 梗概](./v3.md) · [V4 梗概](./v4.md)
> **DSpark 开源**：[DeepSpec](https://github.com/deepseek-ai/DeepSpec) · [DSpark_paper.pdf](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf) · [DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark)
> **外部材料**：[版本演进总览](../reports/deepseek-version-lineage-20260625.md) · [V3 论文 MTP 原文摘录](https://arxiv.org/pdf/2412.19437)

**本文档**聚合 DeepSeek 系列 **投机解码（speculative decoding）** 与 **DSpark** 全部材料：通用循环、V3/V4 **MTP**、外挂 draft 自测、生产 **MTP-1** 基线、**DSpark** 机制与线上数据。演进总览 / V3 / V4 / 基础设施线 **只保留链接，细节以本文为准**。

## 核心结论摘要

- 投机解码把 **k 次搬大模型权重**压成 **1 次** target 前向 + 便宜 draft。
- V3 **MTP** 链可做 draft；V4 生产侧 **DSpark** 为半自回归 draft + 置信度调度验证。
- 与 HiSparse **正交**，可叠加提升 decode 吞吐。
- 开源参考：[DeepSpec](https://github.com/deepseek-ai/DeepSpec)。

---

## 1. 投机解码：为何能加速推理

> **场景**：batch=1、自回归 decode。下文以 **gpt2-small draft + gpt2-xl target** 为例说明通用机制；DeepSeek 路线见 §2。

### 1.1 瓶颈：每 token 搬一遍大模型权重

自回归生成**严格串行**：第 $N$ 个 token 必须等第 $N{-}1$ 个算完。每步都要把 **target 全部权重从 HBM 搬进 SM（计算单元）**（[名词：SM](./qa/gpu-sm-term.md)；大矩阵 × 单向量，GEMV 算术强度极低）。

在 batch=1 decode 下，时间主要花在**搬权重**而非 FLOPS。因此：

- target 生成 **1** 个 token；
- target **一次并行验证 $k$ 个**候选 token；

在带宽瓶颈下耗时 **几乎相同**（同样搬一遍权重，多算几位几乎不增时延）。

突破口：**把「$k$ 次搬大模型权重」压成「1 次」**，用便宜的 draft 换少跑几次 target。

> **答疑**：[Compute-Bound vs Memory-Bound — DFlash / Eagle 如何对应？](./qa/spec-decode-compute-vs-memory-bound.md) — §1.1 的 Memory-Bound 串行读；§4 Eagle（draft 串行、$\tau_q \propto K$）vs DFlash（draft 并行、一次 HBM 流）。

### 1.2 一轮怎么走：draft → verify → accept

| | 出 4 个 token 时（$k{=}4$） |
|--|---------------------------|
| **传统** | xl $\to t_1 \to$ xl $\to t_2 \to$ xl $\to t_3 \to$ xl $\to t_4$ → **4 次** target 前向 |
| **投机** | small 猜 $t_1 t_2 t_3 t_4$ → xl **并行验证 1 次** → 接受最长前缀（最多 4 个） |

**① Draft（便宜）猜 $k$ 个 token**

- 例：gpt2-small 权重约为 gpt2-xl 的 $\sim 1/12$，搬一次快得多。
- 小模型自回归猜出 $k$ 个候选，总代价远低于 xl 跑一步。
- 生产里 draft 也可来自 **MTP 头**、DSpark 草稿模块等（§2、§5），循环相同。

**② Target（贵，但每轮只一次）并行验证这 $k$ 个**

- 把 $k$ 个候选拼成长度 $k$ 的序列，**一次前向**喂给 xl。
- Attention 并行处理多个位置 → **1 次前向同时得到这 $k$ 个位置的概率分布**。
- 耗时 $\approx$ 原来生成 1 个 token（同样只搬一遍 xl 权重）。

**③ Accept / reject（保证 lossless）**

用 xl 的真实分布逐位校验 draft（[Leviathan et al.](https://arxiv.org/abs/2211.17002) / [Xia et al.](https://arxiv.org/abs/2202.01318) 的 rejection sampling）：猜中的**连续前缀**直接采纳；**第一个猜错处**用 xl 分布重采 1 个，其后丢弃，进入下一轮。单步「提议 → 校验」的概率含义见 §1.3。

**端到端加速取决于**：

| 杠杆 | 含义 |
|------|------|
| **每轮接受长度** | 平均每次 target 前向收下几个 token（头号因素） |
| **draft 成本** | 猜 $k$ 个要花多少小模型前向；过大或命中率太低会吃掉收益 |
| **验证占用**（高并发） | target 算力是否花在「高存活概率」前缀上（DSpark §6） |

### 1.3 净收益、lossless 与最坏情况

设单次 target 前向耗时 $\approx T$（与 $k$ 弱相关）：

| | 传统（$k$ 步） | 投机（$k$ 候选，全接受） |
|--|----------------|-------------------------|
| target 前向 | $k$ 次 | **1** 次 |
| 产出 token | $k$ | $k$ |
| 单次 target 耗时 | $T$ | $\approx T$ |

- **猜得准**：1 次 target 吐出多个 token → 理论 $\approx k\times$，实际约 **2–3×**（§3 Qwen 自测 **2.4–2.9×**），需扣除 draft 开销。

**lossless**：rejection sampling 保证与「逐 token 采样 target」**同分布**；draft 只改 $q$ 与接受有多快，不改输出分布。

> **答疑**：[接受路径、拒绝路径与 lossless 证明](./qa/spec-decode-rejection-sampling.md) — 单步 $X \sim q$、$U \le \alpha(X)$ 校验；$E_{\mathrm{acc}}(x)$ 上 $P=q\alpha$；$\alpha=\min(1,p/q)$ 与 $p_{\mathrm{resample}}\propto\max(0,p-q)$ 合并得 $\pi(x)=p(x)$。

- **全猜错**：target 仍 **1** 次前向，保底 **1** 个正确 token（与逐 token 相同）；额外多花 $k$ 次 draft 小前向 → 最坏 $\approx$ 原速 $\times (1 + \text{小常数})$，略慢但不翻车。

### 1.4 PoC 参照

- **gpt2-small（124M）+ gpt2-xl（1.5B）**：同 GPT-2 BPE、同词表，draft 权重约 target 的 $1/12$，**无需训练**即可验证约 **2×** 延迟收益；机理见 §1.1–§1.2。
- **Qwen3-4B + 0.6B draft**：本仓库 vLLM 自测见 §3。

---

## 2. DeepSeek 路线：MTP

[V3 技术报告](https://arxiv.org/abs/2412.19437) 引入 **Multi-Token Prediction（MTP）**：在 **主 next-token 目标** 之外，用 $D$ 个串联 **MTP 模块** 预测 $t{+}2, t{+}3, \ldots$ 额外 token。训练时共享 Embedding / Output Head；每层 MTP 通过 **Integration Layer** 把 **上一层 hidden** 与 **中间 token 的 Emb** 融合后再过 1 个 Transformer Block。

> **中间 token 融合读图**：[MTP 融合 scheme](./qa/mtp-fusion-scheme.md) · [MTP 融合 scheme 图](./figures/mtp-fusion-scheme.svg)

**训练目标**：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{main}} + \lambda \sum_{k=1}^{D} \mathcal{L}_{\mathrm{MTP}}^{(k)}
$$

- **主目标**：提升主模型性能、数据效率。
- **推理**：可 **丢弃** MTP 模块、主模型独立 decode；也可 **复用 MTP 模块做 speculative decoding** 进一步降延迟。

[V4](https://arxiv.org/abs/2606.19348) **保留** MTP 模块与训练目标（与 V3 同族实现）。

<img src="./figures/mtp-speculative.svg" alt="V3 MTP 训练结构；MTP 原生投机 vs 外挂 draft vs DSpark 对照" width="920"/>

[图示详情](./figures/mtp-speculative.svg) · [V3 §三 MTP](./v3.md#三mtp-multi-token-predictionv3-新增顶层结构)

### 2.1 MTP 原生 vs 外挂 draft

| 维度 | **MTP 原生（V3/V4）** | **外挂 draft 小模型** |
|------|----------------------|----------------------|
| Draft 来源 | 同 checkpoint 的 **MTP 辅助头** | **独立**小模型（如 0.6B） |
| 额外显存 | 无第二套权重 | 需同时加载 target + draft |
| 接受率 | MTP 与主模型 **联合训练** 对齐 | 依赖 draft 与 target **分布匹配** |
| 训练目标 | $\mathcal{L}_{\mathrm{main}} + \lambda \mathcal{L}_{\mathrm{MTP}}$ | draft 常单独 SFT / 蒸馏 |

**推理侧循环相同**；差异只在 draft **从哪来**。

> **答疑**：[酱紫君解读：MTP 一次前向与中间 token 融合](../reports/zhihu-jiangzijun-dspark-highlights-20260627.md#mtp一次前向如何融合中间-token) · [融合 scheme 专文](./qa/mtp-fusion-scheme.md)

### 2.2 生产基线 MTP-1

V4 Flash / Pro **预览引擎**在 DSpark 上线前，生产环境采用 **MTP-1**：基于 MTP 的 **单 token 推测解码**基线（每轮 propose/verify 粒度为 1）。**DSpark** 相对 **MTP-1** 报告线上增益（§8），不是相对「无投机纯自回归」。


> **接受率参照**：[ESS 论文梗概](./ess-paper-highlights.md) 的 **MTP-Accept-Ratio**（如 1.7）描述 MTP 投机链路上 **每轮平均接受长度** 的量级，可与下文各草稿范式的块长、接受率对照阅读。

## 3. 草稿范式总览

每轮投机：草稿模型 $M_q$ 一次提出 $K$ 个候选，目标模型 $M_p$ 用时间 $\tau_p$ 做 **1 次**验证；草稿侧总耗时约为 $K \times \tau_q$（$\tau_q$ 为与块长相关的单步 draft 代价，自回归路线下 $\tau_q$ 随 $K$ **线性累积**）。设本轮 **平均接受长度**为 $\mathbb{E}[N_{\mathrm{acc}}]$（$0 \le \mathbb{E}[N_{\mathrm{acc}}] \le K$），相对「每 token 各跑 1 次 $\tau_p$」的粗算加速比为：

$$
S_{\uparrow} = \frac{\bigl(\mathbb{E}[N_{\mathrm{acc}}] + 1\bigr)\,\tau_p}{K\,\tau_q + \tau_p}
$$

> **设计目标**：要把 $S_{\uparrow}$ 做大，需 **同时**抬高 $\mathbb{E}[N_{\mathrm{acc}}]$（draft 猜得准、接受得长）并 **压低** $K\tau_q$（draft 生成别太慢）。两项目标往往拉扯——下文各范式正是在这对矛盾上的不同取舍。

在 **DSpark**（及同期工业界的半自回归路线）成熟之前，外挂草稿 $M_q$ 长期分两派：**Eagle 系**（最新 **Eagle3**，自回归、高接受率）与 **DFlash 系**（并行整块、低 draft 延迟）。DeepSeek 另备 **MTP 原生**头（§2）；**DSpark** 则在两派之间取 **半自回归**折中。

> **自回归 vs 半自回归**：**自回归**（Eagle3）按 token **串行**猜 draft，$\mathbb{E}[N_{\mathrm{acc}}]$ 高但 $K\tau_q$ 随块长 **线性涨**；**半自回归**（DSpark）先 **并行**出整块（压低 $\tau_q$ 侧），再用轻量 **顺序头**逐位补依赖（抬 $\mathbb{E}[N_{\mathrm{acc}}]$ 后缀）——并行拿速度，顺序补准确率。

| 路径 | 代表 | Draft 生成 | 对 $S_{\uparrow}$ 的侧重 |
|------|------|------------|-------------------------|
| **MTP 原生** | V3/V4 MTP 头 | 同模型辅助头，可 MTP-1 单步 | $\mathbb{E}[N_{\mathrm{acc}}]$ 与 target 对齐；无第二套 $M_q$ |
| **自回归外挂** | Eagle3 | 逐 token 串行 | 抬 $\mathbb{E}[N_{\mathrm{acc}}]$；$K\tau_q$ 大 |
| **并行外挂** | DFlash | 一次前向整块 | 压 $K\tau_q$；后缀 $\mathbb{E}[N_{\mathrm{acc}}]$ 易掉 |
| **半自回归外挂** | **DSpark** | 并行主干 + 顺序头 | 两者折中；V4 线上相对 MTP-1 |
| **独立小模型** | Qwen 0.6B draft | 另一套权重 | §3 自测量级参照 |

| 范式 | 优势 | 瓶颈 |
|------|------|------|
| Eagle3 | 块内依赖强、接受率稳 | draft 延迟 $\propto K$（**Memory-Bound 串行**，见 [答疑](./qa/spec-decode-compute-vs-memory-bound.md#3-为何-eagle3-是-memory-bound-串行draft-侧)） |
| DFlash | draft 延迟几乎与 $K$ 弱相关（**Compute-Bound 并行**，见 [答疑](./qa/spec-decode-compute-vs-memory-bound.md#4-为何-dflash-更偏-compute-bound-并行draft-侧)） | 第 2 位起 $\mathbb{E}[N_{\mathrm{acc}}]$ **骤降** |
| DSpark | **一次并行出整块** + 后缀顺序补依赖 | 并行主干仍须生成完整 $K$ 块（§9） |

---

## 4. DSpark 概述

**DSpark**：面向 **V4-Flash / V4-Pro 预览引擎** 高并发 decode。用 **半自回归草稿** + **置信度调度验证**，相对生产基线 **MTP-1**，在 **同等吞吐量** 下单用户生成速度 **+57%–85%**（官方claim区间 **60%–85%**）。论文、训练代码、检查点：[DeepSpec](https://github.com/deepseek-ai/DeepSpec)。

<img src="./figures/dspark-speculative.svg" alt="DSpark：半自回归 draft + 置信度调度验证；与 Eagle3、DFlash、MTP-1 对照" width="920"/>

[图示详情](./figures/dspark-speculative.svg)

---

## 6. DSpark 机制

### 6.1 半自回归候选生成

| 阶段 | 做什么 |
|------|--------|
| **并行主干**（DFlash 改进） | 一次前向 → 整块 hidden + 基础 logits |
| **轻量顺序模块** | 逐 token 注入前缀依赖：**马尔可夫头**（仅 $t{-}1$）或 **RNN 头**（完整前缀状态） |

**2 层 Transformer 深度的 DSpark** 在测试领域 **平均每轮接受长度** 超过 **5 层 DFlash**。

| 位置 | DFlash | Eagle3 | DSpark |
|------|--------|--------|--------|
| 第 1 位 | 高（深并行网） | 受浅网限制 | **第 1 位** 接近深并行网 |
| 第 2 位起 | **快速衰减** | 稳定/上升 | 顺序模块 **缓解衰减** |

#### 为何 draft 上能「堆叠多层」来抬接受率

投机解码的瓶颈在 **块内逐位条件分布** 是否与 target 对齐：第 $i$ 位 token 能否被接受，取决于 draft 提议分布 $q_i(\cdot)$ 能否逼近 target 在 **同一条前缀** 下的 $p_i(\cdot)=p(\cdot\mid\text{context}, x_{<i})$（校验时由 target **整段前向** 得到，是 ground truth）。**对齐的对象是分布 $q_i \approx p_i$，不是单说 hidden 相等。**

外挂 draft（Eagle / DFlash / **DSpark**）的常见做法是：target 先给出（或缓存）某层 **hidden** $h$，再由投机模块 $f_{\mathrm{spec}}$ 叠若干层，必要时注入块内已猜 $x_{<i}$，输出 draft logits $\ell_i=f_{\mathrm{spec}}(h, x_{<i})$ → $q_i=\mathrm{softmax}(\ell_i)$。多一层 Transformer（或等价模块）就多一轮 **$h \to \ell$ 的非线性修正**，让 $q_i$ 更贴近校验时的 $p_i$，单点接受概率 $\uparrow$ → 前缀连乘后存活更长 → §4 的 $\mathbb{E}[N_{\mathrm{acc}}]$ $\uparrow$。

> **易混**：**MTP 原生** draft 不走「外挂 $f_{\mathrm{spec}}$」，而在 **同一 checkpoint** 内用 MTP 块接主模型 hidden 链出 logits（§2）。**独立小模型** draft 甚至不读 target hidden，靠蒸馏/SFT 让整网 $q$ 逼近 $p$（§3）。

关键不在「层数越多越好」，而在 **堆的是什么依赖**：

| 堆法 | 补齐的依赖 | 对 $\mathbb{E}[N_{\mathrm{acc}}]$ 的典型效果 |
|------|------------|---------------------------------------------|
| **加深并行 Transformer**（DFlash） | 各位 **共享** target 前缀，**互不见** 块内已猜 $x_{<i}$ | 第 1 位随深度变强；第 2 位起缺 **块内因果** → 接受率 **快速衰减**；加到 5 层也难救后缀 |
| **串行再算**（Eagle3） | 每步 draft 显式看见 **上一位已猜 token** | 后缀稳定/上升；代价是 $K\tau_q$ 随块长 **线性涨**（§4） |
| **并行主干 + 顺序模块**（DSpark） | **单轮 draft 一次前向并行猜 $K$ 位**（尤其保第 1 位）；马尔可夫/RNN 头 **逐位注入** $x_{i-1}$ | **2 层** 总深度即可超过 **5 层** 纯并行 DFlash（上表实证句） |

<img src="./figures/dspark-semi-ar-draft.svg" alt="DSpark 半自回归候选生成：并行主干一次猜 K 位，顺序头逐位补因果，q 对齐 target 校验 p" width="920"/>

[图示详情](./figures/dspark-semi-ar-draft.svg)

因此 DSpark 的「堆叠」是 **两阶段叠在同一轮 draft**：先用并行主干 **一次前向出整块 $K$ 位**（低 $\tau_q$、第 1 位接近 DFlash），再用极轻的顺序头做 **因果方向的层叠**（等价于在后缀位上补 Eagle 式依赖，而不付满额 $K$ 次串行大前向）。V3 **MTP 因果链**（Main → MTP-1 → MTP-2）则是 target 权重内的另一种「按步数堆模块」：每多一块 MTP 模块，多预测更远 $t{+}k$，推理可作原生 draft（§2）。

> **答疑**：[酱紫君解读：并行主干 vs 顺序头为何 $\tau_q$ 主次分明](../reports/zhihu-jiangzijun-dspark-highlights-20260627.md#dspark-半自回归草稿并行主干-vs-顺序头)

> **文献与对照**：[DSpark_paper.pdf](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf)（Semi-Autoregressive Draft；深度 vs 接受长度消融）· [§2 MTP 因果链](#2-deepseek-路线mtpv3--v4) · [MTP 投机解码总览图](./figures/mtp-speculative.svg) · 加速比读法 [§4](#4-草稿范式总览eagle3--dflash--mtp--dspark) · [酱紫君解读：半自回归 draft](../reports/zhihu-jiangzijun-dspark-highlights-20260627.md#dspark-半自回归草稿并行主干-vs-顺序头)

### 6.2 置信度调度验证

1. 每候选位置输出 **置信度**（此前缀全接受条件下该 token **存活**概率）；
2. 验证集 **温度缩放** 校准置信度 $\approx$ 经验接受率；
3. **硬件感知前缀调度器**：结合并发请求置信度 + **实测吞吐量曲线**，为每请求动态选验证长度，**最大化全局吞吐**。

**负载自适应**：低并发验证 **4–6** token；高并发 **平滑缩短**，避免争用。

> **答疑**：[酱紫君解读：验证截断对效率与准确率的影响](../reports/zhihu-jiangzijun-dspark-highlights-20260627.md#置信度调度与验证截断)

<img src="./figures/dspark-confidence-scheduler.svg" alt="DSpark 置信度调度验证：逐位置信度、温度缩放校准、硬件感知前缀调度与负载自适应" width="920"/>

[图示详情](./figures/dspark-confidence-scheduler.svg)

---

## 7. DSpark 离线基准

**Target**：Qwen3（4B/8B/14B）、Gemma4-12B · **对比**：Eagle3、DFlash
**任务**：数学 / 代码 / 对话（GSM8K、MATH500、AIME25、MBPP、HumanEval、LiveCodeBench、MT-Bench 等）

| 指标 | 结果 |
|------|------|
| 平均每轮接受长度 | DSpark **全面优于** 两基线 |
| Qwen3-4B | vs Eagle3 **+30.9%**，vs DFlash **+16.3%** |

---

## 8. V4 生产部署

| 项 | 配置 |
|----|------|
| 绑定模型 | V4-Flash / V4-Pro **预览引擎** |
| 并行主干 | **3 MoE 层** + **滑动窗口注意力** |
| 最大候选块 | **5**（DSpark-5） |
| 顺序模块 | **马尔可夫头** |
| 对比基线 | **MTP-1** |

---

## 9. 训练与在线引擎

> **范畴**：**DSpark 线上推理**（§8、§10）不改 V4 基座权重，是纯 decode 加速栈。**本节「训练」**指 [DeepSpec](https://github.com/deepseek-ai/DeepSpec) 里 **外挂 draft 模块**（DSpark / DFlash / Eagle3）的 **训练与蒸馏 pipeline**；**「在线引擎」**指该 draft 接入 V4 预览引擎后的调度与 kernel 集成。二者都不是 V3/V4 **主模型预训练**。

> **答疑**：[酱紫君解读：draft 训练 vs 主模型 fine-tune](../reports/zhihu-jiangzijun-dspark-highlights-20260627.md#deepspec-draft-训练-vs-主模型-fine-tune)

### 9.1 训练优化

| 优化 | 说明 |
|------|------|
| 隐状态通信 | 只传 target hidden state，不传词表 logits：$O(V) \to O(d)$ |
| 锚点定长打包 | 多预测块压成密集 batch，减 padding 浪费 |

### 9.2 引擎集成

| 约束 | 对策 |
|------|------|
| CUDA Graph 需提前定 batch 大小 | **异步调度**：截断长度用 **两轮前** 历史置信度预测 |
| 动态变长验证 → kernel padding 浪费 | token **展平** + 稀疏 attention **标记张量** 传依赖；改索引 attention / 压缩 kernel |

---

## 10. 在线生产实测

真实用户流量（2026-06 报道口径）：

### V4-Flash

| SLA（单用户 $\ge$ … token/s） | 聚合吞吐量 vs MTP-1 |
|-------------------------------|---------------------|
| **80** | **+51%** |
| **120**（MTP-1 近边界） | **+661%** |

### V4-Pro

| SLA | 聚合吞吐量 |
|-----|-----------|
| **35 token/s** | **+52%** |
| **50 token/s** | **+406%** |

**同等吞吐量**下单用户速度 **+57%–85%**。

---

## 11. 局限与边界

**局限**：调度器截断后缀后，**并行主干仍生成完整候选块**；低接受率复杂查询的草稿算力 **不可回收**。

| 项 | 说明 |
|----|------|
| **不是新基座** | 与 [V4](./v4.md) 同 checkpoint；`DeepSeek-V4-*-DSpark` = target + 投机模块 |
| **不是 KV / offload** | 与 [ESS](./ess-latent-cache-offload.md)、[Index Share](./index-share.md) **正交** |
| **不是 V3 MTP 论文本身** | MTP 是训练结构；DSpark 是 V4 线上 **独立 draft/verify 栈**，基线 **MTP-1** |

---

## 12. 开源与外部材料

### DeepSpec

| 内容 | 说明 |
|------|------|
| DSpark | `config/dspark/`、评测、检查点 |
| DFlash / Eagle3 | 对照草稿训练评测 |
| 推理 | `inference/` 最小 demo |
| 论文 | [DSpark_paper.pdf](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf) |
| HF | [DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark) |

### 第三方解读

| 材料 | 说明 |
|------|------|
| [酱紫君（GalAster）：DSpark 与投机解码](../reports/zhihu-jiangzijun-dspark-highlights-20260627.md) | 投机背景、半自回归 draft、验证截断、MTP 融合、draft 训练 vs fine-tune；[知乎原文](https://www.zhihu.com/question/2054255700407055156/answer/2054537823915619954) |
| [批量验证机制笔记](../wiki/投机解码之批量验证.md) | 目标模型单次前向并行验 K 个 draft token |

---

## 13. 在全景中的位置

| 层级 | 投机解码 / DSpark |
|------|-------------------|
| 模型结构 | MTP 属 **[架构-train](../reports/deepseek-version-lineage-20260625.md#优化方向分类)** |
| Decode infra | DSpark、MTP 投机调度 → **100% 架构-infer** |
| KV / 长上下文 | 否 → [基础设施线](../reports/deepseek-infra-line.md) |

---

## 14. 参考

1. DeepSeek-AI & Peking University. *DSpark* — [DeepSpec](https://github.com/deepseek-ai/DeepSpec)
2. DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437 — MTP 训练与推理复用
3. DeepSeek-AI. *DeepSeek-V4.* arXiv:2606.19348
4. Leviathan, Y., Kalman, M., & Matias, Y. Fast inference from transformers via speculative decoding. 2022.
5. IT之家. *DeepSeek 联合北大发布 DSpark.* 2026-06-27
