# DeepSeek V1 → V2 → V3：前代到旗舰基座

> 更新：2026-06-25
> [← 全系列演进总览](./deepseek-version-lineage-20260625.md) · [V1 正文](../versions/v1.md) · [V2](../versions/v2.md) · [V3](../versions/v3.md)

---

## 1. 三代在系列中的位置

DeepSeek 开源主线可粗分为两段：

1. **V1 → V2 → V3（2024）**：从稠密双语基座，到 **MLA + MoE** 效率架构，再到 **规模化旗舰 MoE**（671B）
2. **V3.1 → V3.2 → V4（2025–2026）**：在同一 V3 权重架构上 post-train、加 **DSA**、再 **[架构大步进](./deepseek-version-lineage-20260625.md#2-版本时间线与关系)**

本文梳理第一段：**V1 → V2 → V3**。

---

## 2. 对照总表

| 版本 | 时间 | 机构 | arXiv | 总参 / 激活 | 上下文 | 注意力 | FFN | 预训练 |
|------|------|------|-------|-------------|--------|--------|-----|--------|
| **DeepSeek-LLM V1** | 2024-01 | DeepSeek | [2401.02954](https://arxiv.org/abs/2401.02954) | 7B / 7B；67B / 67B | 4K | MHA / **GQA** | **稠密** SwiGLU | **2T** |
| **DeepSeek-V2** | 2024-05 | DeepSeek | [2405.04434](https://arxiv.org/abs/2405.04434) | 236B / **21B** | **128K** | **MLA** | **DeepSeekMoE**（6 routed + shared） | **8.1T** |
| **DeepSeek-V3** | 2024-12 | DeepSeek | [2412.19437](https://arxiv.org/abs/2412.19437) | 671B / **37B** | 128K | **MLA**（同 V2 族） | MoE **256 / 8 act** + **aux-loss-free** | **14.8T** |

---

## 3. 演进逻辑

### 3.1 注意力：标准 GQA → MLA

<img src="../../diagrams/v1-v3-mla-evolution.svg" alt="V1 GQA → V2/V3 MLA latent KV 压缩" width="920"/>

[图示详情](../../diagrams/v1-v3-mla-evolution.svg)

- **V2 首创 MLA**（[2405.04434](https://arxiv.org/abs/2405.04434)）；**V3 沿用**同一 latent 格式（[MLA 详解](../versions/mla-latent-attention.md)）
- V3.1 再在 Prefill/Decode 间切换 MHA/MQA 模式；V3.2 叠加 DSA — 均属 **V3 代之后**，不在 V1–V3 段

### 3.2 FFN：稠密 → MoE → 大规模 aux-loss-free MoE

| 代际 | 做法 |
|------|------|
| **V1** | 全参数激活；67B 用 **加深（95 层）** 而非单纯加宽 FFN |
| **V2** | **[DeepSeekMoE](../versions/deepseek-moe.md)**：160 routed，每 token 6 个 + shared；稀疏激活降训练/推理 FFN 成本（[MoE 线 §②](deepseek-moe-line.md)） |
| **V3** | 扩到 **256 experts / 8 activated**；路由改为 **sigmoid + 动态 bias**（[aux-loss-free](../versions/aux-loss-free-moe-routing.md)），并加 **MTP** 辅助头 |

### 3.3 规模与数据：scaling laws → 产品化旗舰

| 代际 | 训练叙事 |
|------|----------|
| **V1** | 系统研究 **IsoFLOP / batch-LR scaling**；7B/67B 同训 **2T** 双语语料 |
| **V2** | **8.1T** 多源语料；证明 21B 激活可打过 67B 稠密 |
| **V3** | **14.8T** + 完整后训练 pipeline；671B 成为 **R1 / V3.1 / V3.2** 的共同架构母版 |

---

## 4. 能力代际

<img src="../../diagrams/v1-v3-capability-timeline.svg" alt="V1 → V2 → V3 能力代际及 R1 / V3.1 / V3.2 分叉" width="920"/>

[图示详情](../../diagrams/v1-v3-capability-timeline.svg)

---

## 5. 推理 infra 代际差异

| 维度 | V1 | V2 | V3 |
|------|----|----|-----|
| KV 格式 | 标准 GQA/MHA | **MLA latent** | **MLA latent**（同 V2） |
| 引擎适配 | 通用 HF/vLLM | 需 MLA / MoE 定制 | FlashMLA、DeepGEMM、`block-size=1` |
| 长上下文瓶颈 | 4K 上限 | 128K latent 线性涨 | 同左；V3.2 才拆 Indexer/Latent |

---

## 6. 阅读顺序

1. [V1 正文](../versions/v1.md)（DeepSeek-LLM 完整译文）
2. [DeepSeekMoE 架构](../versions/deepseek-moe.md) · [V2 梗概](../versions/v2.md) · [MLA 前向流程图](../versions/mla-latent-attention.md#forward-flow)
3. [V3 梗概](../versions/v3.md) · [演进总览 §3](./deepseek-version-lineage-20260625.md#3-各版本详解)
4. 后续代际：[R1](../versions/r1.md) → [V3.1](../versions/v3-1.md) → [V3.2](../versions/v3-2.md) → [V4](../versions/v4.md)

---

## 7. 参考

- [V1 BBPE 词表专文](../versions/v1-bbpe-tokenizer.md)
- [V1 技术报告精读 stub](./deepseek-llm-v1-highlights.md)

1. DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* arXiv:2401.02954, 2024.
2. DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* arXiv:2405.04434, 2024.
3. DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024.
