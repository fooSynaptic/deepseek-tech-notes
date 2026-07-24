# DeepSeek-V2 梗概

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §3.2](../01-总览/01-版本演进总览.md#32-deepseek-v2) · [← MoE 线导读](../01-总览/07-MoE线导读.md) · [← 版本目录](../01-总览/02-版本梗概索引.md) · [V1→V3 演进](../01-总览/04-V1到V3演进.md) · [MLA 详解](../02-基座架构/02-MLA低秩注意力.md) · [V1 BBPE 词表](00-V1-BBPE词表与Tokenizer.md) · [Raschka 解读](../08-外部解读/01-Raschka要点速读.md#mla-要点31)

## 核心结论摘要

- V2（2024-05）是架构代际跃迁：首次引入 **MLA** 与 **DeepSeekMoE**。
- 236B total / 21B activated，128K 上下文，8.1T 预训练 tokens。
- MLA latent KV 将 cache 体积降至标准 GQA 的约 **6.7%**。
- 相对 67B 稠密：训练成本 **-42.5%**、生成吞吐 **5.76×**。

---

## 定位

2024 年 5 月发布。相对 **DeepSeek-LLM 67B 稠密**，V2 是架构**代际跃迁**：首次引入 **[MLA](../02-基座架构/02-MLA低秩注意力.md)** 与 **[DeepSeekMoE](../02-基座架构/05-DeepSeekMoE.md)**，236B 总参、每 token 激活 **21B**，**128K** 上下文。论文称相对 67B 稠密：训练成本 **-42.5%**、KV cache **-93.3%**、生成吞吐 **5.76×**。

## 核心架构

| 组件 | 要点 |
|------|------|
| **[MLA](../02-基座架构/02-MLA低秩注意力.md)** | K/V **低秩 latent 联合压缩**进 cache；V3/R1/V3.1/V3.2 **沿用同一 MLA 结构** |
| **[DeepSeekMoE](../02-基座架构/05-DeepSeekMoE.md)** | 每层 **160 routed + 2 shared experts**，每 token 激活 **6** 个 routed |
| **规模** | 236B total / **21B activated** |
| **上下文** | **128K**（Lite 版 16B / 2.4B act，32K） |
| **预训练** | **8.1T** tokens |
| **Tokenizer** | **沿用 V1 DeepSeek 67B 同一 [BBPE 词表](00-V1-BBPE词表与Tokenizer.md)**（100K + special，embedding 102,400） |
| **后训练** | SFT + RL → Chat 版 |

## 相对 V1 的关键变化

| 维度 | V1（67B 稠密） | V2 |
|------|----------------|-----|
| FFN | 稠密 SwiGLU | **MoE 稀疏激活** |
| 注意力 | GQA（8 KV 头） | **MLA latent KV** |
| 上下文 | 4K | **128K** |
| 预训练 | 2T | **8.1T** |
| KV cache | 标准 GQA | **~6.7%** 体积 |

## 推理 infra 关注点

- KV cache 变为 **MLA latent 格式**（后续 V3 系继承）
- 需自定义 kernel / vLLM 适配（DeepSeek 后续提供 FlashMLA 等）
- MoE 路由为 **softmax 系**（V3 改为 **aux-loss-free sigmoid 路由**，见 [aux-loss-free](../02-基座架构/03-aux-loss-free-MoE路由.md)）

| 方向 | 文档 |
|------|------|
| **MoE 架构** | [DeepSeekMoE 详解](../02-基座架构/05-DeepSeekMoE.md) |
| **本版本** | V2 为 DeepSeekMoE **首发落地**（见上表配置） |
| **下游 ③ aux-loss-free** | [aux-loss-free MoE 路由](../02-基座架构/03-aux-loss-free-MoE路由.md) · [DeepSeek-V3](../02-基座架构/01-V3基座.md) |

---

## 上下游

| 方向 | 关系 |
|------|------|
| 上游 | [DeepSeek-LLM V1](00-V1-LLM.md)（稠密 + scaling laws） |
| 下游 | **[DeepSeek-V3](../02-基座架构/01-V3基座.md)**：671B / 37B act、256 experts / 8 act、MTP、aux-loss-free、14.8T |

## 参考

- 论文：[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- 仓库：[deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)
- MLA 前向：[MLA前向计算流程§流程图](../02-基座架构/02-MLA低秩注意力.md#forward-flow) · [公式详解](../02-基座架构/02-MLA低秩注意力.md#forward-flow)