# DeepSeek 技术报告

> **《ds-技术报告》** — `deepseek-mechanism-atlas` 文档的书籍式读本（**独立目录**，与仓库 `docs/` 并行）。  
> [中文导读](./00-前言/02-中文导读.md) · **从这里开始**：[01 版本演进总览](./01-总览/01-版本演进总览.md)（全书主线；其余章节经该文内链展开）

---

## 全书目录

### 01 总览

| 章 | 内容 |
|----|------|
| [01 版本演进总览](./01-总览/01-版本演进总览.md) | 全系列算法线 + infra 线全景（主线） |
| [05 算法线导读](./01-总览/05-算法线导读.md) | MLA → DSA → CSA/HCA + mHC 专题 |
| [06 基础设施线导读](./01-总览/06-基础设施线导读.md) | MLA KV → 异构 Cache → Index Share → ESS → V4 HiSparse |
| [07 MoE 线导读](./01-总览/07-MoE线导读.md) | 稠密 FFN → DeepSeekMoE → aux-loss-free → Hash MoE |
| [04 V1→V3 演进](./01-总览/04-V1到V3演进.md) | 稠密 → MLA+MoE → 671B 旗舰 |
| [02 版本梗概索引](./01-总览/02-版本梗概索引.md) | 各版本一页纸索引 |
| [03 技术报告索引](./01-总览/03-技术报告索引.md) | 报告与外部解读目录 |

### 02 基座架构

| 章 | 内容 |
|----|------|
| [01 V3 基座](./02-基座架构/01-V3基座.md) | 671B MoE + MLA + MTP |
| [02 MLA 低秩注意力](./02-基座架构/02-MLA低秩注意力.md) | Eq. 37–47、KV 压缩 |
| [05 DeepSeekMoE](./02-基座架构/05-DeepSeekMoE.md) | 细粒度 routed + shared experts |
| [03 aux-loss-free MoE 路由](./02-基座架构/03-aux-loss-free-MoE路由.md) | 动态 bias 负载均衡 |
| [04 序列均衡损失](./02-基座架构/04-序列均衡损失.md) | $L_{\mathrm{Bal}}$ 互补均衡 |
| [06 V3 FP8 动态量化](./02-基座架构/06-V3-FP8动态量化.md) | FP8 训练量化专文 |

### 03 后训练与 R1

| 章 | 内容 |
|----|------|
| [01 RLVR](./03-后训练与R1/01-RLVR.md) | 可验证奖励 + GRPO |
| [02 R1](./03-后训练与R1/02-R1.md) | V3-Base + RLVR |
| [03 RL 笔记索引](./03-后训练与R1/03-RL笔记索引.md) | 后训练延伸阅读 |
| [04 GRPO 长程局限](./03-后训练与R1/04-GRPO长程局限.md) | 社区讨论附录 |
| [05 R1 训练 pipeline](./03-后训练与R1/05-R1训练pipeline.md) | 四阶段 Dev-1→R1（参考材料副本） |

### 04 版本代际

| 章 | 内容 |
|----|------|
| [00 V1 LLM](./04-版本代际/00-V1-LLM.md) | 2401.02954 完整中文译文 + Figure 2–5 / Table 3–4 |
| [00 V1 BBPE 词表](./04-版本代际/00-V1-BBPE词表与Tokenizer.md) | Byte-level BPE、预分词、102,400 embedding |
| [00 V2 MoE 与 MLA](./04-版本代际/00-V2-MoE与MLA.md) | MLA + DeepSeekMoE 首次引入 |
| [01 V3.1-Terminus](./04-版本代际/01-V3.1-Terminus.md) | Hybrid 推理、128K |
| [02 V3.2-DSA](./04-版本代际/02-V3.2-DSA.md) | 稀疏注意力正式版 |
| [03 V4](./04-版本代际/03-V4.md) | V4-Pro / V4-Flash 梗概，1M context |
| [05 CSA/HCA 混合压缩注意力](./04-版本代际/05-CSA-HCA混合压缩注意力.md) | 4:1 稀疏 + 128:1 dense；算法线 ③ |
| [04b Hyper-Connections](./04-版本代际/04b-Hyper-Connections.md) | HC 多路残差流（mHC 前置） |
| [04 mHC 流形约束超连接](./04-版本代际/04-mHC流形约束超连接.md) | 双随机流形约束；V4 落地 |
| [06 Hash MoE + FP4](./04-版本代际/06-Hash-MoE-FP4.md) | Hash 路由 + FP4 量化；MoE 线 ⑤ |

### 05 DSA 稀疏注意力

| 章 | 内容 |
|----|------|
| [01 系列导读](./05-DSA稀疏注意力/01-系列导读.md) | 阅读顺序与示意图 |
| [02 DSA 梗概](./05-DSA稀疏注意力/02-DSA梗概.md) | 三阶段 + 异构 Cache |
| [03 DSA 逻辑详解](./05-DSA稀疏注意力/03-DSA逻辑详解.md) | 算法深度 |
| [04 Lightning Indexer 详解](./05-DSA稀疏注意力/04-Lightning-Indexer详解.md) | 打分公式、Indexer-Cache |
| [05 Index Share 梗概](./05-DSA稀疏注意力/05-Index-Share梗概.md) | IndexCache infra |
| [06 Index Share 逻辑](./05-DSA稀疏注意力/06-Index-Share逻辑.md) | FFFS 跨层复用 |

### 06 推理基础设施

| 章 | 内容 |
|----|------|
| [01 ESS 概念](./06-推理基础设施/01-ESS概念.md) | Latent-Cache offload |
| [02 ESS 论文梗概](./06-推理基础设施/02-ESS论文梗概.md) | Fig.1–9 / Table 1–2 |
| [03 投机解码自测](./06-推理基础设施/03-投机解码自测加速比.md) | 外挂 draft 加速比参考 |
| [04 DSpark 投机解码](./06-推理基础设施/04-DSpark投机解码.md) | MTP + DSpark 专文 |
| [05 V4 KV Layout](./06-推理基础设施/05-V4-KV-Layout.md) | 异构 KV 布局 |
| [06 V4 HiSparse](./06-推理基础设施/06-V4-HiSparse.md) | 稀疏推理栈 |
| [07 V4 磁盘 Prefix Cache](./06-推理基础设施/07-V4-磁盘Prefix-Cache.md) | 磁盘前缀缓存 |

### 07 Engram

| 章 | 内容 |
|----|------|
| [01 Engram 官方 README](./07-Engram/01-Engram官方README.md) | 条件记忆 / n-gram 稀疏轴 |
| [02 Engram 系列导读](./07-Engram/02-Engram系列导读.md) | CXL / Nine / Tiny 深度笔记（副本） |

### 08 外部解读

| 章 | 内容 |
|----|------|
| [01 Raschka 要点速读](./08-外部解读/01-Raschka要点速读.md) | V3→V3.2 一文要点 |
| [02 Raschka 全文解析](./08-外部解读/02-Raschka全文解析.md) | 分章对照 |
| [03 酱紫君 DSpark 解读](./08-外部解读/03-酱紫君DSpark阅读笔记.md) | GalAster：投机解码、半自回归、验证截断、MTP、draft 训练 |

### 09 附录

| 章 | 内容 |
|----|------|
| [01 开发索引](./09-附录/01-开发索引.md) | 源路径与构建（**非阅读入口**） |
| [02 文档系列结构审查](./09-附录/02-文档系列结构审查.md) | 双向引用、导航、SVG 复用审计 |
| [material/](./09-附录/material/README.md) | 补充参考材料（本仓内）（R1 pipeline、Engram 导读、V1 Wiki 等） |

---

## 阅读入口

**[01 版本演进总览](./01-总览/01-版本演进总览.md)** — 推荐唯一入口；三线导读、版本表与各卷跳转均在该章维护。

<details>
<summary>推荐阅读顺序（需要时展开）</summary>

1. **01 版本演进总览** — 全系列地图  
2. **04 版本代际** + **02 基座架构** — 前代与 V3 底座  
3. **03 后训练与 R1** — 推理模型  
4. **05 DSA** + **06 ESS** — V3.2 算法与 infra  
5. **07 Engram** · **08 外部解读** — 专题与对照

</details>

---

## 维护说明

```bash
# 在 deepseek-mechanism-atlas 根目录执行（仅复制整理，不改原文件）
python3 《ds-技术报告》/build_book.py
```

原仓库路径对照见 `build_book.py` 内 `CHAPTER_MAP` / `ASSET_MAP`。
