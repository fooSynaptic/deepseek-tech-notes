# DeepSeek-LLM V1：BBPE 词表与 Tokenizer

> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · [← V1 正文](./v1.md) · [← 演进总览 §3.1 V1](../reports/deepseek-version-lineage-20260625.md#31-deepseek-llm-v1) · [V1→V3 演进](../reports/deepseek-v1-to-v3-lineage.md)
> **论文**：[DeepSeek-LLM arXiv:2401.02954](https://arxiv.org/abs/2401.02954) §2.1 · [V2 沿用同词表](./v2.md)

## 核心结论摘要

- V1 采用 **Byte-level BPE (BBPE)** 词表：100K + special tokens。
- Embedding 维度 102,400；V2/V3 系 **沿用同一词表**。
- 预分词规则与 byte fallback 影响多语言与代码 token 效率。
- 是理解后续 checkpoint 词表一致性的前置阅读。

---

## 一句话

DeepSeek-LLM V1 采用 **BBPE（Byte-level Byte-Pair Encoding）**：在 UTF-8 字节序列上做 BPE，配合 **字符类预分词** 与 **数字单 digit 拆分**；常规词表 **100,000**，加 **15** 个 special token 后 **100,015**，模型 embedding 维配置 **102,400** 以预留扩展位。

---

<a id="bpe-primer"></a>

## BPE 简述

**Byte Pair Encoding（BPE）** 是 subword 分词里最常用的训练式词表构造法之一（Sennrich et al., 2016；GPT-2 / RoBERTa 系广泛采用）。相对词级分词，它在 **词表大小** 与 **序列长度 / OOV** 之间折中：常见片段共享 id，罕见词由若干 subword 拼成，无需 `<unk>`。

**训练**：

1. 初始符号表 = 基础单元（字符、或 **字节** —— 见下节 BBPE）
2. 重复：统计当前序列中 **相邻 symbol pair** 的频率，将最高频 pair 合并为新 symbol，写入词表
3. 达到目标 merge 次数或词表大小 $V$ 后停止

**编码**：对输入做同样的预切分后，按 merge 优先级 **贪心最长匹配**（或等价 trie 查表），得到 token id 序列。解码为 merge 的逆过程：id → subword 字符串 → 拼接还原文本。

对照时更值得看的是两轴：**词表大小怎么定**，以及 **OOV 能压到什么程度**（下表后两列）。

| 方法 | 与 BPE 的关系 | 典型场景 | 词表大小 | OOV |
|------|---------------|----------|----------|-----|
| **BPE / BBPE** | 频率驱动 merge | GPT-2、LLaMA、DeepSeek V1 | 目标 $V$ 由 **merge 次数** 定；自底向上从基础单元「长」到 $V$（BBPE 底为 **256 bytes**；V1 常规 **100K**） | 纯字符 BPE 仍可能对未见字符 OOV；**BBPE 有字节兜底 → 实质无 OOV**（罕见片段拆成更多 token，序列变长而非 `<unk>`） |
| **WordPiece** | 选 merge 时最大化语言模型似然 | BERT | 同样以目标 $V$ 为停、自底向上；BERT 系常见 **~30K** | 未知片段尽量 `##` 子词切；仍切不动则 **`[UNK]`**，OOV 风险 **高于 BBPE** |
| **Unigram LM** | 从大词表 **删** 子词（SentencePiece） | T5、部分多语模型 | 先造大候选集再 **剪到** $V$（自上而下）；多语常配更大 $V$ | 通常靠 **字符覆盖** 压低 OOV；无 byte fallback 时罕见字符仍可能 UNK；配 byte fallback 则接近 BBPE |

**小结**：三者都能把 $V$ 调到目标规模；差别在构造方向（BPE/WordPiece **加** merge vs Unigram **减** 子词）。OOV 上 **BBPE 最硬**（字节闭包），WordPiece 最依赖词表内字符，Unigram 介于「字符覆盖」与「可选 byte fallback」之间。

对已有 NLP 基础的读者：BPE 不建模语义，只压缩词表；**预分词（pre-tokenization）** 规则（空格、标点、CJK 边界等）往往比 merge 算法本身更影响中英混排质量。DeepSeek V1 在标准 BPE 之上把基础单元降到 **UTF-8 字节**，并约束字符类不跨类 merge —— 即下文 [§2 BBPE](#bbpe-overview)。

---

## 1. 与预训练数据 pipeline 的关系

Tokenizer 训练语料来自 V1 **三阶段数据 pipeline** 的多语言子集（与 [V1 §2.1 数据](./v1.md#21-数据) / [#tokenizer](./v1.md#tokenizer) 同一论文段落）：

<img src="../figures/v1/bbpe/v1-data-pipeline.svg" alt="原始语料 → 去重 → 过滤 → 重混 → BBPE 训练 / 预训练" width="920"/>

[图示详情](../figures/v1/bbpe/v1-data-pipeline.svg)

| 阶段 | 与 Tokenizer 相关的要点 |
|------|-------------------------|
| **去重** | 跨 Common Crawl **91 个 dump** 全局去重，去重率 **89.8%**（单 dump 内仅 22.2%） |
| **过滤** | 语言 + 语义质量 |
| **重混** | 提升低占比 domain，保证多语言多样性 |

BBPE 本身在约 **24GB 多语言语料** 上训练 merge 规则；**2.0T tokens** 的预训练则走完整 pipeline 后再分词。

---

<a id="bbpe-overview"></a>

## 2. BBPE 是什么

**BBPE** = 在 **字节（byte）序列** 上运行 BPE，而非 Unicode 字符或 SentencePiece 的 pretokenized 单元。

| 对比 | 典型做法 | V1 BBPE |
|------|----------|---------|
| 词表基础 | 字符 / 子词 | **UTF-8 字节** → 合并为 subword |
| OOV | 需 unk 或 fallback | 任意 UTF-8 文本可 **字节级覆盖** |
| 实现 | 多种 | HuggingFace **tokenizers** 库 |

直觉：先 `text → bytes`，再在 byte 序列上统计 pair 频率、迭代 merge，得到 subword 词表。

<img src="../figures/v1/bbpe/bbpe-process-example.svg" alt="BBPE 训练过程示例：UTF-8 字节、迭代 merge、编码" width="920"/>

[图示详情](../figures/v1/bbpe/bbpe-process-example.svg)

**示意说明**：预分词后文本变为 UTF-8 字节流；初始符号表为 256 个字节；每轮合并语料中最高频 **相邻 byte pair**，写入 merge 表；重复至 **100,000** 次 merge。推理时对同样预切分后的字节序列 **贪心最长匹配**。因任意 UTF-8 文本可先落到字节层，**不会出现字符级 OOV**（罕见片段至多拆成更多 byte/subword token，序列变长而非 `<unk>`）。

> **图上 `48 69`、`E4 B8 96` 是十六进制记法**，不是单独一套「16 进制词表」。实现里每个符号是 **8-bit 字节** $b \in \{0,\ldots,255\}$（即 `0x00`–`0xFF`）；`48` = 十进制 72 = `'H'`，`69` = `'i'`。`E4 B8 96` 等为 **中文等 CJK 字符**在 UTF-8 下的 3 字节编码示意。HuggingFace tokenizers 在内部用 `u8` 序列做 merge/encode。

<a id="pre-tokenization"></a>

## 3. 预分词规则

论文在 BPE 合并前做 **pre-tokenization**，限制 **不同字符类之间不得跨类 merge**（做法类似 GPT-2）：

| 规则 | 目的 |
|------|------|
| **换行 / 标点 / CJK** 等分属不同字符类 | 避免中文与标点、换行被错误粘成单一 token |
| **数字拆成单 digit**（同 LLaMA 系列） | 稳定算术、代码、编号类文本的 token 边界 |

效果：中英双语 + 代码混排时，CJK 与 Latin 边界清晰，数字可组合性更好。

---

<a id="vocab-size"></a>

## 4. 词表规模与 embedding 配置

| 项 | 数值 | 说明 |
|----|------|------|
| 常规 merge tokens | **100,000** | BPE 训练目标词表大小 |
| Special tokens | **+15** | 如 BOS/EOS/PAD 及任务相关控制符 |
| Tokenizer 总大小 | **100,015** | 100,000 + 15 |
| **模型 `vocab_size`（embedding 行数）** | **102,400** | 训练与推理配置；**预留**未来 special / 扩展 token |

> **易混点**：HF `config.json` 里的 `vocab_size=102400` 是 **embedding 矩阵行数**；有效 token id 仍以 tokenizer 的 100,015 为准，未用 id 对应 embedding 行可视为预留。

---

<a id="encode-flow"></a>

## 5. 编码流程

<img src="../figures/v1/bbpe/bbpe-encode-flow.svg" alt="UTF-8 文本 → pre-tokenization → byte → BBPE → token id → embedding" width="720"/>

[图示详情](../figures/v1/bbpe/bbpe-encode-flow.svg)

与后续 **MLA / MoE / DSA** 正交：改 attention 或 FFN **不改变** tokenizer；V1 Chat 的 SFT/DPO 也在同一词表上继续。

---

<a id="lineage"></a>

## 6. 在 DeepSeek 系列中的延续

| 版本 | Tokenizer 相对 V1 |
|------|-------------------|
| **V1（7B/67B）** | **定义 BBPE 词表**（本文） |
| **V2** | **沿用 DeepSeek 67B 同一 tokenizer**（[2405.04434](https://arxiv.org/abs/2405.04434)）；8.1T 预训练 |
| **V3** | 仍 BBPE 族，**扩展至 128K** 词表；pretokenizer 与训练语料针对多语言压缩优化（[2412.19437](https://arxiv.org/abs/2412.19437) §4.1） |
| **V3.2 / R1** | 与 V3 **同词表** |
| **V4** | 随新旗舰再扩展（[DeepSeek-V4 梗概 · 训练章节](./v4.md#训练要点)） |

V1 的 **102,400 embedding 预留** 与 V3 的 **128K 扩展** 是不同代际决策；读 checkpoint 时须对齐 `config.vocab_size` 与 tokenizer 文件。

---

## 7. 推理 infra 关注点

| 项 | V1 |
|----|-----|
| 与 MLA 兼容性 | V1 为 **标准 MHA/GQA**，无 MLA；词表独立问题 |
| 引擎配置 | 常规 Transformers / vLLM；加载 `deepseek-llm-7b-base` 等官方权重即可 |
| 上下文 | **4K**；tokenizer 无 RoPE，位置在模型侧 |
| 续训 / 扩词表 | 若新增 token，须 **扩 embedding + lm_head** 并重训或至少微调新增行 |

---

## 参考

1. DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* arXiv:2401.02954, 2024. — §2.1 Data, Tokenizer; Table 1 dedup.
2. DeepSeek-V2：同 tokenizer（§2 / Appendix 数据节）。
3. 本地精读：[V1 §2.1 Tokenizer](./v1.md#tokenizer) · [technical-highlights（复现向）](./v1-bbpe-tokenizer.md)
