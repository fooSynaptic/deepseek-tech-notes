# DeepSeek 文档系列结构审查

> [← 演进总览](./deepseek-version-lineage-20260625.md) · [← 开发索引](../WIKI-INDEX.md) · [《ds-技术报告》](../../《ds-技术报告》/README.md) · [书中总览](../../《ds-技术报告》/01-总览/01-版本演进总览.md)

本报告对 `deepseek-mechanism-atlas` 仓库文档系列做**结构、双向引用、章节导航、概念去重、SVG 复用**审查，并记录已落地修复与后续维护命令。

---

## 1. 审查范围

| 层级 | 路径 |
|------|------|
| **总览 hub** | [版本演进总览](./deepseek-version-lineage-20260625.md) |
| 三线导读 | [算法线](./deepseek-algorithm-line.md) · [基础设施线](./deepseek-infra-line.md) · [MoE 线](./deepseek-moe-line.md) |
| 版本梗概 | [版本梗概索引](../versions/README.md) |
| 专题卷 | [DeepSeek DSA 与 Index Share 系列](../dsa/README.md) · [RL / 后训练笔记](../rl/README.md) · [Engram 系列](../engram/README.md) |
| 答疑 | [答疑索引](../versions/qa/README.md) |
| 成书 | [《ds-技术报告》/build_book.py](../../《ds-技术报告》/build_book.py) |

---

## 2. 入口与阅读顺序

**唯一主入口**：演进总览 → 按 §3 版本节 / §7 专题表 / 三线导读分叉。

**全书线性顺序**由 `build_book.py` 的 `READING_ORDER` 定义；`append_chapter_nav()` 在每章文末追加「上一章 / 下一章」表。2026-06-27 已确认 **FP8 专文**在链中：

```text
… → 04-序列均衡损失 → 06-V3-FP8动态量化 → 03-后训练 RLVR → …
```

维护：改 `READING_ORDER` 后须重跑 `python3 《ds-技术报告》/build_book.py`，否则书中导航会跳过新章（本次已手工校正 `04-序列均衡损失` / `06-V3-FP8` / `01-RLVR` 三处页脚）。

---

## 3. 双向引用

约定：每篇正文**首段 blockquote** 含 `← 演进总览 §x.x` 或 `← 系列目录`，并链到书中对应章。

### 3.1 已修复 / 已对齐

| 文档 | 状态 |
|------|------|
| [../rl/README.md](../rl/README.md) | blockquote → 演进总览 §3.4 |
| [../rl/optimize.md](../rl/optimize.md) | 补标题 + blockquote → RLVR / §3.4 / 书中 GRPO 章 |
| [../dsa/README.md](../dsa/README.md) | blockquote 增 §3.6 V3.2 |
| [../dsa/dsa-logic.md](../dsa/dsa-logic.md) · [Index Share 逻辑详解](../dsa/index-share-logic.md) | 顶栏改为 blockquote + §3.6 锚点 |
| [投机解码自测加速比](./spec-decode-draft-acceleration-20260604.md) | 顶栏合并为 blockquote |
| [Engram 系列导读](../engram/README.md) | hr 下 blockquote → §7 专题关系 |
| [答疑索引](../versions/qa/README.md) | blockquote → 梗概索引 / 演进总览 |
| [v1-technical-report.zh.md](../versions/v1-technical-report.zh.md) · [deepseek-llm-v1-highlights.md](./deepseek-llm-v1-highlights.md) | stub 页补 §3.1 回链 |

### 3.2 反向引用

演进总览 §3 各版本节已链到对应梗概 / 专文（含 [FP8 动态量化 §3.3](../versions/v3-fp8-dynamic-quantization.md)）。§7 专题表链 Engram / R1 pipeline 等。

**建议**：§7 可增补 DSA 系列、[ESS 概念](../versions/ess-latent-cache-offload.md)、[Index Share](../versions/index-share.md) 一行，与三线导读 §5 反向表完全对称。

---

## 4. 概念与文档去重

| 概念 | 唯一主文档 | 摘要嵌入 | 答疑 |
|------|-----------|----------|------|
| FP8 训练量化 | [V3 FP8 动态量化](../versions/v3-fp8-dynamic-quantization.md) | [v3.md §附 FP8](../versions/v3.md) 同图同链 | [fp8-partial-sum-drift.md](../versions/qa/fp8-partial-sum-drift.md)；[fp8-mma-term.md](../versions/qa/fp8-mma-term.md) |
| GRPO vs PPO | [rlvr.md#grpo](../versions/rlvr.md) | 演进总览 §3.4 同图 | — |
| MTP / 投机 | [v3.md §三](../versions/v3.md) | [spec-decode 报告](./spec-decode-draft-acceleration-20260604.md) 复用 `mtp-speculative.svg` | — |
| DSA pipeline | [DSA 逻辑详解](../dsa/dsa-logic.md) | 梗概 / Raschka 链同一 `dsa-pipeline.svg` | — |
| ESS 双 Cache | [ESS Latent offload](../versions/ess-latent-cache-offload.md) | DSA 系列 / Lightning Indexer **同一路径** | [h2d-d2h-pcie-transfer.md](../versions/qa/h2d-d2h-pcie-transfer.md) |
| V1 译文 | **已并入** [DeepSeek-LLM V1](../versions/v1.md) | stub 页仅重定向 | — |

**未发现**：同一概念两篇并列「概览」；`deepseek-v1-to-v3-lineage.md` 与演进总览分工明确（前者 V1→V3 纵切，后者全系列）。

---

## 5. SVG 规范与 canonical 路径

生成器：`scripts/svg/gen_*.py`、`dsa/scripts/svg/gen_dsa_svgs.py`；校验：`scripts/svg/check_svgs.py`（含 Markdown `<img>` 嵌入与布局启发式）。

### 5.1 复用表

| SVG | canonical 源 | 复用位置 |
|-----|-------------|----------|
| `dsa-pipeline.svg` | `../dsa/diagrams/dsa-pipeline.svg` | dsa 系列、梗概、Raschka |
| `ess-dual-cache.svg` | `../dsa/diagrams/ess-dual-cache.svg` | ESS 概念、Lightning Indexer、dsa/README（**勿再用** `docs/figures/ess/ess-dual-cache.svg`） |
| `index-share-fffs.svg` | `../dsa/diagrams/index-share-fffs.svg` | Index Share 逻辑 |
| `v3-moe-vs-v2.svg` | `docs/figures/v3/`（与 `diagrams/` 同步） | v3 梗概、演进总览 §3.3 |
| `grpo-vs-ppo.svg` | `docs/figures/rl/` | rlvr、演进总览 §3.4 |
| `mla-mode-switch.svg` | `docs/figures/v3/` | v3-1、演进总览 §3.5 |
| `mtp-speculative.svg` | `docs/figures/v3/` | v3、spec-decode 报告 |
| `v3-fp8-dynamic-quant.svg` | `docs/figures/v3/` | FP8 专文、v3 摘要 |
| `deepseek-version-lineage.svg` | `diagrams/` | 演进总览 |
| `mla-forward-flow.svg` | `docs/figures/mla/`（生成器 [gen_mla_forward_flow_svg.py](../../scripts/svg/gen_mla_forward_flow_svg.py)） | MLA 专文、演进总览 §3.2 |

`build_book.py` 中 `ASSET_MULTI_DEST` 将 `ess-dual-cache.svg` 从 **单一源**复制到书中 DSA 卷与 ESS 卷。

### 5.2 校验状态

| 项 | 说明 |
|----|------|
| `mla-forward-flow.svg` | **已重绘**；自 `gen_mla_forward_flow_svg.py` 生成；`LEGACY_SKIP_LAYOUT` 已清空 |
| Engram README | 官方 badge 图为外链 SVG，不计入本地 check |

### 5.3 维护命令

```bash
cd <deepseek-mechanism-atlas 仓库根>
python3 scripts/svg/gen_deepseek_svgs.py # 或单独 gen_*.py
python3 dsa/scripts/svg/gen_dsa_svgs.py
python3 scripts/svg/check_svgs.py # 须 exit 0
python3 《ds-技术报告》/build_book.py
```

---

## 6. 答疑结构

- 6 篇正文均有 **文首回父节** blockquote。
- [qa/README.md](../versions/qa/README.md) 索引表链到演进总览对应 § 与父文档锚点。
- FP8：`partial-sum` = 机制；`mma-term` = 名词，索引表已标注避免误读为重复专文。

---

## 7. 遗留 / 低优先级

| 项 | 建议 |
|----|------|
| `00-V1-技术报告译文.md` | 书中 stub，与源 `v1-technical-report.zh.md` 一致；可保留或从 `CHAPTER_MAP` 移除 |
| 演进总览 §7 | 补 DSA / ESS / Index Share 反向一行 |
| `docs/figures/ess/ess-dual-cache.svg` | 生成器仍同步副本；Markdown 已统一引用 `../dsa/diagrams/`，副本仅作 build 冗余，可日后从 gen 脚本去掉 |

---

## 8. 结论

- **入口**：演进总览 + 成书 `READING_ORDER` 已形成层层递进；FP8 章已入链。
- **双向引用**：系列主文档顶栏 blockquote 已对齐；stub 页已补回链。
- **概念**：一概念一主文档；摘要 vs 专文 vs 答疑层级清晰；SVG 按上表 canonical 复用。
- **SVG**：`mla-forward-flow` 已接生成器；改图后须跑 `check_svgs.py` + `build_book.py`；CI 门禁见 [scripts/doc_series_gate.sh](../../scripts/doc_series_gate.sh)。
