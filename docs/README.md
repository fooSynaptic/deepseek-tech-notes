# deepseek-tech-notes · 中文导读

> **丝滑阅读 × 深度拆解 × 前沿跟进** — 非官方 DeepSeek 技术笔记（V1→V4）：双向引用 wiki + 成书读本，机制、公式与推理 infra（DSA、DSpark、Engram 等）逐级展开。**与 DeepSeek 官方无隶属关系**。
>
> **Smooth, deep notes on frontier DeepSeek tech** — bidirectional navigation from V1 through **V4**: formulas, walkthroughs, and infra patches in wiki + book form. Unofficial; not affiliated with DeepSeek.

> [← English homepage](../README.md) · **[在线成书（mdBook）](https://fooSynaptic.github.io/deepseek-tech-notes/)**

仓库 GitHub 首页为英文；**正文、成书读本与深度答疑以简体中文为主**，面向**中文技术社区**。

**在线成书（公式 / 引用块推荐）：** [GitHub Pages mdBook](https://fooSynaptic.github.io/deepseek-tech-notes/) — 与本地 IDE Preview 同一套 Markdown + MathJax；GitHub 仓库内 `.md` 预览并非标准渲染器。

> **项目仍在完善中**：梗概补全、书中镜像、链接与图示校验仍在推进。阅读时请以各篇文首的 arXiv / 官方 PDF 为准；发现断链、口径不一致或表述错误，**欢迎提 issue**。

---

## 这个项目在做什么

我们从 DeepSeek **最早公开的技术报告**一路跟到 **V4**，并把**大部分**主要技术文章里的**机制与细节**拆开写清楚：架构怎么变、训练/推理在优化什么、版本之间如何衔接。

范围包括：

- **DeepSeek 主线**：MLA、DeepSeekMoE、aux-loss-free 路由、MTP、RLVR/R1、DSA、CSA/HCA、mHC、Hash MoE、V4 异构 KV 等。
- **V4 及衍生的推理技术**：如 **[DSpark](versions/dspark-speculative-decoding.md)** 投机解码（半自回归 draft + 置信度调度验证）、HiSparse、磁盘 Prefix 等。
- **叠在 DeepSeek checkpoint 上的衍生工作**——尤其 **AI Infrastructure** 向：
  - **[Index Share / IndexCache](versions/index-share.md)**（清华 + 智谱）：跨层复用 DSA indexer 的 top-$k$ index，纯推理补丁；
  - **[ESS](versions/ess-latent-cache-offload.md)**（百度百舸）：Latent-Cache CPU offload，与 DSA 算法正交。

### 丝滑阅读：双向引用

本仓库的一大特性是**连贯、友好的阅读体验**：从 [中文导读](#deepseek-tech-notes--中文导读) 或 [演进总览](reports/deepseek-version-lineage-20260625.md) 进入任意梗概、逻辑详解或 `qa/` 答疑后，文首均有**反向引用**——可一键回到来源章节、专题索引、[仓库首页（EN）](../README.md)。深入 DSA / Engram / MTP 等子专题后，同样能沿链接原路返回，阅读路径不断档。

三条专题导读：[算法线](reports/deepseek-algorithm-line.md) · [基础设施线](reports/deepseek-infra-line.md) · [MoE 线](reports/deepseek-moe-line.md)

栅格图导出（仅 PNG）：仓库根目录 `png/`（`python3 scripts/export_png.py` 生成）。

---

## 演进

**[版本演进总览](reports/deepseek-version-lineage-20260625.md)** — 全系列唯一主线入口：时间线 + 算法 / 基础设施 / MoE 三线；各版本与 infra 补丁的内链均从此文展开。

<img src="../diagrams/deepseek-version-lineage.svg" alt="DeepSeek 版本时间线：V3 至 V4 算法演进与 Index Share / ESS / DSpark / HiSparse 基础设施补丁" width="920"/>

[直接打开 SVG](../diagrams/deepseek-version-lineage.svg) · 与 [演进总览 §1](reports/deepseek-version-lineage-20260625.md#1-总览) 对照阅读

<img src="../diagrams/grpo-vs-ppo.svg" alt="PPO vs GRPO：RLHF 神经 RM + Critic 与 RLVR 验证器 + 组内 baseline 对比" width="920"/>

[直接打开 SVG](../diagrams/grpo-vs-ppo.svg) · [RLVR / GRPO](versions/rlvr.md) · [R1](versions/r1.md)

<img src="../diagrams/mtp-fusion-scheme.svg" alt="MTP 融合：主网单步 1 次前向，MTP 链补 draft，无需 K 遍完整前向" width="920"/>

[直接打开 SVG](../diagrams/mtp-fusion-scheme.svg) · [DSpark 投机解码](versions/dspark-speculative-decoding.md) · [MTP 融合 scheme](versions/qa/mtp-fusion-scheme.md)

---

## 文章

| 主题 | 一句话 |
|------|--------|
| [**V1**](versions/v1.md) | DeepSeek-LLM 完整中文译文 |
| [**V1 BBPE**](versions/v1-bbpe-tokenizer.md) | Byte-level BPE 词表与预分词 |
| [**V2**](versions/v2.md) | 236B/21B；MLA + DeepSeekMoE 首次引入 |
| [**V3**](versions/v3.md) | 671B MoE + MLA 开源旗舰基座 |
| [**V3 FP8**](versions/v3-fp8-dynamic-quantization.md) | 训练侧 FP8 块级动态量化 |
| [**R1**](versions/r1.md) | V3-Base + RLVR；架构不变 |
| [**RLVR / GRPO**](versions/rlvr.md) | 可验证奖励 + 组内相对优化 |
| [**V3.1**](versions/v3-1.md) | Hybrid 推理，128K |
| [**V3.2**](versions/v3-2.md) | DSA 稀疏注意力 |
| [**DSA**](versions/dsa-sparse-attention.md) · [逻辑详解](dsa/dsa-logic.md) | indexer + top-$k$ + Core MLA |
| [**Index Share**](versions/index-share.md) · [逻辑详解](dsa/index-share-logic.md) | IndexCache 纯 infra 补丁 |
| [**ESS**](versions/ess-latent-cache-offload.md) · [论文梗概](versions/ess-paper-highlights.md) | Latent-Cache CPU offload |
| [**V4**](versions/v4.md) | CSA + HCA + mHC；1M context |
| [**CSA / HCA**](versions/csa-hca-mixed-attention.md) | 4:1 稀疏 + 128:1 dense 混合压缩注意力 |
| [**mHC**](versions/mhc-manifold-hyper-connections.md) | 双随机流形约束超连接 |
| [**Hash MoE + FP4**](versions/hash-moe-fp4.md) | Hash 路由 + routed expert FP4 |
| [**V4 KV**](versions/v4-kv-layout.md) | Classical + State 双池 |
| [**V4 HiSparse**](versions/v4-hisparse.md) | inactive C4 CPU offload |
| [**V4 磁盘 Prefix**](versions/v4-disk-prefix-cache.md) | CSA/HCA 落盘 + SWA 三档策略 |
| [**DSpark**](versions/dspark-speculative-decoding.md) | V4 投机解码：半自回归 draft + 置信度验证 |
| [**MLA**](versions/mla-latent-attention.md) | latent 压缩 KV |
| [**DeepSeekMoE**](versions/deepseek-moe.md) | 细粒度 routed + shared experts |
| [**MoE 路由**](versions/aux-loss-free-moe-routing.md) | aux-loss-free 动态 bias 负载均衡 |
| [**$L_{\mathrm{Bal}}$**](versions/moe-sequence-wise-balance-loss.md) | 序列内专家均衡损失 |
| [**Hyper-Connections**](versions/hyper-connections.md) | $n$ 路并行残差流；mHC 前置 |

---

## 项目结构（方案 A：mdBook + GitHub Pages）

| 路径 | 作用 |
|------|------|
| [`docs/`](.) | **源稿** — 在此编辑 |
| [`../《ds-技术报告》/`](../《ds-技术报告》/) | **成书镜像** — `build_book.py` 生成，勿手改 |
| [`../book.toml`](../book.toml) + [`../theme/`](../theme/) | mdBook 配置与样式 |
| [`../scripts/build_pages.sh`](../scripts/build_pages.sh) | 本地 / CI 构建 Pages 站点 |

在线阅读请用 **[Pages mdBook](https://fooSynaptic.github.io/deepseek-tech-notes/)**，不要用 GitHub blob 预览对照公式。

---

## 如何新增内容 & 维护成书

**正文源稿在 `docs/`。** [《ds-技术报告》/](../《ds-技术报告》/) 是 **build 生成的书籍镜像**，不要手改其中的 Markdown（会被 `build_book.py` 覆盖）。

新增或移动文档时：

1. **在 `docs/` 下写稿**（如 `docs/versions/`、`docs/dsa/`、`docs/reports/`、`docs/versions/qa/`）。
2. **在 [`《ds-技术报告》/build_book.py`](../《ds-技术报告》/build_book.py) 里登记**：
   - `CHAPTER_MAP` — `docs/...` → 书中章节路径；
   - `READING_ORDER` — 上下章导航顺序；
   - `QA_DESTINATIONS` — 若是答疑页（可镜像到多个书内目录）；
   - `ASSET_MAP` — 仅当新图需要复制进书内目录时。
3. **补导航** — 文首 blockquote 顶栏加 `←` 回链（参考现有篇章）；在相关总览 / 索引里链到新文。
4. **重建并校验**（仓库根目录）：

```bash
python3 《ds-技术报告》/build_book.py
python3 scripts/validate_refs.py
python3 scripts/validate_backlinks.py
```

或一键：`bash scripts/doc_series_gate.sh`。

本地预览 mdBook 站点（需安装 [mdBook](https://rust-lang.github.io/mdBook/)）：

```bash
bash scripts/build_pages.sh
# 打开 mdbook-out/index.html
```

只在 `docs/` 阅读/wiki 模式可不跑 build；要让章节进入 [《ds-技术报告》](../《ds-技术报告》/01-总览/01-版本演进总览.md) 并重写链路与章节导航时，必须执行 `build_book.py`。推送到 `main` 会自动重建 GitHub Pages。

---

## 许可

| 范围 | 许可 |
|------|------|
| 导读、图示、成书读本 | [CC BY 4.0](../LICENSE) |
| `scripts/` | [MIT](../LICENSE-MIT) |
| `docs/engram/` | [Apache 2.0](engram/LICENSE) |
| `docs/material/` 镜像 | 上游 / 原论文许可 |

DeepSeek 论文、权重与官方代码库另有其许可；引用时请以 **arXiv / 官方发布** 为准。
