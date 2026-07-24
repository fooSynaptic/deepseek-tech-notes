#!/usr/bin/env python3
"""将 deepseek-mechanism-atlas 文档复制整理为《ds-技术报告》书籍目录（不修改原仓库结构）。"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOK = Path(__file__).resolve().parent
REPO_HOME_URL = "https://github.com/fooSynaptic/deepseek-mechanism-atlas"

# 原路径 -> 书中路径（相对 BOOK）
CHAPTER_MAP: dict[str, str] = {
    "docs/README.md": "00-前言/02-中文导读.md",
    "docs/reports/deepseek-v1-to-v3-lineage.md": "01-总览/04-V1到V3演进.md",
    "docs/reports/deepseek-version-lineage-20260625.md": "01-总览/01-版本演进总览.md",
    "docs/reports/deepseek-algorithm-line.md": "01-总览/05-算法线导读.md",
    "docs/reports/deepseek-infra-line.md": "01-总览/06-基础设施线导读.md",
    "docs/reports/deepseek-moe-line.md": "01-总览/07-MoE线导读.md",
    "docs/versions/README.md": "01-总览/02-版本梗概索引.md",
    "docs/reports/README.md": "01-总览/03-技术报告索引.md",
    "docs/versions/v3.md": "02-基座架构/01-V3基座.md",
    "docs/versions/mla-latent-attention.md": "02-基座架构/02-MLA低秩注意力.md",
    "docs/versions/deepseek-moe.md": "02-基座架构/05-DeepSeekMoE.md",
    "docs/versions/aux-loss-free-moe-routing.md": "02-基座架构/03-aux-loss-free-MoE路由.md",
    "docs/versions/moe-sequence-wise-balance-loss.md": "02-基座架构/04-序列均衡损失.md",
    "docs/versions/v3-fp8-dynamic-quantization.md": "02-基座架构/06-V3-FP8动态量化.md",
    "docs/versions/rlvr.md": "03-后训练与R1/01-RLVR.md",
    "docs/versions/r1.md": "03-后训练与R1/02-R1.md",
    "docs/rl/README.md": "03-后训练与R1/03-RL笔记索引.md",
    "docs/rl/optimize.md": "03-后训练与R1/04-GRPO长程局限.md",
    "docs/versions/v1.md": "04-版本代际/00-V1-LLM.md",
    "docs/versions/v1-technical-report.zh.md": "04-版本代际/00-V1-技术报告译文.md",
    "docs/versions/v1-bbpe-tokenizer.md": "04-版本代际/00-V1-BBPE词表与Tokenizer.md",
    "docs/versions/v2.md": "04-版本代际/00-V2-MoE与MLA.md",
    "docs/versions/v3-1.md": "04-版本代际/01-V3.1-Terminus.md",
    "docs/versions/v3-2.md": "04-版本代际/02-V3.2-DSA.md",
    "docs/versions/v4.md": "04-版本代际/03-V4.md",
    "docs/versions/csa-hca-mixed-attention.md": "04-版本代际/05-CSA-HCA混合压缩注意力.md",
    "docs/versions/hash-moe-fp4.md": "04-版本代际/06-Hash-MoE-FP4.md",
    "docs/versions/muon-optimizer.md": "04-版本代际/07-Muon优化器.md",
    "docs/versions/mhc-manifold-hyper-connections.md": "04-版本代际/04-mHC流形约束超连接.md",
    "docs/versions/hyper-connections.md": "04-版本代际/04b-Hyper-Connections.md",
    "docs/dsa/README.md": "05-DSA稀疏注意力/01-系列导读.md",
    "docs/versions/dsa-sparse-attention.md": "05-DSA稀疏注意力/02-DSA梗概.md",
    "docs/dsa/dsa-logic.md": "05-DSA稀疏注意力/03-DSA逻辑详解.md",
    "docs/dsa/lightning-indexer.md": "05-DSA稀疏注意力/04-Lightning-Indexer详解.md",
    "docs/versions/index-share.md": "05-DSA稀疏注意力/05-Index-Share梗概.md",
    "docs/dsa/index-share-logic.md": "05-DSA稀疏注意力/06-Index-Share逻辑.md",
    "docs/versions/ess-latent-cache-offload.md": "06-推理基础设施/01-ESS概念.md",
    "docs/versions/ess-paper-highlights.md": "06-推理基础设施/02-ESS论文梗概.md",
    "docs/reports/spec-decode-draft-acceleration-20260604.md": "06-推理基础设施/03-投机解码自测加速比.md",
    "docs/versions/dspark-speculative-decoding.md": "06-推理基础设施/04-DSpark投机解码.md",
    "docs/versions/v4-kv-layout.md": "06-推理基础设施/05-V4-KV-Layout.md",
    "docs/versions/v4-hisparse.md": "06-推理基础设施/06-V4-HiSparse.md",
    "docs/versions/v4-disk-prefix-cache.md": "06-推理基础设施/07-V4-磁盘Prefix-Cache.md",
    "docs/engram/README.md": "07-Engram/01-Engram官方README.md",
    "docs/reports/raschka-technical-deepseek-v3-v32-highlights.md": "08-外部解读/01-Raschka要点速读.md",
    "docs/reports/raschka-technical-deepseek-v3-v32.md": "08-外部解读/02-Raschka全文解析.md",
    "docs/reports/zhihu-jiangzijun-dspark-highlights-20260627.md": "08-外部解读/03-酱紫君DSpark阅读笔记.md",
    "docs/reports/deepseek-doc-series-audit-20260627.md": "09-附录/02-文档系列结构审查.md",
    "docs/WIKI-INDEX.md": "09-附录/01-开发索引.md",
    "docs/material/papers/deepseek-r1/training-pipeline.md": "03-后训练与R1/05-R1训练pipeline.md",
    "docs/material/papers/engram/engram-series-overview.md": "07-Engram/02-Engram系列导读.md",
    "docs/papers/thinking-with-visual-primitives-highlights.md": "09-附录/03-Visual-Primitives论文要点.md",
}

ASSET_MAP: dict[str, str] = {
    "diagrams/deepseek-version-lineage.svg": "01-总览/figures/deepseek-version-lineage.svg",
    "diagrams/deepseek-version-quick.svg": "01-总览/figures/deepseek-version-quick.svg",
    "diagrams/opt-direction-ternary.svg": "01-总览/figures/opt-direction-ternary.svg",
    "diagrams/v1-v3-mla-evolution.svg": "01-总览/figures/v1-v3-mla-evolution.svg",
    "diagrams/v1-v3-capability-timeline.svg": "01-总览/figures/v1-v3-capability-timeline.svg",
    "docs/figures/mla/mla-forward-flow.svg": "02-基座架构/figures/mla-forward-flow.svg",
    "diagrams/mla-forward-flow.svg": "02-基座架构/figures/mla-forward-flow.svg",
    "docs/figures/v3/v3-moe-vs-v2.svg": "02-基座架构/figures/v3-moe-vs-v2.svg",
    "diagrams/v3-moe-vs-v2.svg": "02-基座架构/figures/v3-moe-vs-v2.svg",
    "docs/figures/rl/grpo-vs-ppo.svg": "03-后训练与R1/figures/grpo-vs-ppo.svg",
    "diagrams/grpo-vs-ppo.svg": "03-后训练与R1/figures/grpo-vs-ppo.svg",
    "docs/figures/rl/rlvr-posttrain-branch.svg": "03-后训练与R1/figures/rlvr-posttrain-branch.svg",
    "docs/figures/rl/dspark-draft-target-parallel.svg": "08-外部解读/figures/dspark-draft-target-parallel.svg",
    "docs/versions/figures/aux-loss-free-bias-update.svg": "02-基座架构/figures/aux-loss-free-bias-update.svg",
    "docs/versions/figures/moe-bal-loss-pipeline.svg": "02-基座架构/figures/moe-bal-loss-pipeline.svg",
    "docs/versions/figures/csa-hca-evolution-chain.svg": "04-版本代际/figures/csa-hca-evolution-chain.svg",
    "docs/versions/figures/hash-moe-evolution-chain.svg": "04-版本代际/figures/hash-moe-evolution-chain.svg",
    "docs/versions/figures/gpu-sm-hierarchy.svg": "06-推理基础设施/figures/gpu-sm-hierarchy.svg",
    "docs/versions/figures/h2d-pipeline-serial.svg": "06-推理基础设施/figures/h2d-pipeline-serial.svg",
    "docs/figures/rl/dspark-speculative.svg": "06-推理基础设施/figures/dspark-speculative.svg",
    "diagrams/dspark-speculative.svg": "06-推理基础设施/figures/dspark-speculative.svg",
    "docs/versions/figures/dspark-speculative.svg": "06-推理基础设施/figures/dspark-speculative.svg",
    "diagrams/dspark-semi-ar-draft.svg": "06-推理基础设施/figures/dspark-semi-ar-draft.svg",
    "docs/versions/figures/dspark-semi-ar-draft.svg": "06-推理基础设施/figures/dspark-semi-ar-draft.svg",
    "diagrams/dspark-confidence-scheduler.svg": "06-推理基础设施/figures/dspark-confidence-scheduler.svg",
    "docs/versions/figures/dspark-confidence-scheduler.svg": "06-推理基础设施/figures/dspark-confidence-scheduler.svg",
    "docs/figures/v3/mtp-speculative.svg": "06-推理基础设施/figures/mtp-speculative.svg",
    "diagrams/mtp-speculative.svg": "06-推理基础设施/figures/mtp-speculative.svg",
    "docs/versions/figures/mtp-speculative.svg": "06-推理基础设施/figures/mtp-speculative.svg",
    "diagrams/mtp-fusion-scheme.svg": "06-推理基础设施/figures/mtp-fusion-scheme.svg",
    "diagrams/mtp-draft-chain-depth.svg": "06-推理基础设施/figures/mtp-draft-chain-depth.svg",
    "docs/versions/figures/mtp-fusion-scheme.svg": "06-推理基础设施/figures/mtp-fusion-scheme.svg",
    "docs/versions/figures/mtp-draft-chain-depth.svg": "06-推理基础设施/figures/mtp-draft-chain-depth.svg",
    "docs/figures/v3/mla-mode-switch.svg": "02-基座架构/figures/v3/mla-mode-switch.svg",
    "diagrams/mla-mode-switch.svg": "02-基座架构/figures/v3/mla-mode-switch.svg",
    "docs/figures/v3/v3-fp8-dynamic-quant.svg": "02-基座架构/figures/v3/v3-fp8-dynamic-quant.svg",
    "diagrams/v3-fp8-dynamic-quant.svg": "02-基座架构/figures/v3/v3-fp8-dynamic-quant.svg",
    "docs/figures/v4/v4-hetero-kv.svg": "01-总览/figures/v4/v4-hetero-kv.svg",
    "diagrams/v4-hetero-kv.svg": "04-版本代际/figures/v4/v4-hetero-kv.svg",
    "docs/figures/v4/hash-moe-routing.svg": "04-版本代际/figures/v4/hash-moe-routing.svg",
    "diagrams/hash-moe-routing.svg": "04-版本代际/figures/v4/hash-moe-routing.svg",
    "docs/figures/v4/muon-optimizer.svg": "04-版本代际/figures/v4/muon-optimizer.svg",
    "diagrams/muon-optimizer.svg": "04-版本代际/figures/v4/muon-optimizer.svg",
    "docs/figures/mhc/mhc-doubly-stochastic-matrix.svg": "04-版本代际/figures/mhc/mhc-doubly-stochastic-matrix.svg",
    "diagrams/mhc-doubly-stochastic-matrix.svg": "04-版本代际/figures/mhc/mhc-doubly-stochastic-matrix.svg",
    "docs/figures/mhc/hyper-connections.svg": "04-版本代际/figures/mhc/hyper-connections.svg",
    "diagrams/hyper-connections.svg": "04-版本代际/figures/mhc/hyper-connections.svg",
    "docs/figures/mla/mla-forward-flow.png": "02-基座架构/figures/mla-forward-flow.png",
    "docs/dsa/diagrams/dsa-pipeline.svg": "05-DSA稀疏注意力/figures/dsa-pipeline.svg",
    "docs/dsa/diagrams/dsa-three-stage.svg": "05-DSA稀疏注意力/figures/dsa-three-stage.svg",
    "docs/dsa/diagrams/ess-kv-lineage-tree.svg": "05-DSA稀疏注意力/figures/ess-kv-lineage-tree.svg",
    "docs/dsa/diagrams/lightning-indexer-forward.svg": "05-DSA稀疏注意力/figures/lightning-indexer-forward.svg",
    "docs/dsa/diagrams/lightning-indexer-t-s-direction.svg": "05-DSA稀疏注意力/figures/lightning-indexer-t-s-direction.svg",
    "docs/dsa/diagrams/index-share-fffs.svg": "05-DSA稀疏注意力/figures/index-share-fffs.svg",
    "docs/dsa/diagrams/ess-dual-cache.svg": "05-DSA稀疏注意力/figures/ess-dual-cache.svg",
    "docs/figures/ess/v3-mla-latent-kv-offload.svg": "01-总览/figures/v3-mla-latent-kv-offload.svg",
    "docs/dsa/diagrams/v3-mla-latent-kv-offload.svg": "01-总览/figures/v3-mla-latent-kv-offload.svg",
    "docs/figures/papers/thinking-with-visual-primitives/fig-1-token-efficiency.png": "09-附录/figures/visual-primitives/fig-1-token-efficiency.png",
    "docs/figures/papers/thinking-with-visual-primitives/fig-2-architecture-pipeline.png": "09-附录/figures/visual-primitives/fig-2-architecture-pipeline.png",
    "docs/figures/papers/thinking-with-visual-primitives/fig-3-cold-start-counting.png": "09-附录/figures/visual-primitives/fig-3-cold-start-counting.png",
    "docs/figures/papers/thinking-with-visual-primitives/table-1-benchmark.png": "09-附录/figures/visual-primitives/table-1-benchmark.png",
    "docs/papers/Thinking_with_Visual_Primitives.pdf": "09-附录/papers/Thinking_with_Visual_Primitives.pdf",
}

# 同一源 SVG 复制到多个书中卷（canonical 源：docs/dsa/diagrams/）
ASSET_MULTI_DEST: dict[str, list[str]] = {
    "docs/dsa/diagrams/ess-dual-cache.svg": [
        "05-DSA稀疏注意力/figures/ess-dual-cache.svg",
        "06-推理基础设施/figures/ess-dual-cache.svg",
    ],
    "docs/figures/v4/v4-hetero-kv.svg": [
        "01-总览/figures/v4/v4-hetero-kv.svg",
        "04-版本代际/figures/v4/v4-hetero-kv.svg",
    ],
    "docs/figures/ess/v3-mla-latent-kv-offload.svg": [
        "01-总览/figures/v3-mla-latent-kv-offload.svg",
        "06-推理基础设施/figures/v3-mla-latent-kv-offload.svg",
    ],
    "docs/figures/v4/hash-moe-routing.svg": [
        "04-版本代际/figures/v4/hash-moe-routing.svg",
        "02-基座架构/figures/v4/hash-moe-routing.svg",
    ],
    "docs/versions/figures/gpu-sm-hierarchy.svg": [
        "06-推理基础设施/figures/gpu-sm-hierarchy.svg",
        "01-总览/figures/gpu-sm-hierarchy.svg",
    ],
    "docs/versions/figures/h2d-pipeline-serial.svg": [
        "06-推理基础设施/figures/h2d-pipeline-serial.svg",
        "01-总览/figures/h2d-pipeline-serial.svg",
    ],
}
for _fig in Path(REPO / "docs/figures/v1/scaling-law").glob("*.png"):
    ASSET_MAP[f"docs/figures/v1/scaling-law/{_fig.name}"] = (
        f"04-版本代际/figures/v1/scaling-law/{_fig.name}"
    )
for _fig in Path(REPO / "docs/figures/v1/bbpe").glob("*.svg"):
    ASSET_MAP[f"docs/figures/v1/bbpe/{_fig.name}"] = (
        f"04-版本代际/figures/v1/bbpe/{_fig.name}"
    )
for _fig in Path(REPO / "docs/figures/v2").glob("*"):
    ASSET_MAP[f"docs/figures/v2/{_fig.name}"] = (
        f"02-基座架构/figures/v2/{_fig.name}"
    )

# 答疑文档：repo 路径 -> 书中路径（同一源文件可映射到多个卷 qa/）
QA_DESTINATIONS: dict[str, list[str]] = {
    "docs/versions/qa/v1-scaling-law-c-vs-md.md": [
        "01-总览/qa/v1-scaling-law-c-vs-md.md",
        "04-版本代际/qa/v1-scaling-law-c-vs-md.md",
    ],
    "docs/versions/qa/moe-fine-grained-segmentation.md": [
        "02-基座架构/qa/moe-fine-grained-segmentation.md",
    ],
    "docs/versions/qa/moe-centroid-vs-gate-weight.md": [
        "02-基座架构/qa/moe-centroid-vs-gate-weight.md",
    ],
    "docs/versions/qa/moe-expert-parallel-ep.md": [
        "02-基座架构/qa/moe-expert-parallel-ep.md",
        "04-版本代际/qa/moe-expert-parallel-ep.md",
    ],
    "docs/versions/qa/hash-moe-shallow-vs-deep.md": [
        "02-基座架构/qa/hash-moe-shallow-vs-deep.md",
        "04-版本代际/qa/hash-moe-shallow-vs-deep.md",
    ],
    "docs/versions/qa/v4-swa-sliding-window.md": [
        "04-版本代际/qa/v4-swa-sliding-window.md",
        "06-推理基础设施/qa/v4-swa-sliding-window.md",
    ],
    "docs/versions/qa/v4-indexer-kv.md": [
        "04-版本代际/qa/v4-indexer-kv.md",
        "05-DSA稀疏注意力/qa/v4-indexer-kv.md",
    ],
    "docs/versions/qa/v4-tail-buffer.md": [
        "04-版本代际/qa/v4-tail-buffer.md",
        "06-推理基础设施/qa/v4-tail-buffer.md",
    ],
    "docs/versions/qa/v4-kv-dual-pool-alignment.md": [
        "06-推理基础设施/qa/v4-kv-dual-pool-alignment.md",
        "04-版本代际/qa/v4-kv-dual-pool-alignment.md",
    ],
    "docs/versions/qa/mhc-birkhoff-polytope.md": [
        "04-版本代际/qa/mhc-birkhoff-polytope.md",
    ],
    "docs/versions/qa/fp8-partial-sum-drift.md": [
        "02-基座架构/qa/fp8-partial-sum-drift.md",
    ],
    "docs/versions/qa/fp8-mma-term.md": [
        "02-基座架构/qa/fp8-mma-term.md",
    ],
    "docs/versions/qa/h2d-d2h-pcie-transfer.md": [
        "06-推理基础设施/qa/h2d-d2h-pcie-transfer.md",
        "01-总览/qa/h2d-d2h-pcie-transfer.md",
    ],
    "docs/versions/qa/spec-decode-rejection-sampling.md": [
        "06-推理基础设施/qa/spec-decode-rejection-sampling.md",
    ],
    "docs/versions/qa/spec-decode-compute-vs-memory-bound.md": [
        "06-推理基础设施/qa/spec-decode-compute-vs-memory-bound.md",
    ],
    "docs/versions/qa/gpu-sm-term.md": [
        "06-推理基础设施/qa/gpu-sm-term.md",
    ],
    "docs/versions/qa/mtp-fusion-scheme.md": [
        "06-推理基础设施/qa/mtp-fusion-scheme.md",
    ],
    "docs/versions/qa/README.md": [
        "01-总览/qa/README.md",
        "02-基座架构/qa/README.md",
    ],
}
QA_MAP: dict[str, str] = {
    src: dests[0] for src, dests in QA_DESTINATIONS.items()
}

# ESS 论文图（若已生成）
for name in [
    "table-1.png",
    "fig-1.png",
    "fig-2.png",
    "fig-3.png",
    "fig-4.png",
    "fig-5.png",
    "fig-6.png",
    "fig-7.png",
    "fig-8.png",
    "fig-9.png",
    "table-2.png",
]:
    ASSET_MAP[f"docs/figures/ess/paper/{name}"] = f"06-推理基础设施/figures/paper/{name}"

ASSET_MAP["docs/figures/ess/paper/screenshots"] = (
    "06-推理基础设施/figures/paper/screenshots"
)

BOOK_PATH_TO_REPO = {v: k for k, v in CHAPTER_MAP.items()}
BOOK_PATH_TO_REPO.update({v: k for k, v in ASSET_MAP.items()})
for _src, _dests in QA_DESTINATIONS.items():
    for _d in _dests:
        BOOK_PATH_TO_REPO[_d] = _src

# 旧文档路径 -> 书中目标（含默认锚点）；仅用于链接重写，不参与 copy
LINK_REDIRECTS: dict[str, tuple[str, str]] = {
    "docs/versions/mhc-doubly-stochastic-manifold.md": (
        "docs/versions/mhc-manifold-hyper-connections.md",
        "#3-mhc-核心双随机流形约束",
    ),
    "docs/reports/deepseek-llm-v1-highlights.md": (
        "docs/versions/v1.md",
        "",
    ),
    "docs/versions/v1-technical-report.zh.md": (
        "docs/versions/v1.md",
        "",
    ),
}

# 全书线性阅读顺序（用于文末「上一章 / 下一章」导航）
READING_ORDER: list[str] = [
    "01-总览/01-版本演进总览.md",
    "01-总览/05-算法线导读.md",
    "01-总览/06-基础设施线导读.md",
    "01-总览/07-MoE线导读.md",
    "01-总览/04-V1到V3演进.md",
    "01-总览/02-版本梗概索引.md",
    "01-总览/03-技术报告索引.md",
    "02-基座架构/01-V3基座.md",
    "02-基座架构/02-MLA低秩注意力.md",
    "02-基座架构/05-DeepSeekMoE.md",
    "02-基座架构/03-aux-loss-free-MoE路由.md",
    "02-基座架构/04-序列均衡损失.md",
    "02-基座架构/06-V3-FP8动态量化.md",
    "03-后训练与R1/01-RLVR.md",
    "03-后训练与R1/02-R1.md",
    "03-后训练与R1/03-RL笔记索引.md",
    "03-后训练与R1/04-GRPO长程局限.md",
    "03-后训练与R1/05-R1训练pipeline.md",
    "04-版本代际/00-V1-LLM.md",
    "04-版本代际/00-V1-BBPE词表与Tokenizer.md",
    "04-版本代际/00-V2-MoE与MLA.md",
    "04-版本代际/01-V3.1-Terminus.md",
    "04-版本代际/02-V3.2-DSA.md",
    "04-版本代际/03-V4.md",
    "04-版本代际/05-CSA-HCA混合压缩注意力.md",
    "04-版本代际/04b-Hyper-Connections.md",
    "04-版本代际/04-mHC流形约束超连接.md",
    "04-版本代际/06-Hash-MoE-FP4.md",
    "04-版本代际/07-Muon优化器.md",
    "05-DSA稀疏注意力/01-系列导读.md",
    "05-DSA稀疏注意力/02-DSA梗概.md",
    "05-DSA稀疏注意力/03-DSA逻辑详解.md",
    "05-DSA稀疏注意力/04-Lightning-Indexer详解.md",
    "05-DSA稀疏注意力/05-Index-Share梗概.md",
    "05-DSA稀疏注意力/06-Index-Share逻辑.md",
    "06-推理基础设施/01-ESS概念.md",
    "06-推理基础设施/02-ESS论文梗概.md",
    "06-推理基础设施/03-投机解码自测加速比.md",
    "06-推理基础设施/04-DSpark投机解码.md",
    "06-推理基础设施/05-V4-KV-Layout.md",
    "06-推理基础设施/06-V4-HiSparse.md",
    "06-推理基础设施/07-V4-磁盘Prefix-Cache.md",
    "07-Engram/01-Engram官方README.md",
    "07-Engram/02-Engram系列导读.md",
    "08-外部解读/01-Raschka要点速读.md",
    "08-外部解读/02-Raschka全文解析.md",
    "08-外部解读/03-酱紫君DSpark阅读笔记.md",
    "09-附录/03-Visual-Primitives论文要点.md",
    "09-附录/01-开发索引.md",
    "09-附录/02-文档系列结构审查.md",
]

MATERIAL_PREFIX = "docs/material/"

ENGRAM_EXTRA: dict[str, str] = {
    "docs/engram/Engram_paper.pdf": "07-Engram/Engram_paper.pdf",
    "docs/engram/drawio/Engram.drawio": "07-Engram/drawio/Engram.drawio",
    "docs/engram/LICENSE": "07-Engram/LICENSE",
    "docs/engram/engram_demo_v1.py": "07-Engram/engram_demo_v1.py",
}
for _ef in Path(REPO / "docs/engram/figures").glob("*"):
    if _ef.is_file():
        ENGRAM_EXTRA[f"docs/engram/figures/{_ef.name}"] = f"07-Engram/figures/{_ef.name}"

NAV_FOOTER_RE = re.compile(r"\n---\n\n## 章节导航\n.*\Z", re.DOTALL)
# 源稿中 <!-- book:omit --> … <!-- /book:omit --> 区块不写入成书（如 docs/README 维护说明）
BOOK_OMIT_RE = re.compile(r"<!-- book:omit -->[\s\S]*?<!-- /book:omit -->\n?", re.MULTILINE)
HEADER_BREADCRUMB_RE = re.compile(r"^# [^\n]+\n\n((?:> [^\n]*\n)+)", re.MULTILINE)

LINK_RE = re.compile(r"(\[[^\]]*\]\()([^)]+)(\))")
IMG_SRC_RE = re.compile(r'(<img[^>]+src=")([^"]+)(")', re.I)


def material_book_path(repo_rel: str) -> str | None:
    if repo_rel.startswith(MATERIAL_PREFIX):
        return "09-附录/material/" + repo_rel[len(MATERIAL_PREFIX) :]
    return None


def resolve_repo_path(from_repo_rel: str, href: str) -> str | None:
    """将相对链接解析为 repo 根相对路径（仅本地路径）。"""
    href = href.split("#")[0].strip()
    if not href or href.startswith(("http://", "https://", "mailto:")):
        return None

    # 目录链接 -> README
    dir_map = {
        "dsa": "docs/dsa/README.md",
        "docs/dsa/": "docs/dsa/README.md",
        "./dsa": "docs/dsa/README.md",
        "./docs/dsa/": "docs/dsa/README.md",
        "Engram": "docs/engram/README.md",
        "docs/engram/": "docs/engram/README.md",
        "./Engram": "docs/engram/README.md",
        "./docs/engram/": "docs/engram/README.md",
    }
    if href in dir_map:
        return dir_map[href]

    bases = [(REPO / from_repo_rel).parent]
    # qa/ 等深层目录：../reports 需沿父链解析（versions/qa/../reports 不存在）
    _p = (REPO / from_repo_rel).parent.resolve()
    _root = REPO.resolve()
    while True:
        if _p not in bases:
            bases.append(_p)
        if _p == _root:
            break
        _p = _p.parent
    if href.startswith(("./docs/", "./docs/dsa/", "./docs/rl/", "./diagrams/", "./docs/engram/")):
        bases.append(REPO)
    if href.startswith(("docs/", "docs/dsa/", "docs/rl/", "diagrams/", "docs/engram/")):
        bases.append(REPO)
    if href.startswith(("./docs/material/", "docs/material/")):
        bases.append(REPO)

    seen: set[Path] = set()
    for base in bases:
        for raw in (href, href.removeprefix("./")):
            candidate = (base / raw).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                rel = candidate.relative_to(REPO.resolve())
            except ValueError:
                continue
            rel_s = str(rel).replace("\\", "/")
            if candidate.is_file():
                return rel_s
            if candidate.is_dir():
                key = rel_s.rstrip("/")
                if key in ASSET_MAP:
                    return key
                if (candidate / "README.md").is_file():
                    return str((rel / "README.md").as_posix()).replace("\\", "/")
    return None


def resolve_local_path(from_repo_rel: str, href: str) -> Path | None:
    """解析本地路径（相对本仓根目录）。"""
    href_path = href.split("#")[0].strip()
    if not href_path or href_path.startswith(("http://", "https://", "mailto:")):
        return None

    bases = [
        (REPO / from_repo_rel).parent,
        REPO,
        REPO.parent,
        REPO.parent.parent,
    ]
    seen: set[Path] = set()
    for base in bases:
        for raw in (href_path, href_path.removeprefix("./")):
            cand = (base / raw).resolve()
            if cand in seen:
                continue
            seen.add(cand)
            if cand.is_file():
                return cand
            if cand.is_dir() and (cand / "README.md").is_file():
                return cand / "README.md"
    return None


def book_rel_to_target(from_book: Path, target: Path, anchor: str = "") -> str:
    rel = os.path.relpath(target, from_book.parent).replace("\\", "/")
    return f"{rel}{anchor}"


def book_rel_link(from_book: Path, target_book_rel: str, anchor: str = "") -> str:
    rel = os.path.relpath(BOOK / target_book_rel, from_book.parent).replace("\\", "/")
    return f"{rel}{anchor}"


def qa_book_target(from_book_rel: str, qa_repo_path: str) -> str | None:
    """按当前章节所在卷，选择最合适的 qa 副本路径。"""
    dests = QA_DESTINATIONS.get(qa_repo_path)
    if not dests:
        return None
    from_vol = Path(from_book_rel).parts[0] if Path(from_book_rel).parts else ""
    for d in dests:
        if d.startswith(from_vol + "/"):
            return d
    return dests[0]


def material_relative_book_link(
    from_book: Path, from_repo_rel: str, href: str, anchor: str = ""
) -> str | None:
    """参考专文内的 `./` 相对链 → 书中 09-附录/material 镜像路径。"""
    if not from_repo_rel.startswith(MATERIAL_PREFIX):
        return None
    if not href.startswith("./"):
        return None
    mirror = material_book_path(from_repo_rel)
    if not mirror:
        return None
    base = (BOOK / mirror).parent
    target = (base / href.removeprefix("./")).resolve()
    if target.is_file() or (target.is_dir() and (target / "README.md").is_file()):
        return book_rel_to_target(from_book, target, anchor)
    if target.is_dir():
        return book_rel_to_target(from_book, target, anchor)
    return None


def book_self_link(from_book: Path, href: str, anchor: str = "") -> str | None:
    marker = "《ds-技术报告》/"
    if marker not in href:
        return None
    sub = href.split(marker, 1)[1]
    return book_rel_link(from_book, sub, anchor)


def rewrite_content(text: str, from_repo_rel: str, from_book_rel: str) -> str:
    from_book = BOOK / from_book_rel

    def repl_link(m: re.Match[str]) -> str:
        prefix, href, suffix = m.group(1), m.group(2), m.group(3)
        anchor = ""
        if "#" in href:
            href, anchor = href.split("#", 1)
            anchor = "#" + anchor
        else:
            anchor = ""
        repo_path = resolve_repo_path(from_repo_rel, href)
        if repo_path == "README.md":
            return prefix + REPO_HOME_URL + anchor + suffix
        self_href = book_self_link(from_book, href, anchor)
        if self_href:
            return prefix + self_href + suffix
        rel_ext = material_relative_book_link(from_book, from_repo_rel, href, anchor)
        if rel_ext:
            return prefix + rel_ext + suffix
        if repo_path in LINK_REDIRECTS:
            target_repo, default_anchor = LINK_REDIRECTS[repo_path]
            if not anchor:
                anchor = default_anchor
            if target_repo in CHAPTER_MAP:
                return prefix + book_rel_link(from_book, CHAPTER_MAP[target_repo], anchor) + suffix
        if repo_path and repo_path in CHAPTER_MAP:
            return prefix + book_rel_link(from_book, CHAPTER_MAP[repo_path], anchor) + suffix
        if repo_path and repo_path in ASSET_MAP:
            return prefix + book_rel_link(from_book, ASSET_MAP[repo_path], anchor) + suffix
        if repo_path and repo_path in QA_DESTINATIONS:
            target = qa_book_target(from_book_rel, repo_path)
            if target:
                return prefix + book_rel_link(from_book, target, anchor) + suffix
        if repo_path:
            ext = material_book_path(repo_path)
            if ext:
                return prefix + book_rel_link(from_book, ext, anchor) + suffix
        local = resolve_local_path(from_repo_rel, href)
        if local and local.is_file():
            try:
                repo_rel = str(local.relative_to(REPO.resolve())).replace("\\", "/")
                if repo_rel in ENGRAM_EXTRA:
                    return prefix + book_rel_link(from_book, ENGRAM_EXTRA[repo_rel], anchor) + suffix
                if repo_rel in CHAPTER_MAP:
                    return prefix + book_rel_link(from_book, CHAPTER_MAP[repo_rel], anchor) + suffix
            except ValueError:
                pass
            return prefix + book_rel_to_target(from_book, local, anchor) + suffix
        return m.group(0)

    def repl_img(m: re.Match[str]) -> str:
        prefix, src, suffix = m.group(1), m.group(2), m.group(3)
        anchor = ""
        if "#" in src:
            src, anchor = src.split("#", 1)
            anchor = "#" + anchor
        rel_ext = material_relative_book_link(from_book, from_repo_rel, src, anchor)
        if rel_ext:
            return prefix + rel_ext + suffix
        repo_path = resolve_repo_path(from_repo_rel, src)
        if repo_path and repo_path in ASSET_MAP:
            return prefix + book_rel_link(from_book, ASSET_MAP[repo_path]) + suffix
        if repo_path:
            ext = material_book_path(repo_path)
            if ext:
                return prefix + book_rel_link(from_book, ext) + suffix
        local = resolve_local_path(from_repo_rel, src)
        if local and local.is_file():
            try:
                repo_rel = str(local.relative_to(REPO.resolve())).replace("\\", "/")
                if repo_rel in ENGRAM_EXTRA:
                    return prefix + book_rel_link(from_book, ENGRAM_EXTRA[repo_rel]) + suffix
            except ValueError:
                pass
            return prefix + book_rel_to_target(from_book, local) + suffix
        return m.group(0)

    text = LINK_RE.sub(repl_link, text)
    text = IMG_SRC_RE.sub(repl_img, text)
    return text


def strip_book_omit_blocks(body: str) -> str:
    """Remove <!-- book:omit --> sections (repo-only; not for mdBook readers)."""
    return BOOK_OMIT_RE.sub("", body)


def _chapter_title(path: Path) -> str:
    if not path.is_file():
        return path.stem
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def extract_header_breadcrumb(body: str) -> str:
    """提取文首 blockquote 回链（与章节开头一致，用于文末复贴）。"""
    m = HEADER_BREADCRUMB_RE.match(body.lstrip("\ufeff"))
    if not m:
        return ""
    return m.group(1).rstrip()


def append_chapter_nav(body: str, book_rel: str) -> str:
    """Strip stale「章节导航」footer. Prev/next table disabled — use mdBook sidebar + 文首回链."""
    return NAV_FOOTER_RE.sub("", body.rstrip())


def apply_chapter_nav() -> None:
    """Remove legacy「章节导航」footers from book chapters."""
    for book_rel in READING_ORDER:
        path = BOOK / book_rel
        if not path.is_file():
            print(f"SKIP nav (missing): {book_rel}")
            continue
        body = path.read_text(encoding="utf-8")
        path.write_text(append_chapter_nav(body, book_rel), encoding="utf-8")
    print(f"STRIP_NAV {len(READING_ORDER)} chapters")


def copy_material_mirror() -> None:
    src = REPO / "docs/material"
    dst = BOOK / "09-附录/material"
    if not src.is_dir():
        return
    if dst.exists():
        shutil.rmtree(dst)

    _MG_SKIP_DIRS = frozenset(
        {
            "tests",
            "scripts",
            "logs",
            "data",
            "checkpoints",
            ".git",
            ".cache",
            ".tmp",
            ".cursor",
            ".pytest_cache",
        }
    )
    _MG_SKIP_SUFFIX = frozenset({".py", ".pt", ".bin", ".safetensors", ".ckpt"})

    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        skip: set[str] = set()
        for n in names:
            if n.endswith(tuple(_MG_SKIP_SUFFIX)):
                skip.add(n)
            elif n in _MG_SKIP_DIRS:
                skip.add(n)
        return skip

    def _copy_subtree(rel: str) -> None:
        s = src / rel
        if not s.exists():
            return
        d = dst / rel
        if s.is_file():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
        else:
            shutil.copytree(s, d, ignore=_ignore)

    for rel in (
        "README.md",
        "papers",
        "meta/deepseek-version-lines-crossrefs.md",
        "meta/torch-training.md",
    ):
        _copy_subtree(rel)
    print("COPY material mirror -> 09-附录/material")


def process_material_mirror() -> None:
    """对材料库镜像内的 md 做链接重写，使其在书中可解析。"""
    mirror = BOOK / "09-附录/material"
    if not mirror.is_dir():
        return
    for md in mirror.rglob("*.md"):
        book_rel = str(md.relative_to(BOOK)).replace("\\", "/")
        repo_rel = "docs/material/" + book_rel[len("09-附录/material/") :]
        body = md.read_text(encoding="utf-8")
        body = rewrite_content(body, repo_rel, book_rel)
        md.write_text(body, encoding="utf-8")
    print("REWRITE material mirror md links")


def copy_engram_extra() -> None:
    for src_rel, dst_rel in ENGRAM_EXTRA.items():
        src = REPO / src_rel
        dst = BOOK / dst_rel
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"COPY engram extras {len(ENGRAM_EXTRA)} files")


def copy_ess_screenshots() -> None:
    src = REPO / "docs/figures/ess/paper/screenshots"
    dst = BOOK / "06-推理基础设施/figures/paper/screenshots"
    if not src.is_dir():
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name.endswith(".md"):
            continue
        if f.is_file():
            shutil.copy2(f, dst / f.name)


def copy_assets() -> None:
    copied_multi: set[str] = set()
    for src_rel, dst_rels in ASSET_MULTI_DEST.items():
        src = REPO / src_rel
        if not src.is_file():
            continue
        for dst_rel in dst_rels:
            dst = BOOK / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied_multi.add(src_rel)

    for src_rel, dst_rel in ASSET_MAP.items():
        if src_rel in copied_multi:
            continue
        src = REPO / src_rel
        dst = BOOK / dst_rel
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_qa() -> None:
    for src_rel, dst_rels in QA_DESTINATIONS.items():
        src = REPO / src_rel
        if not src.is_file():
            print(f"SKIP missing qa: {src_rel}")
            continue
        body_src = src.read_text(encoding="utf-8")
        for dst_rel in dst_rels:
            dst = BOOK / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            body = rewrite_content(body_src, src_rel, dst_rel)
            dst.write_text(body, encoding="utf-8")
            print(f"COPY qa {dst_rel}")


def copy_chapters() -> None:
    for src_rel, dst_rel in CHAPTER_MAP.items():
        src = REPO / src_rel
        if not src.is_file():
            print(f"SKIP missing: {src_rel}")
            continue
        dst = BOOK / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        body = src.read_text(encoding="utf-8")
        body = strip_book_omit_blocks(body)
        body = rewrite_content(body, src_rel, dst_rel)
        dst.write_text(body, encoding="utf-8")
        print(f"COPY {dst_rel}")


def write_master_toc() -> None:
    toc = """# DeepSeek 技术报告

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
| [04 序列均衡损失](./02-基座架构/04-序列均衡损失.md) | $L_{\\mathrm{Bal}}$ 互补均衡 |
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
"""
    (BOOK / "README.md").write_text(toc, encoding="utf-8")
    print("WRITE README.md")


def copy_root_license_files() -> None:
    for name in ("LICENSE", "LICENSE-MIT"):
        src = REPO / name
        if src.is_file():
            shutil.copy2(src, BOOK / name)
            print(f"COPY {name}")


def main() -> None:
    # 清空旧章节（保留 build_book.py 与 README 将重写）
    for child in BOOK.iterdir():
        if child.name in ("build_book.py", "README.md"):
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    copy_assets()
    copy_ess_screenshots()
    copy_material_mirror()
    process_material_mirror()
    copy_engram_extra()
    copy_chapters()
    copy_qa()
    copy_root_license_files()
    write_master_toc()
    apply_chapter_nav()
    _gen_mdbook_summary()
    print("DONE")


def _gen_mdbook_summary() -> None:
    import subprocess
    import sys

    script = REPO / "scripts/gen_mdbook_summary.py"
    subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
