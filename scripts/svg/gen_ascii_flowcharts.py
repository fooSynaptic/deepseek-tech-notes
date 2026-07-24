#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replace ASCII box/arrow flowcharts in docs with SVG (see gen_dsa_svgs.dsa-three-stage)."""
from __future__ import annotations

import html
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAGRAMS, DSA_DIAGRAMS, FIGURES, REPO  # noqa: E402

RL_FIG = FIGURES / "rl"
VERSIONS_FIG = REPO / "docs" / "versions" / "figures"
R1_DIAG = REPO / "docs" / "material" / "papers" / "deepseek-r1" / "diagrams"


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def svg_header(w: int, h: int, prefix: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
<style><![CDATA[
  .t {{ font: 700 15px sans-serif; fill: #1a1a2e; text-anchor: middle; }}
  .st {{ font: 11px sans-serif; fill: #666; text-anchor: middle; }}
  .lb {{ font: 600 11px sans-serif; fill: #222; text-anchor: middle; }}
  .dt {{ font: 10px sans-serif; fill: #555; text-anchor: middle; }}
  .an {{ font: 9px sans-serif; fill: #2563eb; text-anchor: middle; }}
  .arr {{ stroke: #666; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}a); }}
  .arr-b {{ stroke: #4A90D9; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}ab); }}
  .arr-g {{ stroke: #27AE60; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}ag); }}
  .arr-o {{ stroke: #E67E22; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}ao); }}
]]></style>
<defs>
  <marker id="{prefix}a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#666"/></marker>
  <marker id="{prefix}ab" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4A90D9"/></marker>
  <marker id="{prefix}ag" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#27AE60"/></marker>
  <marker id="{prefix}ao" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#E67E22"/></marker>
</defs>
'''


def box(x, y, w, h, fill="#f8f9fc", stroke="#c5cee0", rx=6):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )


def txt(x, y, s, cls="lb", anchor="middle"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{esc(s)}</text>'


def line(x1, y1, x2, y2, cls="arr"):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"/>'


def polyline(points, cls="arr"):
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polyline points="{pts}" class="{cls}" fill="none"/>'


def node(cx, cy, w, h, title, lines=(), fill="#f8f9fc", stroke="#c5cee0"):
    x, y = cx - w // 2, cy - h // 2
    s = box(x, y, w, h, fill, stroke)
    s += txt(cx, cy - (8 if lines else 0), title)
    for i, ln in enumerate(lines):
        s += txt(cx, cy + 10 + i * 14, ln, "dt")
    return s


def node_top(cx, top, w, h, title, lines=(), fill="#f8f9fc", stroke="#c5cee0"):
    """Box anchored by top edge — clearer multi-line layout."""
    x = cx - w // 2
    s = box(x, top, w, h, fill, stroke)
    if lines:
        s += txt(cx, top + 22, title)
        for i, ln in enumerate(lines):
            s += txt(cx, top + 38 + i * 14, ln, "dt")
    else:
        s += txt(cx, top + h // 2 + 5, title)
    return s


def branch_down(cx, bus_y, child_x, child_top, cls="arr"):
    """Trunk → horizontal bus → drop into child box top."""
    mid = bus_y + 14
    return polyline([(cx, bus_y), (cx, mid), (child_x, mid), (child_x, child_top)], cls)


def write(path: Path, body: str, w: int, h: int, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = svg_header(w, h, prefix) + body + "</svg>"
    path.write_text(content, encoding="utf-8")
    ET.parse(path)
    print("OK", path.relative_to(REPO))


def gen_rlvr_branch() -> None:
    w, h = 920, 248
    cx = 460
    base_top, base_h = 44, 44
    base_bottom = base_top + base_h
    bus_y = 118
    child_top, child_h, child_w = 156, 58, 210
    xs = (160, 460, 760)

    s = txt(cx, 22, "V3-Base 后训练分叉", "t")
    # connectors under nodes
    s += line(cx, base_bottom, cx, bus_y, "arr-b")
    s += branch_down(cx, bus_y, xs[0], child_top, "arr-o")
    s += branch_down(cx, bus_y, xs[1], child_top, "arr-o")
    s += branch_down(cx, bus_y, xs[2], child_top, "arr-g")
    # label above R1 branch bus segment (not on the stroke)
    s += txt((cx + xs[2]) // 2, bus_y + 8, "RLVR / GRPO", "an")
    # nodes
    s += node_top(cx, base_top, 220, base_h, "V3-Base", fill="#eef4fc", stroke="#4A90D9")
    s += node_top(
        xs[0], child_top, child_w, child_h,
        "V3.1 Hybrid", ["另一套 post-train"], fill="#fff8ee", stroke="#E67E22",
    )
    s += node_top(
        xs[1], child_top, child_w, child_h,
        "V3.2", ["DSA + 混合奖励 RL"], fill="#fff8ee", stroke="#E67E22",
    )
    s += node_top(
        xs[2], child_top, child_w, child_h,
        "R1（专用推理）", ["RLVR + GRPO · 架构同 V3"], fill="#f0faf0", stroke="#27AE60",
    )
    write(RL_FIG / "rlvr-posttrain-branch.svg", s, w, h, "rlv")


def gen_ess_kv_tree() -> None:
    w, h = 640, 260
    s = txt(320, 22, "KV-offload 演进（ESS 视角）", "t")
    nodes = [
        (320, 58, 260, 40, "V3 / V3.1 同质 MLA latent"),
        (320, 118, 320, 40, "V3.2 DSA：Indexer + Latent 双 Cache"),
        (160, 178, 200, 40, "ESS（Latent offload）"),
        (480, 178, 220, 40, "Index Share（indexer 复用）"),
        (320, 228, 280, 40, "V4 异构 KV + HiSparse"),
    ]
    for cx, cy, nw, nh, label in nodes:
        s += node(cx, cy, nw, nh, label, fill="#f8f9fc", stroke="#94a3b8")
    s += line(320, 78, 320, 98)
    s += polyline([(320, 138), (320, 158), (160, 158), (160, 158)], "arr")
    s += polyline([(320, 138), (320, 158), (480, 158), (480, 158)], "arr")
    s += polyline([(160, 198), (160, 208), (320, 208), (320, 208)], "arr")
    s += polyline([(480, 198), (480, 208), (320, 208), (320, 208)], "arr")
    write(DSA_DIAGRAMS / "ess-kv-lineage-tree.svg", s, w, h, "ekt")
    write(FIGURES / "ess" / "ess-kv-lineage-tree.svg", s, w, h, "ekt2")


def gen_v1_v3_mla_evolution() -> None:
    w, h = 920, 160
    s = txt(460, 22, "注意力：GQA -> MLA", "t")
    s += node(230, 88, 380, 72, "V1 67B: GQA（8 KV 头）", ["减 KV，仍按头存完整 K/V"], "#fff8ee", "#E67E22")
    s += line(460, 88, 460, 118, "arr")
    s += txt(460, 108, "压缩 KV 维", "an")
    s += node(690, 88, 380, 72, "V2/V3: MLA", ["c^KV [512] + RoPE [64]", "O(n_h d_h) -> O(d_c + d_h^R)"], "#eef4fc", "#4A90D9")
    write(DIAGRAMS / "v1-v3-mla-evolution.svg", s, w, h, "mla")


def gen_v1_v3_capability_timeline() -> None:
    w, h = 920, 320
    s = txt(460, 22, "能力代际（V1 -> V3 及后续）", "t")
    steps = [
        (460, 58, "2024-01 V1", "稠密 7B/67B · 4K · GQA · 2T"),
        (460, 118, "2024-05 V2", "236B/21B · MLA · MoE · 8.1T"),
        (460, 178, "2024-12 V3", "671B/37B · 256/8 MoE · 14.8T"),
    ]
    for cx, cy, title, sub in steps:
        s += node(cx, cy, 420, 48, title, [sub], "#eef4fc", "#4A90D9")
    s += line(460, 82, 460, 94)
    s += txt(520, 100, "MLA + MoE + 128K", "an")
    s += line(460, 142, 460, 154)
    s += txt(520, 160, "aux-loss-free + MTP", "an")
    s += polyline([(460, 202), (460, 232), (220, 232), (220, 248)], "arr-g")
    s += polyline([(460, 202), (460, 232), (460, 248)], "arr-g")
    s += polyline([(460, 202), (460, 232), (700, 232), (700, 248)], "arr-g")
    s += node(220, 278, 200, 44, "R1", ["架构不变 · RLVR"], "#f0faf0", "#27AE60")
    s += node(460, 278, 180, 44, "V3.1 Hybrid", fill="#fff8ee", stroke="#E67E22")
    s += node(700, 278, 180, 44, "V3.2 + DSA", fill="#fff8ee", stroke="#E67E22")
    write(DIAGRAMS / "v1-v3-capability-timeline.svg", s, w, h, "v13")


def gen_r1_pipeline() -> None:
    w, h = 920, 200
    s = txt(460, 22, "DeepSeek-R1 训练 pipeline 分叉", "t")
    s += node(160, 88, 180, 48, "DeepSeek-V3 Base", fill="#eef4fc", stroke="#4A90D9")
    s += polyline([(250, 88), (340, 88), (340, 58), (620, 58), (620, 66)], "arr-g")
    s += txt(480, 50, "纯 RL", "an")
    s += node(620, 88, 220, 48, "DeepSeek-R1-Zero", fill="#f0faf0", stroke="#27AE60")
    s += polyline([(250, 88), (340, 88), (340, 118), (620, 118), (620, 110)], "arr-o")
    s += txt(420, 128, "冷启动 SFT -> RL -> 拒绝采样 SFT -> RL", "an")
    s += node(620, 148, 200, 48, "DeepSeek-R1", fill="#f0faf0", stroke="#27AE60")
    write(R1_DIAG / "deepseek-r1-training-branch.svg", s, w, h, "r1b")


def gen_r1_refinement_chain() -> None:
    w, h = 920, 120
    s = txt(460, 22, "拒绝采样后处理链", "t")
    xs = [100, 280, 440, 620, 780]
    labels = ["Dev-2 / R1-Zero / V3", "采样", "Filter", "Refine(V3+Human)", "SFT 混合数据"]
    for x, lb in zip(xs, labels):
        s += node(x, 72, 140, 44, lb, fill="#f8f9fc", stroke="#94a3b8")
    for i in range(len(xs) - 1):
        s += line(xs[i] + 70, 72, xs[i + 1] - 70, 72)
    write(R1_DIAG / "deepseek-r1-refinement-chain.svg", s, w, h, "r1c")


def gen_gpu_sm_hierarchy() -> None:
    w, h = 520, 280
    cx = 260
    s = txt(cx, 22, "GPU 存储与计算层级", "t")
    s += node(cx, 58, 360, 40, "HBM（片外显存，容量大、带宽有限）", fill="#fff0f0", stroke="#e8a0a0")
    s += line(cx, 78, cx, 92)
    s += txt(cx + 80, 86, "DMA / 内存控制器", "an")
    s += node(cx, 112, 320, 40, "L2 Cache（片上，全芯片共享）", fill="#fff8ee", stroke="#fdba74")
    s += line(cx, 132, cx, 146)
    s += node(cx, 168, 400, 40, "SM x N（H100 ~132 SM）", fill="#eef4fc", stroke="#4A90D9")
    children = [
        (100, 228, "寄存器 / Shared Memory"),
        (260, 228, "CUDA Core"),
        (420, 228, "Tensor Core"),
    ]
    s += polyline([(cx, 188), (cx, 200), (260, 200)], "arr")
    for x, _, lb in children:
        s += polyline([(260, 200), (x, 200), (x, 206)], "arr")
        s += node(x, 238, 150, 36, lb, fill="#f8f9fc", stroke="#cbd5e1")
    write(VERSIONS_FIG / "gpu-sm-hierarchy.svg", s, w, h, "gsm")


def gen_h2d_serial() -> None:
    w, h = 720, 100
    s = txt(360, 22, "无 Overlap：串行 PD 时序", "t")
    xs = [120, 360, 600]
    lbs = ["Indexer 完成", "H2D 完成", "Attention 开始"]
    for x, lb in zip(xs, lbs):
        s += node(x, 62, 160, 44, lb, fill="#fff8ee", stroke="#E67E22")
    s += line(200, 62, 280, 62)
    s += line(440, 62, 520, 62)
    s += txt(360, 88, "全串行，GPU 常在等 H2D", "st")
    write(VERSIONS_FIG / "h2d-pipeline-serial.svg", s, w, h, "h2d")


def gen_evolution_chain(title: str, steps: list[str], out: Path, prefix: str) -> None:
    w, h = 920, 60 + 56 * len(steps)
    s = txt(460, 22, title, "t")
    y = 48
    for i, step in enumerate(steps):
        s += node(460, y + 24, 680, 44, step, fill="#f8f9fc", stroke="#94a3b8")
        if i < len(steps) - 1:
            s += line(460, y + 46, 460, y + 56)
        y += 56
    write(out, s, w, h, prefix)


def gen_csa_hca_chain() -> None:
    gen_evolution_chain(
        "Attention / KV 演进链",
        [
            "MLA（per-token latent KV）",
            "DSA（indexer + top-k 稀疏）",
            "CSA（4:1 块压缩 + 压缩序列 top-k）",
            "HCA（128:1 重压缩 + 短序列 dense）",
            "+ SWA / Indexer / tail（V4 异构 KV 五类对象）",
        ],
        VERSIONS_FIG / "csa-hca-evolution-chain.svg",
        "csa",
    )


def gen_hash_moe_chain() -> None:
    gen_evolution_chain(
        "MoE / FFN 演进链",
        [
            "稠密 FFN (V1)",
            "DeepSeekMoE：细粒度 routed + shared (V2)",
            "aux-loss-free + L_Bal：sigmoid + bias (V3)",
            "Hash MoE（前几层 hash）+ FP4 routed expert (V4)",
        ],
        VERSIONS_FIG / "hash-moe-evolution-chain.svg",
        "hmoe",
    )


def gen_dspark_draft_parallel() -> None:
    w, h = 720, 200
    s = txt(360, 22, "DSpark：离线能力 vs 在线加速", "t")
    s += node(360, 58, 280, 40, "新样本", fill="#f8f9fc", stroke="#94a3b8")
    s += polyline([(360, 78), (360, 98), (180, 98), (180, 108)], "arr")
    s += polyline([(360, 78), (360, 98), (540, 98), (540, 108)], "arr-o")
    s += node(180, 138, 220, 48, "SFT target（可选）", ["改能力 p"], "#eef4fc", "#4A90D9")
    s += node(540, 138, 220, 48, "蒸馏 draft", ["target 冻结 · q->p"], "#f0faf0", "#27AE60")
    s += txt(360, 118, "并行", "an")
    s += node(360, 178, 420, 40, "在线：draft 猜 + target verify -> 加速（仍服从 p）", fill="#fff8ee", stroke="#E67E22")
    write(RL_FIG / "dspark-draft-target-parallel.svg", s, w, h, "dsp")


def gen_v1_bbpe_pipeline() -> None:
    w, h = 920, 100
    lbs = ["原始语料", "去重", "过滤", "重混", "BBPE 训练 / 预训练"]
    s = txt(460, 22, "V1 数据流程", "t")
    xs = [80, 220, 360, 500, 680, 840]
    for i in range(len(lbs)):
        x = 80 + i * 190
        s += node(x, 62, 150, 44, lbs[i], fill="#f8f9fc", stroke="#94a3b8")
        if i < len(lbs) - 1:
            s += line(x + 75, 62, x + 115, 62)
    write(FIGURES / "v1" / "bbpe" / "v1-data-pipeline.svg", s, w, h, "bbpe")


def gen_moe_bal_pipeline() -> None:
    w, h = 720, 220
    s = txt(360, 22, "L_Bal 计算流水线（每层 MoE）", "t")
    rows = [
        (72, "前向：s+b 选 top-K -> expert 计算"),
        (112, "并行积累 count_i（f_i）与 s'_{i,t}（P_i）"),
        (152, "聚合：f_i = (N_r/K_r T)*count_i"),
        (192, "L_Bal = alpha * sum_i f_i P_i -> 反传 router"),
    ]
    for y, lb in rows:
        s += node(360, y, 620, 36, lb, fill="#f8f9fc", stroke="#cbd5e1")
    for y in (90, 130, 170):
        s += line(360, y + 18, 360, y + 28)
    write(VERSIONS_FIG / "moe-bal-loss-pipeline.svg", s, w, h, "mbal")


def gen_v1_bbpe_encode() -> None:
    steps = [
        "UTF-8 文本",
        "pre-tokenization（字符类切分 + 数字 digit 化）",
        "byte 序列",
        "BBPE merge 表查最长匹配",
        "token id 序列（≤ 100,015 有效 id）",
        "embedding[102,400] 查表 → 模型",
    ]
    gen_evolution_chain("BBPE 编码流程（推理侧）", steps, FIGURES / "v1" / "bbpe" / "bbpe-encode-flow.svg", "benc")


def gen_v3_mla_latent_kv_offload() -> None:
    """V3/V3.1 homogeneous MLA latent cache + generic CPU offload (§5.1)."""
    w, h = 920, 300
    cx = 460
    s = txt(cx, 22, "V3 / V3.1：同质 MLA Latent-Cache 与 KV offload", "t")
    s += txt(cx, 40, "单一流 c^KV + k^R 序列 · 随 L 线性占 HBM · 可选整条搬 CPU", "st")

    # decode step
    s += node(130, 110, 200, 72, "Decode 第 t 步", [
        "MLA 层写出新 entry",
        "c^KV[t] + k^R[t]",
    ], "#eef4fc", "#4A90D9")
    s += line(230, 110, 290, 110, "arr-b")

    # GPU HBM latent stream
    s += box(290, 68, 380, 104, "#fff8ee", "#E67E22")
    s += txt(480, 84, "GPU HBM · Latent-Cache（单层示意）", "lb")
    s += txt(480, 100, "同质 latent 向量流 · 每步 append 一条", "dt")
    # mini cache cells
    xs = [340, 400, 460, 520, 580, 640]
    for i, x in enumerate(xs):
        fill = "#fde68a" if i == len(xs) - 1 else "#fef3c7"
        s += box(x, 112, 48, 36, fill, "#d97706", rx=4)
        s += txt(x + 24, 128, f"t{i+1}" if i < 5 else "tL", "dt")
    s += txt(700, 128, "…", "dt")
    s += txt(480, 158, "容量 O(L) 线性增长 -> 占满 HBM", "an")

    # CPU offload (dashed)
    s += (
        '<rect x="720" y="68" width="170" height="104" rx="6" '
        'fill="#f0faf0" stroke="#27AE60" stroke-width="1.5" stroke-dasharray="6 4"/>'
    )
    s += txt(805, 84, "CPU DRAM", "lb")
    s += txt(805, 102, "通用 KV offload", "dt")
    s += txt(805, 116, "FlexGen / vLLM", "dt")
    s += txt(805, 130, "CPU offload 等", "dt")
    s += (
        '<line x1="670" y1="120" x2="720" y2="120" stroke="#27AE60" '
        'stroke-width="1.5" stroke-dasharray="6 4" marker-end="url(#v3offag)"/>'
    )
    s += (
        '<line x1="720" y1="140" x2="670" y2="140" stroke="#27AE60" '
        'stroke-width="1.5" stroke-dasharray="6 4" marker-end="url(#v3offag)"/>'
    )
    s += txt(695, 108, "H2D", "an")
    s += txt(695, 152, "D2H", "an")

    # notes
    s += box(80, 210, 760, 68, "#f8f9fc", "#94a3b8")
    s += txt(460, 232, "与 V3.2 ESS 不同：无 Indexer/Latent 拆分 · 无稀疏 entry prefetch", "dt")
    s += txt(460, 252, "MLA 自定义 layout -> 很多引擎不支持标准 offload", "dt")
    s += txt(460, 270, "瓶颈：latent 线性涨满 HBM -> decode batch size 受限", "dt")

    out = FIGURES / "ess" / "v3-mla-latent-kv-offload.svg"
    write(out, s, w, h, "v3off")
    write(DSA_DIAGRAMS / "v3-mla-latent-kv-offload.svg", s, w, h, "v3off2")


def gen_aux_loss_free_bias() -> None:
    w, h = 520, 140
    s = txt(260, 22, "aux-loss-free：batch-wise bias 更新", "t")
    s += node(260, 62, 360, 40, "若 expert i 过载（overload）", fill="#fff0f0", stroke="#e8a0a0")
    s += txt(260, 88, "b_i -= γ", "an")
    s += node(260, 112, 360, 40, "若 expert i 欠载（underload）", fill="#f0faf0", stroke="#27AE60")
    s += txt(260, 138, "b_i += γ", "an")
    write(VERSIONS_FIG / "aux-loss-free-bias-update.svg", s, w, h, "alb")


def main() -> None:
    gen_rlvr_branch()
    gen_ess_kv_tree()
    gen_v1_v3_mla_evolution()
    gen_v1_v3_capability_timeline()
    gen_r1_pipeline()
    gen_r1_refinement_chain()
    gen_gpu_sm_hierarchy()
    gen_h2d_serial()
    gen_csa_hca_chain()
    gen_hash_moe_chain()
    gen_dspark_draft_parallel()
    gen_v1_bbpe_pipeline()
    gen_v1_bbpe_encode()
    gen_moe_bal_pipeline()
    gen_aux_loss_free_bias()
    gen_v3_mla_latent_kv_offload()


if __name__ == "__main__":
    main()
