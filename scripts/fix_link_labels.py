#!/usr/bin/env python3
"""Replace markdown link *labels* that look like file paths with human titles."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
PATH_LABEL_MD = re.compile(
    r"(?:\.\./|\./|/)?"
    r"[\w./-]+\.md"
    r"(?:\s*§.*)?$",
    re.IGNORECASE,
)
# label is essentially a filename (optional directory prefix)
FILENAME_LABEL = re.compile(
    r"^(?:[\w./-]+/)?"
    r"[\w.-]+\.(?:md|mdc|pdf|svg|png|jpe?g|gif|webp|py)$",
    re.IGNORECASE,
)
SECTION_SUFFIX = re.compile(r"\s*(§.+)$")

ASSET_LABEL_MAP: dict[str, str] = {
    "Thinking_with_Visual_Primitives.pdf": "Visual Primitives 原文 PDF",
    "deepseek-r1-2501.12948.pdf": "DeepSeek-R1 论文 PDF",
    "src/deepseek-r1-2501.12948.pdf": "DeepSeek-R1 论文 PDF",
    "fig-1-token-efficiency.png": "Figure 1 — Token 效率",
    "fig-2-architecture-pipeline.png": "Figure 2 — 架构与训练 pipeline",
    "fig-3-cold-start-counting.png": "Figure 3 — Cold-start 计数",
    "table-1-benchmark.png": "Table 1 — Benchmark 对比",
    "grpo-vs-ppo.svg": "GRPO vs PPO 对照图",
    "mtp-fusion-scheme.svg": "MTP 融合 scheme 图",
    "mtp-speculative.svg": "MTP 投机解码总览图",
    "mtp-draft-chain-depth.svg": "MTP draft 链深度图",
    "v4-hetero-kv.svg": "V4 异构 KV 总览图",
    "deepseek-r1-training-pipeline.svg": "R1 训练 pipeline 流程图",
    "training-pipeline-reference.png": "R1 训练 pipeline 参考图",
    "deepseek-version-lines-crossrefs.md": "版本演进线文档引用约定",
    "deepseek-version-lines-crossrefs.mdc": "版本演进线文档引用约定",
    "cxl-2603.10087.pdf": "CXL 论文 PDF",
    "engram_demo_v1.py": "Engram demo 脚本",
    "docs/engram/engram_demo_v1.py": "Engram demo 脚本",
    "../../../../engram/engram_demo_v1.py": "Engram demo 脚本",
}


def h1_title(md_path: Path) -> str | None:
    if not md_path.is_file():
        return None
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def resolve_target(from_file: Path, url: str) -> Path | None:
    if not url or url.startswith(("http://", "https://", "mailto:", "#", "data:")):
        return None
    path_part = url.split("#", 1)[0].split("?", 1)[0]
    if not path_part:
        return None
    target = (from_file.parent / path_part).resolve()
    try:
        target.relative_to(DOCS.resolve())
    except ValueError:
        return None
    return target


def fix_label(label: str, title: str) -> str:
    m = SECTION_SUFFIX.search(label)
    suffix = m.group(1) if m else ""
    return f"{title}{suffix}"


def asset_title(label: str, url: str) -> str | None:
    stripped = label.strip()
    if stripped in ASSET_LABEL_MAP:
        return ASSET_LABEL_MAP[stripped]
    base = Path(stripped.replace("\\", "/").split("/")[-1]).name
    if base in ASSET_LABEL_MAP:
        return ASSET_LABEL_MAP[base]
    path_part = url.split("#", 1)[0].split("?", 1)[0]
    url_base = Path(path_part.replace("\\", "/").split("/")[-1]).name
    if url_base in ASSET_LABEL_MAP and (
        stripped == url_base or stripped.endswith("/" + url_base) or stripped == path_part
    ):
        return ASSET_LABEL_MAP[url_base]
    return None


def should_fix_md(label: str, url: str) -> bool:
    if url.startswith(("http://", "https://", "mailto:", "#")):
        return False
    if ".md" not in label and ".mdc" not in label:
        return False
    return bool(PATH_LABEL_MD.match(label.strip()))


def should_fix_asset(label: str, url: str) -> bool:
    if url.startswith(("http://", "https://", "mailto:", "#")):
        return False
    return bool(FILENAME_LABEL.match(label.strip()))


def patch_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    n = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal n
        label, url = m.group(1), m.group(2)
        if should_fix_md(label, url):
            target = resolve_target(path, url)
            if target is not None:
                title = h1_title(target)
                if title and title != label:
                    n += 1
                    return f"[{fix_label(label, title)}]({url})"
        if should_fix_asset(label, url):
            title = asset_title(label, url)
            if title and title != label.strip():
                n += 1
                return f"[{fix_label(label, title)}]({url})"
        return m.group(0)

    new = LINK_RE.sub(repl, text)
    if n:
        path.write_text(new, encoding="utf-8")
    return n


def main() -> None:
    skip_dirs = {"node_modules", ".git"}
    skip_files = {
        DOCS / "reports" / "deepseek-doc-series-audit-20260627.md",
        DOCS / "material" / "meta" / "deepseek-version-lines-crossrefs.md",
    }
    total = 0
    for md in sorted(DOCS.rglob("*.md")):
        if any(p in skip_dirs for p in md.parts):
            continue
        if md in skip_files:
            continue
        c = patch_file(md)
        if c:
            print(f"  {c:3d}  {md.relative_to(REPO)}")
            total += c
    print(f"OK fix_link_labels: {total} links")


if __name__ == "__main__":
    main()
