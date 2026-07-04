#!/usr/bin/env python3
"""Fix relative paths in mdBook root index.html.

mdBook copies the first SUMMARY chapter to book-root index.html. Paths like
``../01-总览/figures/...`` are correct for ``00-前言/02-中文导读.html`` but break
on the root index (resolves outside site-url on GitHub Pages).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "mdbook-out" / "index.html"

ATTR_RE = re.compile(r'(src|href)="\.\./')


def main() -> None:
    if not INDEX.is_file():
        raise SystemExit(f"missing {INDEX.relative_to(REPO)} — run mdbook build first")
    text = INDEX.read_text(encoding="utf-8")
    fixed = ATTR_RE.sub(r'\1="', text)
    fixed = fixed.replace('href="../../', 'href="')
    n = len(ATTR_RE.findall(text)) + text.count('href="../../')
    INDEX.write_text(fixed, encoding="utf-8")
    print(f"FIX index.html paths ({n} adjustments)")


if __name__ == "__main__":
    main()
