#!/usr/bin/env bash
# Build mdBook site for GitHub Pages (run from repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> build_book.py"
python3 "《ds-技术报告》/build_book.py"

echo "==> gen_mdbook_summary.py"
python3 scripts/gen_mdbook_summary.py

echo "==> mdbook build"
mdbook build

echo "==> fix_mdbook_index_paths.py"
python3 scripts/fix_mdbook_index_paths.py

echo "OK mdbook-out/ ready"
