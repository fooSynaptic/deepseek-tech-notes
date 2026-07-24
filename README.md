# DeepSeek Mechanism Atlas

> An **atlas of DeepSeek mechanisms** — architecture, sparse attention, speculative decoding, MoE routing, and adjacent inference infra — organized as a bidirectional wiki + mdBook.
>
> Unofficial. Not affiliated with DeepSeek.

**[Read online (mdBook)](https://fooSynaptic.github.io/deepseek-mechanism-atlas/)** · **[中文导读 / Chinese README](docs/README.md)**

If this atlas helps you, please **Star** to bookmark for later.

---

## Paper catalog

| Series | Article | Keywords |
|--------|---------|----------|
| **V1** | [DeepSeek-LLM](docs/versions/v1.md) | scaling laws, dense base |
| **V2** | [DeepSeek-V2](docs/versions/v2.md) | MLA, DeepSeekMoE, 128K |
| **V3** | [DeepSeek-V3](docs/versions/v3.md) | 671B MoE, MTP, aux-loss-free |
| **R1** | [DeepSeek-R1](docs/versions/r1.md) | reasoning model, RLVR, GRPO |
| **V3.2** | [DeepSeek-V3.2](docs/versions/v3-2.md) | sparse attention, DSA |
| **V4** | [DeepSeek-V4](docs/versions/v4.md) | CSA/HCA, mHC, 1M context |
| **DSA** | [Sparse attention](docs/versions/dsa-sparse-attention.md) | indexer, top-k, lightning indexer |
| **DSpark** | [Speculative decoding](docs/versions/dspark-speculative-decoding.md) | speculative decoding, MTP fusion |
| **MLA** | [Latent attention](docs/versions/mla-latent-attention.md) | latent KV, KV cache compression |
| **MoE** | [DeepSeekMoE](docs/versions/deepseek-moe.md) | routed experts, shared experts |
| **Index Share** | [IndexCache](docs/versions/index-share.md) | cross-layer index reuse, infra patch |
| **ESS** | [Latent offload](docs/versions/ess-latent-cache-offload.md) | CPU KV offload |
| **CSA / HCA** | [Mixed compression attention](docs/versions/csa-hca-mixed-attention.md) | 4:1 sparse + 128:1 dense |
| **Hash MoE** | [Hash MoE + FP4](docs/versions/hash-moe-fp4.md) | hash routing, FP4 quantization |

**Full index:** [Chinese docs home · article list](docs/README.md#文章) · [Version index](docs/versions/README.md) · [Online book](https://fooSynaptic.github.io/deepseek-mechanism-atlas/)

**Search keywords:** `DeepSeek paper` · `DeepSeek-V2` · `DeepSeek-V3` · `DeepSeek-V4` · `DeepSeek R1` · `reasoning model` · `sparse attention` · `DSpark` · `MLA` · `MoE` · `llm-papers` · `paper explanation`

---

## Diagrams

Figures live under [`diagrams/`](diagrams/) and are linked from articles. Prefer the online book or IDE Preview for rendering. Selected entry points:

| Diagram | Related reading |
|---------|-----------------|
| [Version lineage](diagrams/deepseek-version-lineage.svg) | [Evolution overview](docs/reports/deepseek-version-lineage-20260625.md) |
| [MLA forward flow](diagrams/mla-forward-flow.svg) | [MLA notes](docs/versions/mla-latent-attention.md) |
| [GRPO vs PPO](diagrams/grpo-vs-ppo.svg) | [R1](docs/versions/r1.md) · [RLVR](docs/versions/rlvr.md) |
| [MTP fusion](diagrams/mtp-fusion-scheme.svg) | [DSpark](docs/versions/dspark-speculative-decoding.md) |
| [DSpark speculative](diagrams/dspark-speculative.svg) | [Speculative decoding](docs/versions/dspark-speculative-decoding.md) |

---

## Recommended reading

These notes form a **bidirectional wiki** — every article links back at the top and forward in the body. To get the most out of that navigation, use one of these (not GitHub’s in-repo blob preview):

| Mode | When | Navigation |
|------|------|------------|
| **IDE Preview** (VS Code / Cursor) | Cloned repo, deep reading or editing | Click `←` back-links and in-text links; split preview or preview history — **best for forward / back references** |
| **[GitHub Pages (mdBook)](https://fooSynaptic.github.io/deepseek-mechanism-atlas/)** | Online, no clone | Same math/diagram rendering as the IDE; use the browser **Back / Forward** buttons to retrace your reading path |

**Either IDE Preview or Pages works.** Edit and PR in `docs/` as usual.

### Why an online book (not GitHub Preview)?

**Local IDE Preview** (VS Code / Cursor) and **GitHub’s in-repo `.md` Preview** use different Markdown + math renderers — blockquotes, `$...$` / `$$...$$`, and inline math inside links often look wrong on GitHub even when they look fine in the IDE. Source Markdown is **not rewritten** to chase GitHub Preview; instead, `docs/` is built into an **[mdBook site on GitHub Pages](https://fooSynaptic.github.io/deepseek-mechanism-atlas/)** (KaTeX, same `$...$` source as the IDE). Use that for online reading; use the repo for editing and PRs.

---

## What this repo is

This project follows DeepSeek's open-model line **V1 → V2 → V3 → R1 / V3.2 → V4**, and unpacks **most** (not every) major technical reports into readable walkthroughs: architecture changes, training/inference tricks, formulas, and how versions relate.

**Differentiation** — vs scattered blog posts: unified layout, bidirectional wiki navigation, SVG formula diagrams, per-paper takeaway sections, and a living mdBook mirror.

Coverage includes:

- **Core DeepSeek releases** — MLA, MoE routing, MTP, DSA, CSA/HCA, mHC, Hash MoE, V4 KV layout, etc.
- **V4 inference stack** — **[DSpark](docs/versions/dspark-speculative-decoding.md)** speculative decoding, HiSparse, disk prefix cache.
- **Adjacent infra work** layered on DeepSeek checkpoints — **[Index Share / IndexCache](docs/versions/index-share.md)** (Tsinghua + Zhipu) and **[ESS](docs/versions/ess-latent-cache-offload.md)** latent-cache offload (Baidu BaiGe), with a dedicated **infrastructure** thread alongside algorithm and MoE.

Organized as wiki-style articles, SVG diagrams, and a book-style layout under [《ds-技术报告》/](《ds-技术报告》/01-总览/01-版本演进总览.md). For full Chinese navigation and article list, use the **[Chinese README](docs/README.md)** or the **[online mdBook](https://fooSynaptic.github.io/deepseek-mechanism-atlas/)**.

### Why reading here feels smooth

This repo is built for **bidirectional navigation**: every article, deep-dive, and Q&A page links **back** to where you came from — the Chinese home, this English homepage, the evolution hub, or the parent section. Follow a link into DSA logic, MTP fusion, or Engram notes; when you are done, one click returns you to the article or index you started from. No dead ends, no guessing how to resume the thread.

In **[IDE Preview](#recommended-reading)**, click links to jump; on **[Pages](https://fooSynaptic.github.io/deepseek-mechanism-atlas/)**, use browser Back / Forward for the same effect.

> **Work in progress.** Summaries, mirroring, links, and diagrams are still being updated. Prefer arXiv / official PDFs cited at the top of each article. Broken links or errors — **issues welcome**.

---

## Start here

| | |
|--|--|
| **Online book (Pages)** | **[fooSynaptic.github.io/deepseek-mechanism-atlas](https://fooSynaptic.github.io/deepseek-mechanism-atlas/)** — or clone and use **IDE Preview** |
| **Chinese README** | [docs/README.md](docs/README.md) |
| **Evolution hub** | [Version lineage overview](docs/reports/deepseek-version-lineage-20260625.md) — algorithm / infrastructure / MoE threads |
| **Book mirror (repo)** | [《ds-技术报告》/01-总览/01-版本演进总览.md](《ds-技术报告》/01-总览/01-版本演进总览.md) |

---

## Repository layout

| Path | Role |
|------|------|
| [`docs/`](docs/) | **Source of truth** — edit articles here |
| [`docs/README.md`](docs/README.md) | **Chinese README / docs home** |
| [`《ds-技术报告》/`](《ds-技术报告》/) | **Book mirror** — generated by `build_book.py` (do not hand-edit) |
| [`book.toml`](book.toml) + [`theme/`](theme/) | mdBook config & CSS for GitHub Pages |
| [`scripts/build_pages.sh`](scripts/build_pages.sh) | `build_book` → `SUMMARY.md` → `mdbook build` |
| [`.github/workflows/pages.yml`](.github/workflows/pages.yml) | Deploy mdBook to GitHub Pages on push to `main` |

**Reading:** [Recommended reading](#recommended-reading) — **IDE Preview** or **[GitHub Pages mdBook](https://fooSynaptic.github.io/deepseek-mechanism-atlas/)**; not GitHub blob preview. See [Why an online book](#why-an-online-book-not-github-preview) for rendering details.

---

## Contributing & book layout

**Source of truth is `docs/`.** The folder [《ds-技术报告》/](《ds-技术报告》/) is a **generated book mirror** — do not edit those Markdown files by hand; they are overwritten by `build_book.py`.

When you add or move content:

1. **Write the article** under `docs/` (e.g. `docs/versions/`, `docs/dsa/`, `docs/reports/`, `docs/versions/qa/`).
2. **Register it for the book** in [`《ds-技术报告》/build_book.py`](《ds-技术报告》/build_book.py):
   - `CHAPTER_MAP` — map `docs/...` → book chapter path;
   - `READING_ORDER` — prev/next chapter navigation;
   - `QA_DESTINATIONS` — if it is a Q&A page (may mirror to multiple book folders);
   - `ASSET_MAP` — only if new figures need copying into the book tree.
3. **Add navigation** — blockquote top bar with `←` links back to the parent section / index (see existing articles); link the new page from the relevant overview or index.
4. **Add a takeaway section** at the top of each new paper article (3–5 bullet points; Chinese articles use `## 核心结论摘要`).
5. **Rebuild & check** (from repo root):

```bash
python3 《ds-技术报告》/build_book.py
python3 scripts/validate_refs.py
python3 scripts/validate_backlinks.py
```

Or run the full gate: `bash scripts/doc_series_gate.sh`.

To preview the **mdBook site** locally (requires [mdBook](https://rust-lang.github.io/mdBook/)):

```bash
bash scripts/build_pages.sh
# open mdbook-out/index.html
```

Wiki-style reading in `docs/` works without the book step; run `build_book.py` when the chapter should appear in [《ds-技术报告》](《ds-技术报告》/01-总览/01-版本演进总览.md) with rewritten links and chapter nav. Push to `main` rebuilds GitHub Pages automatically.

---

## License

| Scope | License |
|-------|---------|
| Notes, diagrams, book layout | [CC BY 4.0](LICENSE) |
| `scripts/` | [MIT](LICENSE-MIT) |
| `docs/engram/` | [Apache 2.0](docs/engram/LICENSE) |
| `docs/material/` mirrors | upstream / original paper terms |

DeepSeek papers, weights, and official code remain under **their own** licenses.
