# Design Decisions — The Four Tricky Parts

This document describes how the Resume Shortlisting Engine actually solves the four hard
problems from the problem statement. It reflects the shipped code in `engine/parser.py`,
`engine/scorer.py`, and `engine/extractor_llm.py` — not theory.

**Stack.** The engine is pure Python. Stage 1 parsing is fully offline: PDF text and block
geometry come from **PyMuPDF** (`fitz.get_text("dict")`), with **pdfplumber** available as a
secondary extraction aid and **pytesseract** (over PIL, driving the Tesseract binary) for OCR
of scanned pages. Stage 2 scoring layers an optional LLM cascade on top of an offline rules
floor: the primary tier is **Groq `llama-3.3-70b-versatile`**, falling back to
**Google Gemini Flash** (`gemini-1.5-flash`) as a cloud backup, then to a **local Qwen2.5:7B via
Ollama**, and finally to rules-only if every tier is unavailable. Every LLM call runs at
**temperature 0** with a uniform hard 3-second timeout, and results are cached to disk keyed by a
SHA-256 **resume hash** (`resume_hash + jd_id`) so reruns are deterministic and free of new network
calls. Stage 1 never needs an API key; the cascade simply degrades to the offline rules baseline
when keys or models are absent.

## 1. PDF Layout Chaos (multi-column)

Multi-column resumes are reconstructed in `parser.reconstruct_reading_order` using bounding-box
x-clustering rather than trusting PyMuPDF's raw block stream. Each text block from
`get_text("dict")` carries its bbox `(x0, y0, x1, y1)`; the function computes the page mid-x gutter
and a dead-band of ±5% of page width (`_GUTTER_MARGIN_FRAC`) around it. Blocks wider than 70% of the
page (`_FULLWIDTH_FRAC`) are treated as full-width banners/headers and excluded from the column tally
so a wide title bar cannot mask a real split beneath it. Remaining blocks vote as left-evidence
(x0 clearly left of the gutter) or right-evidence (clearly right). The page is judged two-column only
when each side holds at least 25% of the blocks (`_MIN_SIDE_SHARE`) **and** the right side has at
least 2 blocks (`_MIN_RIGHT_BLOCKS`); otherwise it falls back to natural top-to-bottom `y0` order.
When a genuine split is confirmed, every block is assigned to a side by its center-x and the code
emits the **entire left column top-to-bottom, then the entire right column top-to-bottom** — never
reading across the gutter, which is exactly the interleaving bug that splices a sidebar's skills into
the main body's sentences. Known failure cases: layouts with **3 or more columns** (only a single
mid-x gutter is modelled, so a third column gets folded into left or right), **floating Canva-style
text boxes** that don't respect a clean two-column grid, and **overlapping bounding boxes** where
center-x assignment becomes ambiguous.

## 2. The Scanned Resume (OCR) — Bonus B: automatic OCR fallback

Scanned/image-only resumes are handled by an automatic OCR fallback wired into
`parser.parse_resume` around `parser.ocr_page_text`. After normal extraction, the pipeline counts
words; if the recovered text is below 50 words (`config.OCR_MIN_WORDS`), the PDF is almost certainly
image-only or near-empty, so each page is rasterized via PyMuPDF `page.get_pixmap(dpi=config.OCR_DPI)`
at ~200 DPI, decoded into a PIL image from the pixmap's PNG bytes, and passed through
`pytesseract.image_to_string`. The code deliberately distinguishes **two distinct failure modes**,
and both are flagged `Failed` yet still appear in the output — never silently dropped (Req 7.4):
first, an **"OCR execution error"** where `ocr_page_text` itself raises (Tesseract binary missing,
`pytesseract`/PIL not installed, a corrupted image, or an unsupported format) — the exception is
caught, `parse_flag` is set to `Failed`, and `error_reason` becomes
`"OCR execution error: <underlying message>"` preserving the original error for debugging; second,
**"OCR yielded insufficient text"** where OCR runs cleanly but the recovered text is still below 50
words after OCR, giving `error_reason = "OCR yielded insufficient text"`. This split lets the parse
report tell a broken OCR environment apart from a genuinely unreadable scan, and in both cases
`used_ocr` is recorded so the outcome is a first-class, visible result.

## 3. Skill Extraction from Unstructured Text

Skills are extracted as a **union of an offline rules tier and an LLM tier** so nothing depends on a
neatly labelled "Skills:" section. The rules tier (`parser._extract_skills`, consumed by
`scorer.rules_baseline`) normalizes the **entire reconstructed resume text** and does a word-boundary
scan against the full synonym vocabulary in `engine/synonyms.py`, mapping every hit through
`canonical()` — so a skill named only inside project or experience prose (e.g. "Flask" mentioned in a
project bullet) is still caught, and JD skills are matched exact/synonym against those canonical
forms. On top of that, the LLM tier (`extractor_llm.enhance`) reads the raw text at temperature 0 and
returns both additional extracted skills the vocabulary misses and per-JD-skill match labels,
including **implicit** matches where a skill is demonstrated without being named (e.g. "built a REST
API in Flask" ⇒ implicit REST). The merge in `scorer._merge_matches` never lets an LLM result drop a
required skill below the credit the offline floor already awarded, so the LLM can only raise a score.
What it would still miss: skills that are **neither in the synonym vocabulary nor recognized by the
LLM**, and **obscure abbreviations** or house-specific tool names that don't resolve to any canonical
skill.

## 4. Parse Quality Affects Confidence

Parse quality is coupled to confidence in `scorer.resolve_confidence`, keeping the parser's job (which
required fields were recovered → Clean/Partial/Failed) separate from the scorer's job (how much to
trust the result). The parse flag acts as a **hard ceiling**: `Clean` allows up to High, `Partial`
caps at Medium no matter how strong the LLM match looks, and `Failed` yields **no numeric score plus a
human-review flag**. The final confidence is the **lower of two ceilings** — the parse-quality ceiling
and the tier ceiling (`Cloud` → High, `Local`/`Rules` → Medium) — via `_min_confidence`. If the
deciding required-skill evidence is **implicit or partial**, confidence drops one further level
(floored at Low) through `_drop_one_level`. A dedicated conflict rule handles the
**partial-parse + high-score** case: in `score_candidate`, a `Partial` parse scoring ≥ 70 forces
confidence down to Medium and sets `human_review = True`, so a confident-looking LLM can never mask a
shaky extraction. Confidence is strictly three-level — **High / Medium / Low** — never an in-between
label like "Medium-High".
