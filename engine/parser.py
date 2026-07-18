"""Stage 1 Parser — PDF text extraction, column reconstruction, OCR, fields.

This module owns all offline parsing (Req 1.1, 3–8). It is built incrementally
across tasks 2.1–2.6:

- 2.1: ``extract_text_blocks`` + ``raw_text_via_pymupdf`` — raw PyMuPDF
  text/block extraction with per-page self-guarding.
- 2.2 (this task): ``reconstruct_reading_order`` — multi-column bbox
  reconstruction (left column fully, then right column).
- 2.3: ``ocr_page_text`` — pytesseract OCR fallback.
- 2.4: ``extract_fields`` — regex/heuristic field extraction.
- 2.6: ``classify_parse_quality`` + ``parse_resume`` orchestration.

PyMuPDF (``fitz``) and other heavy dependencies are imported lazily inside the
functions that need them so this module imports cleanly even when those
packages are not installed (keeps downstream tasks unblocked).
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime

from engine import config, normalizer
from engine.models import Experience, ParseFlag, Project, ResumeData, TextBlock
from engine.synonyms import _SYNONYMS, canonical


# ---------------------------------------------------------------------------
# Lazy dependency loading
# ---------------------------------------------------------------------------


def _import_fitz():
    """Import PyMuPDF lazily, raising a clear error only at call time.

    Importing at module scope would break imports for other parser tasks in
    environments where PyMuPDF is not installed, so the import is deferred.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF text extraction. "
            "Install it with `pip install PyMuPDF`."
        ) from exc
    return fitz


# ---------------------------------------------------------------------------
# 2.1 — Text / block extraction
# ---------------------------------------------------------------------------


def extract_text_blocks(pdf_path: str) -> list[TextBlock]:
    """Extract text blocks from a PDF using PyMuPDF ``get_text("dict")``.

    Each block becomes a :class:`~engine.models.TextBlock` carrying its bbox
    ``(x0, y0, x1, y1)`` and concatenated span text. To keep blocks sorting
    sensibly across pages, a running per-page y-offset (accumulated from each
    page's height) is added to every block's y coordinates so page 2 always
    sorts below page 1.

    Self-guarding: per-page extraction is wrapped in try/except so a single
    malformed page cannot abort the whole document (Req 8.1). Empty blocks are
    skipped.
    """
    fitz = _import_fitz()

    blocks: list[TextBlock] = []
    doc = fitz.open(pdf_path)
    try:
        y_offset = 0.0
        for page in doc:
            try:
                page_dict = page.get_text("dict")
            except Exception:
                # One bad page must not abort the whole document.
                # Still advance the offset by the page height if available.
                try:
                    y_offset += float(page.rect.height)
                except Exception:
                    pass
                continue

            for block in page_dict.get("blocks", []):
                # Only text blocks carry "lines"; image blocks are skipped.
                lines = block.get("lines")
                if not lines:
                    continue

                parts: list[str] = []
                for line in lines:
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if span_text:
                            parts.append(span_text)

                text = " ".join(parts).strip()
                if not text:
                    continue

                bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
                x0, y0, x1, y1 = (float(v) for v in bbox)
                blocks.append(
                    TextBlock(
                        x0=x0,
                        y0=y0 + y_offset,
                        x1=x1,
                        y1=y1 + y_offset,
                        text=text,
                    )
                )

            # Accumulate this page's height so the next page sorts below it.
            try:
                y_offset += float(page.rect.height)
            except Exception:
                # Fall back to a large constant offset if height is unavailable.
                y_offset += 1000.0
    finally:
        doc.close()

    return blocks


def raw_text_via_pymupdf(pdf_path: str) -> str:
    """Return the naive full text of a PDF (``page.get_text()`` joined).

    Used later as a cheap word-count signal for the OCR trigger (task 2.3) and
    as a fallback when reading-order reconstruction is unnecessary. Per-page
    extraction is self-guarded so one bad page cannot abort the document.
    """
    fitz = _import_fitz()

    pages: list[str] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            try:
                pages.append(page.get_text())
            except Exception:
                # Skip a bad page rather than aborting the whole document.
                continue
    finally:
        doc.close()

    return "\n".join(pages)


# ---------------------------------------------------------------------------
# 2.2 — Multi-column reading-order reconstruction
# ---------------------------------------------------------------------------


# Fraction of ``page_width`` used as the dead-band around the mid-x gutter. A
# block whose x0 falls inside ``mid-x ± margin`` is considered "near the
# gutter" and is not counted as strong evidence of a distinct column.
_GUTTER_MARGIN_FRAC = 0.05

# A block wider than this fraction of the page is treated as full-width (a
# banner/header/section rule spanning both columns) rather than a column body.
_FULLWIDTH_FRAC = 0.7

# Minimum share of blocks each side must hold for a genuine two-column split,
# plus the minimum absolute block count required on the right side.
_MIN_SIDE_SHARE = 0.25
_MIN_RIGHT_BLOCKS = 2


def reconstruct_reading_order(blocks: list[TextBlock], page_width: float) -> str:
    """Reconstruct reading order across multi-column layouts (Req 6.1, 6.2).

    Detects a two-column layout via bounding-box x-clustering around the page
    mid-x gutter. When a genuine split is found, the *entire* left column is
    emitted top-to-bottom, followed by the *entire* right column top-to-bottom
    (Req 6.2) — never reading across, which would splice a sidebar into the
    body. Single-column pages (no clear gutter) fall back to natural
    top-to-bottom ``y0`` order (Design Decisions §1).

    Args:
        blocks: text blocks with bbox coordinates, as produced by
            :func:`extract_text_blocks`.
        page_width: the page width used to locate the mid-x gutter.

    Returns:
        The reconstructed text with block texts joined by newlines, or ``""``
        for an empty ``blocks`` list.
    """
    # Guard: nothing to reconstruct.
    if not blocks:
        return ""

    # A non-positive page width means we cannot locate a gutter; fall back to
    # a simple top-to-bottom (then left-to-right) ordering.
    if page_width <= 0:
        return _join_by_reading_y(blocks)

    mid_x = page_width / 2.0
    margin = page_width * _GUTTER_MARGIN_FRAC
    fullwidth_threshold = page_width * _FULLWIDTH_FRAC

    # Classify blocks by their x0 relative to the gutter band. Full-width blocks
    # are excluded from the left/right evidence tally so a wide header banner
    # does not mask a real two-column split beneath it.
    left_evidence = 0
    right_evidence = 0
    for b in blocks:
        if (b.x1 - b.x0) > fullwidth_threshold:
            continue
        if b.x0 < mid_x - margin:
            left_evidence += 1
        elif b.x0 > mid_x + margin:
            right_evidence += 1

    total = len(blocks)
    two_column = (
        left_evidence >= _MIN_SIDE_SHARE * total
        and right_evidence >= _MIN_SIDE_SHARE * total
        and right_evidence >= _MIN_RIGHT_BLOCKS
    )

    if not two_column:
        # Single-column: natural top-to-bottom reading order.
        return _join_by_reading_y(blocks)

    # Two-column: assign each block to a side by its center-x vs the gutter,
    # then emit the whole left column (y-order) before the whole right column
    # (y-order). This keeps each column contiguous (Req 6.2).
    left: list[TextBlock] = []
    right: list[TextBlock] = []
    for b in blocks:
        center_x = (b.x0 + b.x1) / 2.0
        if center_x <= mid_x:
            left.append(b)
        else:
            right.append(b)

    ordered = _sorted_by_reading_y(left) + _sorted_by_reading_y(right)
    return "\n".join(b.text for b in ordered)


def _sorted_by_reading_y(blocks: list[TextBlock]) -> list[TextBlock]:
    """Sort blocks top-to-bottom by ``y0``, breaking ties left-to-right."""
    return sorted(blocks, key=lambda b: (b.y0, b.x0))


def _join_by_reading_y(blocks: list[TextBlock]) -> str:
    """Join blocks in natural top-to-bottom (then left-to-right) order."""
    return "\n".join(b.text for b in _sorted_by_reading_y(blocks))


# ---------------------------------------------------------------------------
# 2.3 — OCR fallback
# ---------------------------------------------------------------------------


def word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in ``text``.

    Used as the OCR trigger threshold against ``config.OCR_MIN_WORDS`` (Req
    7.1) and, after OCR, to distinguish a successful-but-sparse scan (Req 7.3)
    from a genuine OCR execution error (Req 7.2). The caller (``parse_resume``,
    task 2.6) owns that distinction; this helper is a pure, dependency-free
    word tally.
    """
    if not text:
        return 0
    return len(text.split())


def ocr_page_text(pdf_path: str) -> str:
    """Render each PDF page to an image and OCR it with pytesseract.

    Every page is rasterized via PyMuPDF ``page.get_pixmap(dpi=config.OCR_DPI)``
    (~200 DPI, Design Decisions §2), decoded into a PIL image from the pixmap's
    PNG bytes, and passed through ``pytesseract.image_to_string``. The per-page
    texts are concatenated in page order and returned.

    ``pytesseract`` and ``PIL`` are imported lazily *inside* this function so
    the module still imports cleanly when those optional OCR dependencies are
    absent (they are only needed when the OCR path actually runs).

    Exceptions are deliberately **not** swallowed. Any genuine OCR *execution
    error* — ``pytesseract``/``PIL`` not installed, the tesseract binary
    missing, a corrupted image, an unsupported format — is allowed to
    propagate so the caller can catch it and record
    ``error_reason = "OCR execution error: <message>"`` (Req 7.2). The separate
    "insufficient text" case (successful OCR yielding < 50 words, Req 7.3) is
    decided by the caller via :func:`word_count`; this function simply returns
    whatever text OCR recovered or raises on failure.
    """
    from engine import config

    # Lazy imports: keep the module importable without the OCR stack installed.
    # A missing dependency raises ImportError here, which propagates to the
    # caller as an OCR execution error (Req 7.2) rather than a silent drop.
    import pytesseract
    from PIL import Image

    import io

    # Point pytesseract at the configured Tesseract binary when that path
    # actually exists (Windows installs are not on PATH by default). On hosts
    # where tesseract is already on PATH the configured path won't exist and we
    # leave pytesseract's default lookup untouched — keeping this portable.
    if config.TESSERACT_CMD and os.path.exists(config.TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

    fitz = _import_fitz()

    page_texts: list[str] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            # Rasterize the page at the configured DPI. Any rendering/decoding
            # failure (corrupted image, unsupported format) propagates as an
            # OCR execution error — never swallowed (Req 7.2, 7.4).
            pixmap = page.get_pixmap(dpi=config.OCR_DPI)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            page_texts.append(pytesseract.image_to_string(image))
    finally:
        doc.close()

    return "\n".join(page_texts)


# ---------------------------------------------------------------------------
# 2.4 — Regex / heuristic field extraction
# ---------------------------------------------------------------------------
#
# All extraction here is offline (Req 8.1) and NEVER fabricates a value: any
# field not confidently recovered from the text is returned as ``None`` (or an
# empty list for list-valued fields), per Req 4.2/4.3. Grade is captured as a
# raw string only — normalization to the 10-pt scale is deferred to
# ``normalizer.normalize_grade`` (task 2.5), so this function does no math on it.

# Standard email address.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# A URL / handle line we should not mistake for a name.
_URL_RE = re.compile(r"(https?://|www\.|linkedin\.com|github\.com)", re.IGNORECASE)

# Candidate phone spans: a run starting with an optional "+", containing digits
# and common separators. Digit-count filtering (below) rejects years/grades.
_PHONE_CANDIDATE_RE = re.compile(r"\+?\d[\d\s\-().]{7,}\d")

# 4-digit years, plausibly a graduation year.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_GRAD_KEYWORDS_RE = re.compile(
    r"grad|batch|expected|passing|class of|20\d{2}\s*[-–]\s*20\d{2}", re.IGNORECASE
)

# Grade capture patterns, tried in priority order. Each returns the raw grade
# substring exactly as found (label included where natural) for the normalizer.
_GRADE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:CGPA|GPA)\s*[:\-]?\s*\d{1,2}(?:\.\d+)?\s*(?:/\s*\d{1,2}(?:\.\d+)?)?", re.IGNORECASE),
    re.compile(r"\d{1,2}(?:\.\d+)?\s*/\s*10(?:\.0+)?\b"),
    re.compile(r"\d{1,2}(?:\.\d+)?\s*/\s*4(?:\.0+)?\b"),
    re.compile(r"\d{1,3}(?:\.\d+)?\s*(?:%|percent(?:age)?)", re.IGNORECASE),
    re.compile(r"(?:percentage|percent)\s*[:\-]?\s*\d{1,3}(?:\.\d+)?", re.IGNORECASE),
]

# Degree keyword table — longer / more specific phrases first so the fullest
# phrase wins. Patterns are matched case-insensitively; the matched text is
# returned in its original casing (never fabricated).
_DEGREE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"bachelor of engineering",
        r"bachelor of technology",
        r"bachelor of computer applications",
        r"bachelor of science",
        r"master of technology",
        r"master of computer applications",
        r"master of science",
        r"master of business administration",
        r"\bb\.?\s?tech\b",
        r"\bm\.?\s?tech\b",
        r"\bb\.?\s?e\.?\b",
        r"\bm\.?\s?e\.?\b",
        r"\bb\.?\s?sc\b",
        r"\bm\.?\s?sc\b",
        r"\bb\.?\s?c\.?a\b",
        r"\bm\.?\s?c\.?a\b",
        r"\bmba\b",
        r"\bb\.?\s?com\b",
        r"\bph\.?\s?d\b",
    ]
]

# Branch / specialization keyword table — specific phrases first.
_BRANCH_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"computer science(?: and engineering| & engineering)?",
        r"information technology",
        r"information science(?: and engineering)?",
        r"electronics and communication(?: engineering)?",
        r"electronics & communication(?: engineering)?",
        r"electronics(?: and)? communication",
        r"artificial intelligence(?: and machine learning)?",
        r"electrical(?: and electronics)?(?: engineering)?",
        r"mechanical(?: engineering)?",
        r"civil(?: engineering)?",
        r"data science",
        r"machine learning",
        r"\bCSE\b",
        r"\bECE\b",
        r"\bEEE\b",
        r"\bIT\b",
    ]
]

# College / institution line signal.
_COLLEGE_RE = re.compile(r"institute|college|university|school of", re.IGNORECASE)

# Section header keywords → used to locate and bound content sections.
_SECTION_HEADERS: dict[str, tuple[str, ...]] = {
    "projects": ("project", "personal project", "academic project", "key project"),
    "experience": ("experience", "work experience", "employment", "internship", "work history"),
    "certifications": ("certification", "certificate", "course", "licenses", "training"),
}
# Any line matching one of these keywords is treated as a section boundary.
_ANY_HEADER_RE = re.compile(
    r"^\s*(projects?|personal projects?|academic projects?|experience|work experience|"
    r"employment|internships?|work history|certification[s]?|certificate[s]?|courses?|"
    r"licenses?|training|education|academics?|skills?|technical skills|summary|objective|"
    r"profile|contact|achievements?|awards?|publications?|interests?|hobbies|languages|"
    r"references?)\s*[:\-]?\s*$",
    re.IGNORECASE,
)

# Date-range / duration signal inside an experience line.
_DURATION_RE = re.compile(
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"|\d{1,2}/\d{4}|\d{4})\s*(?:[-–—to]+)\s*"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}"
    r"|\d{1,2}/\d{4}|\d{4}|present|current|now)",
    re.IGNORECASE,
)

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"


def extract_fields(text: str) -> dict:
    """Extract structured fields from reconstructed resume text.

    Uses regex + heuristics only (fully offline, no LLM). Every field that
    cannot be confidently recovered is returned as ``None`` (or ``[]`` for
    list fields); values are never inferred or fabricated (Req 4.2, 4.3).
    Skills are scanned from the *entire* text — including terms embedded in
    project/experience prose — not just a dedicated skills section (Req 4.4).

    Returns a dict with keys: ``full_name``, ``email``, ``phone``, ``college``,
    ``degree``, ``branch``, ``grad_year``, ``raw_grade``, ``skills`` (list),
    ``projects`` (list of ``{title, description}``), ``experience`` (list of
    ``{company, role, duration}``), ``certifications`` (list).
    """
    text = text or ""
    lines = [ln.strip() for ln in text.splitlines()]

    return {
        "full_name": _extract_name(lines),
        "email": _extract_email(text),
        "phone": _extract_phone(text),
        "college": _extract_college(lines),
        "degree": _extract_first_pattern(text, _DEGREE_PATTERNS),
        "branch": _extract_first_pattern(text, _BRANCH_PATTERNS),
        "grad_year": _extract_grad_year(text),
        "raw_grade": _extract_raw_grade(text),
        "skills": _extract_skills(text),
        "projects": _extract_projects(lines),
        "experience": _extract_experience(lines),
        "certifications": _extract_certifications(lines),
    }


# --- individual field extractors ------------------------------------------


def _extract_email(text: str) -> str | None:
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None


def _extract_phone(text: str) -> str | None:
    """Return the first phone-like span with 10–13 digits, else ``None``."""
    for m in _PHONE_CANDIDATE_RE.finditer(text):
        raw = m.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        # Drop a leading country code when counting the national number length.
        national = digits
        if len(digits) > 10 and digits.startswith(("0", "91", "1", "44")):
            national = digits[-10:]
        if 10 <= len(national) <= 13 and len(digits) <= 13:
            return raw
    return None


def _extract_grad_year(text: str) -> int | None:
    """Pick the latest plausible 4-digit year (<= current year + 6).

    Years appearing on lines mentioning graduation/batch/expected/ranges are
    preferred; otherwise the latest plausible year in the whole text is used.
    """
    max_year = datetime.now().year + 6

    def _plausible(years: list[int]) -> list[int]:
        return [y for y in years if 1950 <= y <= max_year]

    keyword_years: list[int] = []
    all_years: list[int] = []
    for line in text.splitlines():
        found = [int(m.group(0)) for m in _YEAR_RE.finditer(line)]
        all_years.extend(found)
        if found and _GRAD_KEYWORDS_RE.search(line):
            keyword_years.extend(found)

    for pool in (_plausible(keyword_years), _plausible(all_years)):
        if pool:
            return max(pool)
    return None


def _extract_raw_grade(text: str) -> str | None:
    """Capture the raw grade substring (label included) for the normalizer.

    Returns e.g. ``"CGPA: 8.7"``, ``"8.2/10"``, or ``"79%"`` exactly as found.
    No normalization is performed here (that is normalizer.normalize_grade).
    """
    for pattern in _GRADE_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0).strip()
    return None


def _extract_first_pattern(text: str, patterns: list[re.Pattern]) -> str | None:
    """Return the matched text of the first pattern that matches ``text``."""
    for pattern in patterns:
        m = pattern.search(text)
        if m:
            return m.group(0).strip()
    return None


def _extract_college(lines: list[str]) -> str | None:
    """First non-empty line naming an Institute/College/University/School of."""
    for line in lines:
        if line and _COLLEGE_RE.search(line):
            return line
    return None


def _looks_like_name(line: str) -> bool:
    """Heuristic: 1–4 title-case words, no digits/email/url/section header."""
    if not line or len(line) > 50:
        return False
    if _EMAIL_RE.search(line) or _URL_RE.search(line):
        return False
    if any(ch.isdigit() for ch in line):
        return False
    if _ANY_HEADER_RE.match(line):
        return False
    words = line.split()
    if not (1 <= len(words) <= 4):
        return False
    for w in words:
        # Each word must be alphabetic (allowing . ' -) and start uppercase,
        # or be fully uppercase (e.g. "JOHN DOE").
        if not re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", w):
            return False
        if not (w[0].isupper() or w.isupper()):
            return False
    return True


def _extract_name(lines: list[str]) -> str | None:
    """Top-of-document name heuristic: first name-like non-empty line."""
    checked = 0
    for line in lines:
        if not line:
            continue
        checked += 1
        if _looks_like_name(line):
            return line
        if checked >= 12:  # names live at the very top; stop scanning early.
            break
    return None


def _extract_skills(text: str) -> list[str]:
    """Scan the FULL text for known skill variants (Req 4.4).

    Normalizes the text the same way :mod:`engine.synonyms` normalizes skill
    variants, then does a word-boundary search for every known raw variant.
    Each hit contributes its canonical skill name. Returns a sorted, de-duped
    list (empty when nothing is found — never fabricated).
    """
    if not text:
        return []
    norm_text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    norm_text = re.sub(r"\s+", " ", norm_text).strip()
    if not norm_text:
        return []

    found: set[str] = set()
    for variant in _SYNONYMS:
        if not variant:
            continue
        if re.search(r"\b" + re.escape(variant) + r"\b", norm_text):
            found.add(canonical(variant))
    return sorted(found)


def _collect_section(lines: list[str], keywords: tuple[str, ...]) -> list[str]:
    """Return the content lines of the first matching section.

    A section starts at a header line whose text matches one of ``keywords``
    and ends at the next recognized section header (``_ANY_HEADER_RE``) or end
    of document. Empty lines inside the section are dropped.
    """
    content: list[str] = []
    in_section = False
    for line in lines:
        if not line:
            continue
        low = line.lower().rstrip(":-").strip()
        is_header = bool(_ANY_HEADER_RE.match(line))
        if not in_section:
            # Enter the section when the header line contains a target keyword.
            if is_header and any(k in low for k in keywords):
                in_section = True
            continue
        # Inside the section: stop at the next section header.
        if is_header:
            break
        content.append(line)
    return content


def _extract_projects(lines: list[str]) -> list[dict]:
    """Best-effort project items from a Projects section.

    Each content line becomes ``{title, description}``: the part before the
    first ``-``/``:``/``–`` is the title, the remainder the description
    (``None`` when there is no separator). Empty when no section is present.
    """
    items: list[dict] = []
    for line in _collect_section(lines, _SECTION_HEADERS["projects"]):
        # Strip common bullet markers.
        clean = re.sub(r"^[\-•*·▪◦o]\s*", "", line).strip()
        if len(clean) < 2:
            continue
        m = re.split(r"\s*[:–—\-]\s*", clean, maxsplit=1)
        title = m[0].strip()
        description = m[1].strip() if len(m) > 1 and m[1].strip() else None
        if title:
            items.append({"title": title, "description": description})
    return items


def _extract_experience(lines: list[str]) -> list[dict]:
    """Best-effort experience items from an Experience/Internship section.

    For each content line, a duration is pulled via a date-range pattern; the
    remaining text is split into role and company on common separators. Missing
    parts are ``None`` (never fabricated). Empty when no section is present.
    """
    items: list[dict] = []
    for line in _collect_section(lines, _SECTION_HEADERS["experience"]):
        clean = re.sub(r"^[\-•*·▪◦o]\s*", "", line).strip()
        if len(clean) < 2:
            continue

        duration = None
        dm = _DURATION_RE.search(clean)
        if dm:
            duration = dm.group(0).strip()
            clean = (clean[: dm.start()] + " " + clean[dm.end():]).strip(" ,-–—|")

        role = None
        company = None
        if clean:
            parts = [p.strip() for p in re.split(r"\s*(?:[|–—]|,|\bat\b)\s*", clean) if p.strip()]
            if len(parts) >= 2:
                role, company = parts[0], parts[1]
            elif len(parts) == 1:
                role = parts[0]

        if role or company or duration:
            items.append({"company": company, "role": role, "duration": duration})
    return items


def _extract_certifications(lines: list[str]) -> list[str]:
    """List each content line of a Certifications/Courses section."""
    certs: list[str] = []
    for line in _collect_section(lines, _SECTION_HEADERS["certifications"]):
        clean = re.sub(r"^[\-•*·▪◦o]\s*", "", line).strip()
        if len(clean) >= 2:
            certs.append(clean)
    return certs


# ---------------------------------------------------------------------------
# 2.6 — Parse-quality classification and orchestration (TODO)
# ---------------------------------------------------------------------------


def classify_parse_quality(data: ResumeData) -> ParseFlag:
    """Classify parse quality as Clean / Partial / Failed (Req 3.1–3.4).

    The required-field set that drives the flag is: ``full_name``, ``email``,
    ``degree``, ``grad_year``, and at least one skill (Req 3.2). Exactly one
    :class:`~engine.models.ParseFlag` is returned:

    * **Clean** — all required fields are present (Req 3.2).
    * **Partial** — ``full_name`` is present plus at least one additional
      field, but not the full required set (Req 3.3).
    * **Failed** — no usable text, or no required field recovered (e.g. an
      empty ``raw_text`` or only the name was found) (Req 3.4).
    """
    # No usable text at all → Failed (Req 3.4).
    if not (data.raw_text and data.raw_text.strip()):
        return ParseFlag.FAILED

    has_name = bool(data.full_name)
    has_email = bool(data.email)
    has_degree = bool(data.degree)
    has_grad_year = data.grad_year is not None
    has_skill = bool(data.skills)

    # Clean: every required field present (Req 3.2).
    if has_name and has_email and has_degree and has_grad_year and has_skill:
        return ParseFlag.CLEAN

    # Count all recovered required fields to distinguish "only name" from a
    # genuine partial recovery.
    recovered = sum(
        (has_name, has_email, has_degree, has_grad_year, has_skill)
    )

    # Partial: name present AND at least one additional required field (Req 3.3).
    if has_name and recovered >= 2:
        return ParseFlag.PARTIAL

    # No required field recovered, or only the name → Failed (Req 3.4).
    return ParseFlag.FAILED


def _sha256_hex(text: str) -> str:
    """Return the SHA-256 hexdigest of ``text`` (the resume cache key)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _first_page_width(pdf_path: str, default: float = 595.0) -> float:
    """Best-effort width of the first page; ``default`` (A4) on any failure."""
    try:
        fitz = _import_fitz()
        doc = fitz.open(pdf_path)
        try:
            for page in doc:
                return float(page.rect.width)
        finally:
            doc.close()
    except Exception:
        pass
    return default


def parse_resume(pdf_path: str) -> ResumeData:
    """Full Stage-1 pipeline for one file; never raises to the caller.

    Wires extraction → column reconstruction → OCR fallback → field extraction
    → grade normalization → parse-quality classification. Every sub-step is
    self-guarded and the whole body is wrapped in a final try/except so an
    unexpected failure downgrades the resume to ``Failed`` rather than raising
    (Req 2.3, 3.3). Failure is always a first-class, visible outcome — never a
    silent drop (Req 7.4).
    """
    file_name = os.path.basename(pdf_path)

    # Safe default: Failed until proven otherwise (Req 2.2, 3.4).
    data = ResumeData(
        file_name=file_name,
        resume_hash="",
        raw_text="",
        parse_flag=ParseFlag.FAILED,
    )

    try:
        # --- Text extraction + reading-order reconstruction ---------------
        raw_text = ""
        try:
            blocks = extract_text_blocks(pdf_path)
            page_width = _first_page_width(pdf_path)
            raw_text = reconstruct_reading_order(blocks, page_width)
        except Exception:
            blocks = []
            raw_text = ""

        # Fallback to naive extraction if block reconstruction yielded little.
        if word_count(raw_text) < config.OCR_MIN_WORDS:
            try:
                fallback = raw_text_via_pymupdf(pdf_path)
            except Exception:
                fallback = ""
            if word_count(fallback) > word_count(raw_text):
                raw_text = fallback

        # --- OCR fallback when text is too sparse (Req 7.1–7.3) -----------
        if word_count(raw_text) < config.OCR_MIN_WORDS:
            try:
                ocr_text = ocr_page_text(pdf_path)
            except Exception as ocr_exc:  # Req 7.2 — OCR execution error.
                data.used_ocr = True
                data.parse_flag = ParseFlag.FAILED
                data.error_reason = f"OCR execution error: {ocr_exc}"
                data.raw_text = raw_text
                data.resume_hash = _sha256_hex(raw_text)
                return data

            data.used_ocr = True
            data.parse_notes.append("OCR used")
            raw_text = ocr_text

            if word_count(raw_text) < config.OCR_MIN_WORDS:  # Req 7.3.
                data.parse_flag = ParseFlag.FAILED
                data.error_reason = "OCR yielded insufficient text"
                data.raw_text = raw_text
                data.resume_hash = _sha256_hex(raw_text)
                return data

        # --- Finalize raw text + hash -------------------------------------
        data.raw_text = raw_text
        data.resume_hash = _sha256_hex(raw_text)

        # --- Field extraction (Req 4.1) -----------------------------------
        fields = extract_fields(raw_text)
        data.full_name = fields.get("full_name")
        data.email = fields.get("email")
        data.phone = fields.get("phone")
        data.college = fields.get("college")
        data.degree = fields.get("degree")
        data.branch = fields.get("branch")
        data.grad_year = fields.get("grad_year")
        data.raw_grade = fields.get("raw_grade")
        data.skills = list(fields.get("skills") or [])
        data.projects = [
            Project(title=p.get("title", ""), description=p.get("description"))
            for p in (fields.get("projects") or [])
        ]
        data.experience = [
            Experience(
                company=e.get("company"),
                role=e.get("role"),
                duration=e.get("duration"),
            )
            for e in (fields.get("experience") or [])
        ]
        data.certifications = list(fields.get("certifications") or [])

        # --- Grade normalization (Req 5) ----------------------------------
        if data.raw_grade:
            normalized, assumption, confidence = normalizer.normalize_grade(
                data.raw_grade
            )
            data.normalized_grade = normalized
            data.grade_assumption = assumption
            data.grade_confidence = confidence

        # --- Parse-quality classification (Req 3.1–3.4) -------------------
        data.parse_flag = classify_parse_quality(data)
        return data

    except Exception as exc:  # Never raise to the caller (Req 2.3, 3.3).
        return ResumeData(
            file_name=file_name,
            resume_hash=data.resume_hash or _sha256_hex(data.raw_text),
            raw_text=data.raw_text,
            parse_flag=ParseFlag.FAILED,
            error_reason=f"Parse error: {exc}",
        )


# ---------------------------------------------------------------------------
# Manual demonstration (task 2.2): left-then-right reconstruction
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - manual sanity check
    # Synthetic two-column page (width=600, gutter at x=300). The left column
    # holds "Skills"/"Python"; the right column holds "Experience"/"Company".
    # Correct reconstruction emits the ENTIRE left column before the right one,
    # never interleaving by y across the gutter.
    demo_blocks = [
        TextBlock(x0=30, y0=10, x1=180, y1=30, text="Skills"),
        TextBlock(x0=350, y0=12, x1=560, y1=32, text="Experience"),
        TextBlock(x0=30, y0=50, x1=180, y1=70, text="Python"),
        TextBlock(x0=350, y0=55, x1=560, y1=75, text="Acme Corp"),
    ]
    print(reconstruct_reading_order(demo_blocks, page_width=600.0))
    # Expected order (left column first, then right column):
    #   Skills
    #   Python
    #   Experience
    #   Acme Corp
