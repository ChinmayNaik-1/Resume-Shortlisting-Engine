"""Central configuration for the Resume Shortlisting Engine.

This module is the single source of truth for every tunable constant and for
loading environment-based credentials/settings from a local ``.env`` file.

Design references:
- Req 8.2  : Stage 1 must run without an API key. Loading keys here (and never
             requiring them at import time) keeps the parser offline-capable.
- Req 10.8 : Uniform 3-second per-tier LLM timeout.
- Req 15.3 : Documented Job Description defaults for missing optional fields.
- Req 18.3 : Deterministic LLM calls (temperature 0) — the scoring layer reads
             these constants rather than hardcoding numbers.

No other module should hardcode these numbers; import them from here instead.
"""

from __future__ import annotations

import os

try:
    # python-dotenv is optional at runtime: if it is missing we simply rely on
    # any variables already present in the process environment. This keeps the
    # module importable in a minimal/offline setup (Req 8.2).
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - defensive fallback
    pass


# ---------------------------------------------------------------------------
# Environment / credentials (.env)
# ---------------------------------------------------------------------------
# API keys are optional; Stage 1 never needs them and the cascade degrades to
# the offline rules tier when they are absent (Req 8.2, 10.4).
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Model identifiers per tier (overridable via .env).
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


# ---------------------------------------------------------------------------
# Timeouts and triggers
# ---------------------------------------------------------------------------
# Uniform hard per-call timeout applied identically to Groq, Gemini, and
# Ollama; a timeout is treated exactly like a call failure (Req 10.8, 18.1).
LLM_TIMEOUT_S: int = 3

# When extracted text falls below this word count, the parser triggers OCR
# because the PDF is almost certainly image-only or near-empty (Req 7.1).
OCR_MIN_WORDS: int = 50

# DPI used when rendering PDF pages to images for OCR (Tricky Part §2).
OCR_DPI: int = 200

# Path to the Tesseract OCR binary. pytesseract only wraps the native binary;
# on Windows it is not on PATH by default, so we point at the standard install
# location. Override via the TESSERACT_CMD env var on other machines (or leave
# the binary on PATH, in which case an empty/absent value lets pytesseract find
# it automatically). The parser only applies this when the path actually exists,
# so a Linux/Mac host with tesseract on PATH is unaffected.
TESSERACT_CMD: str = os.getenv(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ---------------------------------------------------------------------------
# Scoring component weights (sum = 100)
# ---------------------------------------------------------------------------
WEIGHT_REQUIRED: int = 55    # Required-skills match — primary signal.
WEIGHT_PREFERRED: int = 15   # Preferred-skills match — differentiates ties.
WEIGHT_EXPERIENCE: int = 15  # Experience / project relevance.
WEIGHT_ACADEMIC: int = 10    # Normalized grade — deliberately minor.
WEIGHT_EDUCATION: int = 5    # Degree / branch relevance — small nudge.


# ---------------------------------------------------------------------------
# Match-credit table: fraction of a skill's weight awarded per match type.
# ---------------------------------------------------------------------------
CREDIT: dict[str, float] = {
    "exact": 1.0,
    "synonym": 0.9,
    "partial": 0.5,
    "implicit": 0.7,
    "missing": 0.0,
}


# ---------------------------------------------------------------------------
# Job Description defaults (applied when optional fields are absent, Req 15.3)
# ---------------------------------------------------------------------------
DEFAULT_SLOTS: int = 5        # Documented default Slot_Count.
DEFAULT_MIN_CGPA: float = 0.0  # No academic gate by default.


# ---------------------------------------------------------------------------
# Grade normalization anchor
# ---------------------------------------------------------------------------
# A normalized CGPA of 9.0/10 maps to full academic marks in the scoring
# formula (academic sub-score = 10 * clamp(grade / 9.0, 0, 1)).
ACADEMIC_FULL_MARKS_CGPA: float = 9.0


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
# Disk cache directory for LLM results, keyed by resume_hash + jd_id. Deleting
# this folder forces fresh calls; a warm cache makes reruns free and
# deterministic (Req 18.2).
CACHE_DIR: str = ".cache/llm"
