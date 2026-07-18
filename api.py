"""FastAPI web backend for the Resume Shortlisting Engine (Bonus A).

This module is a THIN wrapper around the existing engine facade
(:func:`engine.pipeline.run_shortlist`). It does NOT reimplement any scoring,
parsing, matching, or ranking logic — every real decision is delegated to the
already-built pipeline. The web layer only:

- accepts uploaded PDFs (or a folder path) + a JD selection (predefined id or
  pasted free text),
- builds a :class:`~engine.jd.JobDescription` (loading a JSON file, or doing a
  BASIC defensive inline extraction from pasted text),
- calls ``pipeline.run_shortlist(...)`` and returns its result dict as JSON,
- serves a single self-contained HTML page.

FastAPI / uvicorn are imported lazily so that ``import api`` succeeds even when
those packages are not installed in the environment (Req 20.1). If they are
missing, a clear ``pip install`` hint is printed and the app object is ``None``.
"""

import os
import re
import shutil
import tempfile
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
JDS_DIR = os.path.join(WORKSPACE_ROOT, "jds")
WEB_DIR = os.path.join(WORKSPACE_ROOT, "web")
INDEX_HTML = os.path.join(WEB_DIR, "index.html")

_INSTALL_HINT = (
    "The web UI requires extra packages that are not installed.\n"
    "Install them with:\n\n"
    "    pip install fastapi uvicorn python-multipart\n"
)


# ---------------------------------------------------------------------------
# JD resolution helpers (no scoring logic — only building a JobDescription)
# ---------------------------------------------------------------------------


def _list_jd_files() -> list[str]:
    """Return sorted full paths of ``.json`` files in the ``jds/`` folder."""
    if not os.path.isdir(JDS_DIR):
        return []
    return [
        os.path.join(JDS_DIR, name)
        for name in sorted(os.listdir(JDS_DIR))
        if name.lower().endswith(".json") and os.path.isfile(os.path.join(JDS_DIR, name))
    ]


def list_predefined_jds() -> list[dict[str, str]]:
    """Return ``[{"id","title","stem"}, ...]`` for every predefined JD file.

    ``id`` is the JD's declared id field (e.g. ``frontend-developer``), ``stem``
    is the file name without extension (e.g. ``frontend``). Both are accepted as
    lookup keys by :func:`resolve_predefined_jd`. Files that cannot be read are
    skipped defensively so one bad file never breaks the dropdown.
    """
    from engine import jd as jd_module

    out: list[dict[str, str]] = []
    for path in _list_jd_files():
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            jd = jd_module.load_jd(path)
            out.append({"id": jd.id, "title": jd.title, "stem": stem})
        except Exception:
            # Skip unreadable/invalid JD files rather than failing the endpoint.
            continue
    return out


def resolve_predefined_jd(jd_id: str):
    """Resolve a predefined JD by id or file stem into a ``JobDescription``.

    Resolution order (Req 15.1, 19.3):
    1. Direct file match ``jds/{jd_id}.json`` (accepts the stem like ``frontend``).
    2. Otherwise scan every ``jds/*.json`` for a matching ``id`` field, also
       accepting a stem match.

    Raises ``ValueError`` (mapped to HTTP 400 by the caller) when nothing matches.
    """
    from engine import jd as jd_module

    if not jd_id:
        raise ValueError("No jd_id provided.")

    key = jd_id.strip()

    # 1) Direct file match on stem (e.g. "frontend" -> jds/frontend.json).
    direct = os.path.join(JDS_DIR, f"{key}.json")
    if os.path.isfile(direct):
        return jd_module.load_jd(direct)

    # 2) Scan for a matching id field or stem.
    for path in _list_jd_files():
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            jd = jd_module.load_jd(path)
        except Exception:
            continue
        if key == jd.id or key == stem:
            return jd

    available = ", ".join(
        f"{j['id']} ({j['stem']})" for j in list_predefined_jds()
    ) or "(none found)"
    raise ValueError(
        f"Unknown jd_id {jd_id!r}. Available predefined JDs: {available}."
    )


# CGPA patterns for the BASIC pasted-JD extraction. Kept intentionally simple
# and defensive — this is NOT the real live-JD parser (that is bonus 13.2).
_CGPA_PATTERNS = [
    re.compile(r"(?:cgpa|gpa)\s*(?:of|:|>=|=)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:\+)?\s*cgpa", re.I),
    re.compile(r"(?:minimum|min|at least)\s*(?:cgpa|gpa)?\s*(?:of)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
]


def _extract_min_cgpa(text: str) -> float:
    """Best-effort extraction of a minimum CGPA from pasted text.

    Returns the first plausible value on the 10-point scale (0..10); otherwise
    the documented default. Defensive: never raises.
    """
    from engine import config

    for pattern in _CGPA_PATTERNS:
        for match in pattern.finditer(text or ""):
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                continue
            if 0.0 <= value <= 10.0:
                return value
    return config.DEFAULT_MIN_CGPA


def _extract_skills_from_text(text: str) -> list[str]:
    """Scan pasted text for known canonical skills from the engine vocabulary.

    Uses :data:`engine.synonyms.SKILL_VOCABULARY` and :func:`synonyms.canonical`
    so the pasted JD lines up with the same skill keys the scorer uses. Returns
    a de-duplicated list preserving first-seen order. This is a BASIC scan only.
    """
    from engine import synonyms

    lowered = (text or "").lower()
    found: list[str] = []
    seen: set[str] = set()

    # Scan the raw synonym surface forms so multi-word variants ("rest api",
    # "next js") are detected, then map each hit to its canonical key.
    # We iterate the canonical vocabulary and also probe common surface tokens.
    # Simplest robust approach: tokenize the text and canonicalize n-grams.
    words = re.findall(r"[a-z0-9\.\+/#]+", lowered)
    # Build candidate 1- and 2-grams to catch things like "rest api".
    candidates: list[str] = []
    candidates.extend(words)
    candidates.extend(
        f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)
    )

    vocab = synonyms.SKILL_VOCABULARY
    for cand in candidates:
        canon = synonyms.canonical(cand)
        if canon in vocab and canon not in seen:
            seen.add(canon)
            found.append(canon)
    return found


def build_pasted_jd(jd_text: str):
    """Build a ``JobDescription`` from raw pasted text using a BASIC scan.

    This is deliberately simple and defensive (the full live-JD parser is bonus
    13.2). It:
    - collects ``required_skills`` as canonical skills found in the text,
    - regexes a minimum CGPA if stated (else the documented default),
    - uses the default Slot_Count,
    - titles the JD from the first non-empty line (else "Pasted JD").
    """
    from engine import config
    from engine.jd import JobDescription

    text = (jd_text or "").strip()
    if not text:
        raise ValueError("Pasted JD text is empty.")

    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    title = first_line[:120] if first_line else "Pasted JD"

    required = _extract_skills_from_text(text)
    min_cgpa = _extract_min_cgpa(text)

    return JobDescription(
        id="pasted-jd",
        title=title,
        required_skills=required,
        preferred_skills=[],
        slots=config.DEFAULT_SLOTS,
        min_cgpa=min_cgpa,
    )


# ---------------------------------------------------------------------------
# Core run logic (shared by the endpoint; keeps FastAPI-specific code thin)
# ---------------------------------------------------------------------------


def _run_shortlist_core(
    saved_paths: list[str],
    resume_folder_path: Optional[str],
    jd_id: Optional[str],
    jd_text: Optional[str],
) -> dict[str, Any]:
    """Validate inputs, resolve the JD, and call the engine facade.

    Returns the pipeline result dict. Raises ``ValueError`` for bad input
    (mapped to HTTP 400) and lets other exceptions propagate (HTTP 500).
    """
    from engine import pipeline

    # --- Validation: need resumes (uploaded or folder) AND a JD source. ---
    has_uploads = bool(saved_paths)
    folder = (resume_folder_path or "").strip()
    if not has_uploads and not folder:
        raise ValueError(
            "No resumes provided. Upload one or more PDF files or supply "
            "'resume_folder_path'."
        )
    if not (jd_id and jd_id.strip()) and not (jd_text and jd_text.strip()):
        raise ValueError(
            "No job description provided. Supply 'jd_id' (a predefined role) or "
            "'jd_text' (pasted JD)."
        )

    # --- Resolve resume input: uploaded files take precedence over folder. ---
    if has_uploads:
        resumes: Any = saved_paths
    else:
        if not os.path.isdir(folder):
            raise ValueError(f"resume_folder_path {folder!r} is not a directory.")
        resumes = folder

    # --- Resolve JD: predefined id wins if both are present. ---
    if jd_id and jd_id.strip():
        jd = resolve_predefined_jd(jd_id)
    else:
        jd = build_pasted_jd(jd_text or "")

    # --- Delegate to the engine facade (NO scoring logic here). ---
    return pipeline.run_shortlist(resumes, jd)


# ---------------------------------------------------------------------------
# FastAPI app factory (lazy import so `import api` never hard-crashes)
# ---------------------------------------------------------------------------


def create_app():
    """Create and return the FastAPI app, or ``None`` if deps are missing.

    FastAPI, its helpers, and Starlette pieces are imported here (not at module
    top) so that ``import api`` succeeds in an environment without them (Req
    20.1). On ImportError a clear pip hint is printed and ``None`` is returned.
    """
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, JSONResponse
    except ImportError:
        print(_INSTALL_HINT)
        return None

    app = FastAPI(title="Resume Shortlisting Engine — Web UI", version="1.0")

    # Permissive CORS so a file:// page or any localhost origin can call us.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def index():
        """Serve the single-page HTML frontend."""
        if not os.path.isfile(INDEX_HTML):
            raise HTTPException(
                status_code=404,
                detail=f"Frontend not found at {INDEX_HTML}.",
            )
        return FileResponse(INDEX_HTML, media_type="text/html")

    @app.get("/jds")
    def get_jds():
        """Return the list of predefined JD ids + titles for the dropdown."""
        return JSONResponse(list_predefined_jds())

    @app.post("/run")
    async def run(
        resumes: Optional[list[UploadFile]] = File(default=None),
        resume_folder_path: Optional[str] = Form(default=None),
        jd_id: Optional[str] = Form(default=None),
        jd_text: Optional[str] = Form(default=None),
    ):
        """Run the shortlist for uploaded/folder resumes against one JD.

        Saves uploads to a fresh tempdir, delegates to the engine facade, and
        always cleans up the tempdir. Returns the pipeline result dict as JSON.
        """
        tmp_dir: Optional[str] = None
        saved_paths: list[str] = []
        try:
            # Persist uploaded PDFs to a tempdir so the engine can read them.
            if resumes:
                tmp_dir = tempfile.mkdtemp(prefix="resume_upload_")
                for idx, upload in enumerate(resumes):
                    if upload is None:
                        continue
                    fname = os.path.basename(upload.filename or f"resume_{idx}.pdf")
                    if not fname.lower().endswith(".pdf"):
                        # Keep it defensive but permissive: skip non-PDFs.
                        continue
                    dest = os.path.join(tmp_dir, f"{idx:03d}_{fname}")
                    with open(dest, "wb") as fh:
                        shutil.copyfileobj(upload.file, fh)
                    saved_paths.append(dest)

            try:
                result = _run_shortlist_core(
                    saved_paths, resume_folder_path, jd_id, jd_text
                )
            except ValueError as verr:
                # Bad input -> 400 with a clear JSON error.
                raise HTTPException(status_code=400, detail=str(verr))

            return JSONResponse(result)

        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as a 500 with message
            raise HTTPException(status_code=500, detail=f"Run failed: {exc}")
        finally:
            if tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return app


# Build the app at import time (safe: returns None if deps are missing).
app = create_app()


def main() -> None:
    """Run the dev server on port 8000 (lazy-imports uvicorn)."""
    if app is None:
        # create_app already printed the install hint.
        return
    try:
        import uvicorn
    except ImportError:
        print(_INSTALL_HINT)
        return
    print("Serving Resume Shortlisting Engine web UI at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
