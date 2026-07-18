"""Engine facade — the single synchronous entry point the CLI and the future
web API both call.

This module is the thin orchestration seam between the layered engine
(``parser`` → ``scorer`` → ``output``) and its callers. It exposes two
functions:

- :func:`run_shortlist` — parse a set of resumes, score them against ONE Job
  Description, and return a single result dict matching the web response
  contract.
- :func:`run_many` — parse each resume EXACTLY ONCE, score every resume against
  EVERY Job Description, optionally write the mandatory artifacts
  (``sample_output`` + ``parse_quality_report``), and return one contract dict
  per JD.

Design references:
- Req 16.1-16.3 : per-candidate score/confidence/parse-flag, three reasoning
  bullets, and a per-JD summary line — sourced from :mod:`engine.output`.
- Req 17.1      : sample output written via :func:`output.write_sample_output`.
- Req 17.2      : parse quality report via :func:`output.write_parse_quality_report`.

Parsing is the expensive step, so :func:`run_many` performs it once and reuses
the parsed :class:`~engine.models.ResumeData` list across all JDs. The CLI
ingests resumes itself (folder guards + per-resume isolation) and passes the
already-parsed list straight through, so a multi-JD run never re-parses a PDF.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

from engine import jd as jd_module
from engine import output, parser, scorer
from engine.jd import JobDescription
from engine.models import ParseFlag, ResumeData, ScoredCandidate


# Accepted resume inputs: a folder path, a list of PDF file paths, or a list of
# already-parsed ResumeData (so the CLI can parse once and reuse).
ResumesInput = Union[str, list[str], list[ResumeData]]


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _parse_one(pdf_path: str) -> ResumeData:
    """Parse a single PDF with the same isolation boundary the CLI uses.

    ``parse_resume`` already self-guards and should never raise, but this outer
    try/except is the contract-level guarantee that one bad file becomes a
    ``Failed`` record rather than aborting the run (Req 2.3).
    """
    file_name = os.path.basename(pdf_path)
    try:
        return parser.parse_resume(pdf_path)
    except Exception as exc:  # Isolation boundary — one bad PDF never aborts (Req 2.3).
        return ResumeData(
            file_name=file_name,
            resume_hash="",
            raw_text="",
            parse_flag=ParseFlag.FAILED,
            error_reason=f"Ingestion error: {exc}",
        )


def _gather_pdfs(folder: str) -> list[str]:
    """Return sorted full paths of the ``.pdf`` files directly inside ``folder``.

    A non-directory yields an empty list so callers can treat "missing" and
    "empty" identically (Req 1.3). Sub-directories are ignored.
    """
    if not os.path.isdir(folder):
        return []
    paths: list[str] = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isfile(full) and name.lower().endswith(".pdf"):
            paths.append(full)
    return paths


def _ensure_parsed(resumes: ResumesInput) -> list[ResumeData]:
    """Normalize the ``resumes`` argument into a parsed ``ResumeData`` list.

    Accepts any of:

    - a folder path (``str``) → gather its PDFs and parse each,
    - a list of PDF file paths (``list[str]``) → parse each,
    - a list of already-parsed ``ResumeData`` → returned as-is (lets the CLI
      parse once and reuse across JDs).

    Parsing is done exactly once here; :func:`run_many` never re-parses.
    """
    if isinstance(resumes, str):
        return [_parse_one(p) for p in _gather_pdfs(resumes)]

    items = list(resumes)
    if items and all(isinstance(item, ResumeData) for item in items):
        return items  # already parsed — reuse directly.

    # Otherwise treat the list as PDF file paths.
    return [_parse_one(str(p)) for p in items]


def _resolve_jd(jd: Union[JobDescription, str]) -> JobDescription:
    """Return a :class:`JobDescription`, resolving a raw JD text string if given.

    A ``JobDescription`` is used directly. A ``str`` is treated as raw JD text
    and routed through :func:`jd.parse_live_jd`; since live parsing is a
    not-yet-enabled bonus (a stub that raises ``NotImplementedError``), a clear
    :class:`ValueError` is raised telling the caller to pass a
    ``JobDescription`` or a ``--jd`` file instead (Req 19.3, 20.2).
    """
    if isinstance(jd, JobDescription):
        return jd
    if isinstance(jd, str):
        try:
            return jd_module.parse_live_jd(jd)
        except NotImplementedError as exc:
            raise ValueError(
                "Live JD text parsing is a bonus feature and is not yet enabled. "
                "Pass a JobDescription object or supply a --jd JSON/YAML file."
            ) from exc
    raise TypeError(
        f"jd must be a JobDescription or a raw JD text str, got {type(jd).__name__}."
    )


# ---------------------------------------------------------------------------
# Scoring + contract building
# ---------------------------------------------------------------------------


def _score_against_jd(
    all_resumes: list[ResumeData], jd: JobDescription
) -> tuple[list[ScoredCandidate], list[ScoredCandidate], list[ScoredCandidate]]:
    """Score every resume against ``jd`` and split into (shortlist, reserve, failed)."""
    scored = [scorer.score_candidate(resume, jd) for resume in all_resumes]
    return output.rank_and_split(scored, jd)


def _candidate_dict(cand: ScoredCandidate, jd: JobDescription) -> dict[str, Any]:
    """Build the per-candidate contract record for a scored candidate.

    Name is the candidate's ``full_name`` when present, otherwise the always-set
    ``file_name`` (never nameless). ``quality`` is the parse-flag value, and the
    three reasoning bullets come from :func:`output.build_reasoning` (Req 16.1,
    16.2).
    """
    resume = cand.resume
    name = (resume.full_name or "").strip() or resume.file_name
    return {
        "name": name,
        "score": cand.score,
        "confidence": cand.confidence,
        "quality": resume.parse_flag.value,
        "reasons": output.build_reasoning(cand, jd),
    }


def _failed_dict(cand: ScoredCandidate) -> dict[str, Any]:
    """Build the per-candidate contract record for a Failed-parse candidate."""
    resume = cand.resume
    name = (resume.full_name or "").strip() or resume.file_name
    return {
        "name": name,
        "reason": resume.error_reason or "required fields could not be extracted",
    }


def _contract_dict(
    jd: JobDescription,
    shortlist: list[ScoredCandidate],
    reserve: list[ScoredCandidate],
    failed: list[ScoredCandidate],
) -> dict[str, Any]:
    """Assemble the web-contract result dict for one JD's ranked results."""
    cutoff = min((c.score for c in shortlist), default=None)
    return {
        "role": jd.title,
        "jd_id": jd.id,
        "evaluated": len(shortlist) + len(reserve) + len(failed),
        "shortlisted": len(shortlist),
        "cutoff": cutoff,
        "failures": len(failed),
        "shortlist": [_candidate_dict(c, jd) for c in shortlist],
        "reserve": [_candidate_dict(c, jd) for c in reserve],
        "failed": [_failed_dict(c) for c in failed],
    }


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


def run_shortlist(
    resumes: ResumesInput,
    jd: Union[JobDescription, str],
    out_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Parse resumes, score them against ONE JD, and return the result dict.

    ``resumes`` may be a folder path, a list of PDF file paths, or a list of
    already-parsed :class:`ResumeData`. ``jd`` may be a :class:`JobDescription`
    or a raw JD text string (the latter requires the not-yet-enabled live-parse
    bonus and otherwise raises a clear :class:`ValueError`).

    Returns a dict matching the web response contract::

        {
            "role": str, "jd_id": str,
            "evaluated": int, "shortlisted": int,
            "cutoff": float | None, "failures": int,
            "shortlist": [{"name","score","confidence","quality","reasons"}, ...],
            "reserve":   [ ... same shape ... ],
            "failed":    [{"name","reason"}, ...],
        }

    When ``out_dir`` is given the mandatory artifacts are written for this single
    JD (``sample_output`` + ``parse_quality_report``, Req 17.1, 17.2).
    """
    resolved_jd = _resolve_jd(jd)
    results = run_many(resumes, [resolved_jd], out_dir=out_dir)
    return results[0]


def run_many(
    resumes: ResumesInput,
    jds: list[JobDescription],
    out_dir: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Parse each resume ONCE, score against every JD, return one dict per JD.

    ``resumes`` is normalized/parsed a single time (see :func:`_ensure_parsed`)
    and the parsed list is reused across all ``jds`` so no PDF is parsed more
    than once.

    When ``out_dir`` is provided this ALSO writes the mandatory deliverable
    artifacts:

    - ``sample_output.md`` / ``.json`` via :func:`output.write_sample_output`
      (Req 17.1), built from the ranked Shortlist/Reserve/Failed per JD plus a
      :func:`output.summary_line`.
    - ``parse_quality_report.md`` / ``.json`` via
      :func:`output.write_parse_quality_report` — every resume listed exactly
      once with its parse flag (Req 17.2, DISQUALIFICATION-CRITICAL).

    Returns a list of web-contract result dicts (the :func:`run_shortlist`
    shape), one per JD in input order, for programmatic use.
    """
    all_resumes = _ensure_parsed(resumes)

    contract_results: list[dict[str, Any]] = []
    artifact_results: list[dict[str, Any]] = []

    for jd in jds:
        shortlist, reserve, failed = _score_against_jd(all_resumes, jd)
        contract_results.append(_contract_dict(jd, shortlist, reserve, failed))
        artifact_results.append(
            {
                "jd": jd,
                "shortlist": shortlist,
                "reserve": reserve,
                "failed": failed,
                "summary": output.summary_line(jd, shortlist, reserve, failed),
            }
        )

    if out_dir is not None:
        output.write_sample_output(artifact_results, out_dir)
        output.write_parse_quality_report(all_resumes, out_dir)

    return contract_results
