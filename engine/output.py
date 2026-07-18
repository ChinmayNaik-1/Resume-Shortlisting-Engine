"""Output layer for the Resume Shortlisting Engine.

Responsibilities (design.md → output.py): ranking, min-CGPA gate, slot split
(Shortlist/Reserve), reasoning-bullet builder, summary line, sample-output +
parse-quality-report writers, and CSV export.

This module currently implements the **completeness-writing** portion of the
thin vertical slice (task 3.2): every ingested resume is written to the output
exactly once, with Failed resumes carrying no score plus a human-review
recommendation (Req 2.1, 2.2). Before writing, the record count is asserted
equal to the ingested resume count so a silent drop can never happen (Req 2.4).

Ranking, reasoning, sample output, parse-quality report, and CSV export land in
later tasks (10.x, 11.x, 13.4) and are stubbed below so imports do not break.

Stdlib only (json, os) plus engine.models.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from engine.models import MatchType, ParseFlag, ResumeData, ScoredCandidate


# ---------------------------------------------------------------------------
# Completeness (task 3.2) — Req 2.1, 2.2, 2.4
# ---------------------------------------------------------------------------

HUMAN_REVIEW_RECOMMENDATION = (
    "Parse failed — recommend manual human review; no score assigned."
)


def assert_completeness(input_count: int, output_count: int) -> None:
    """Guarantee no resume is silently dropped from the output (Req 2.4).

    Raises a clear ``AssertionError`` when the number of records about to be
    written does not equal the number of resumes ingested. Used by ``cli`` as a
    hard gate before any output is written.
    """
    assert input_count == output_count, (
        "Output completeness violated (Req 2.4): ingested "
        f"{input_count} resume(s) but produced {output_count} output record(s). "
        "Every resume must appear in the output exactly once."
    )


def _completeness_record(resume: ResumeData) -> dict[str, Any]:
    """Build a single completeness record for one resume.

    Every resume contributes exactly one record keyed by ``file_name`` (the
    stable identity, Req 2.1). Failed resumes carry no score, ``human_review``,
    an ``error_reason`` (when known), and a human-review recommendation
    (Req 2.2).
    """
    flag = resume.parse_flag
    flag_value = flag.value if isinstance(flag, ParseFlag) else str(flag)
    is_failed = flag == ParseFlag.FAILED

    record: dict[str, Any] = {
        "file_name": resume.file_name,
        "parse_flag": flag_value,
        "score": None,
        "human_review": is_failed,
    }

    if is_failed:
        record["error_reason"] = resume.error_reason or "Unknown parse failure"
        record["recommendation"] = HUMAN_REVIEW_RECOMMENDATION

    return record


def _render_markdown(records: list[dict[str, Any]]) -> str:
    """Render the completeness records as a human-readable markdown listing."""
    lines: list[str] = [
        "# Parse Listing",
        "",
        f"Total resumes: {len(records)}",
        "",
    ]
    for record in records:
        lines.append(f"## {record['file_name']}")
        lines.append(f"- Parse flag: {record['parse_flag']}")
        if record.get("human_review"):
            lines.append(f"- Error reason: {record.get('error_reason')}")
            lines.append(f"- Recommendation: {record.get('recommendation')}")
        lines.append("")
    return "\n".join(lines)


def write_completeness_listing(resumes: list[ResumeData], out_dir: str) -> str:
    """Write one record per resume to ``{out_dir}/parse_listing.md`` (+ .json).

    Ensures every resume appears exactly once. For each resume a record is
    emitted with ``file_name`` and ``parse_flag``; Failed resumes additionally
    carry ``error_reason`` and a human-review recommendation (Req 2.1, 2.2).

    The record count is asserted equal to the resume count before writing
    (Req 2.4). Creates ``out_dir`` when missing. Returns the markdown path.
    """
    os.makedirs(out_dir, exist_ok=True)

    records = [_completeness_record(resume) for resume in resumes]

    # Hard completeness gate before any bytes are written (Req 2.4).
    assert_completeness(len(resumes), len(records))

    md_path = os.path.join(out_dir, "parse_listing.md")
    json_path = os.path.join(out_dir, "parse_listing.json")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(records))

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    return md_path


# ---------------------------------------------------------------------------
# Stubs — implemented in later tasks (leave signatures so imports don't break)
# ---------------------------------------------------------------------------


def _below_cgpa_threshold(cand: ScoredCandidate, jd: Any) -> bool:
    """True when the candidate's grade is known and below the JD's min CGPA.

    A ``None`` normalized grade never excludes on CGPA — absence of a grade is
    not proof of failing the gate, so the candidate stays eligible for the
    Shortlist (Req 14.4). When a JD sets no threshold (``min_cgpa`` <= 0), no
    candidate is below it.
    """
    if jd.min_cgpa is None or jd.min_cgpa <= 0:
        return False
    grade = cand.resume.normalized_grade
    return grade is not None and grade < jd.min_cgpa


def rank_and_split(
    cands: list[ScoredCandidate], jd: Any
) -> tuple[list[ScoredCandidate], list[ScoredCandidate], list[ScoredCandidate]]:
    """Return ``(shortlist, reserve, failed)`` from scored candidates (task 10.1).

    Behavior (Req 14.1-14.4):

    - Candidates with ``score is None`` (Failed parse) are separated into
      ``failed`` and never ranked; they still appear in the output elsewhere.
    - Scored candidates are sorted descending by ``score`` with a deterministic
      stable tie-break by ``resume.resume_hash`` ascending (Bonus D).
    - The JD ``min_cgpa`` gate marks a scored candidate as below-threshold when
      its ``normalized_grade`` is known and below the threshold. A ``None`` grade
      is never excluded (can't prove it fails, Req 14.4).
    - ``shortlist`` = the top ``jd.slots`` candidates that PASS the CGPA gate
      (Req 14.1). ``reserve`` = every remaining scored candidate — those beyond
      the slot count AND those below threshold — with scores retained (Req 14.2).
    - No scored candidate is ever dropped from both lists (Req 14.3): every
      scored candidate lands in exactly one of ``shortlist`` or ``reserve``.
    """
    scored = [c for c in cands if c.score is not None]
    failed = [c for c in cands if c.score is None]

    # Descending by score, deterministic tie-break by resume_hash ascending.
    scored.sort(key=lambda c: (-c.score, c.resume.resume_hash))

    slots = jd.slots if jd.slots and jd.slots > 0 else 0
    shortlist: list[ScoredCandidate] = []
    reserve: list[ScoredCandidate] = []

    for cand in scored:
        if not _below_cgpa_threshold(cand, jd) and len(shortlist) < slots:
            shortlist.append(cand)
        else:
            # Below threshold OR beyond the slot count → reserve (score kept).
            reserve.append(cand)

    return shortlist, reserve, failed


def _skill_names(matches: list, limit: int = 3) -> str:
    """Join up to ``limit`` JD skill names from ``matches`` for a bullet."""
    return ", ".join(m.jd_skill for m in matches[:limit])


def build_reasoning(cand: ScoredCandidate, jd: Any) -> list[str]:
    """Return EXACTLY three reasoning bullets for a candidate (task 10.2, Req 16.2).

    For a Failed candidate (``score is None``) the three bullets explain the
    parse failure and recommend human review.

    For a scored candidate the bullets are built from real signals — matched
    required skills (exact/synonym), missing required skills, the most important
    conflict note (surfaced when present), notable partial/implicit matches,
    project/experience relevance, and grade vs the JD minimum CGPA. The
    highest-signal bullets are selected in priority order; the list is always
    normalized to exactly three (padded with a score/confidence summary when
    fewer, truncated when more).
    """
    # --- Failed parse: explain the failure + recommend review ---
    if cand.score is None:
        reason = cand.resume.error_reason or "required fields could not be extracted"
        return [
            f"Parse failed: {reason}.",
            "No score assigned — the resume could not be reliably parsed for scoring.",
            "Recommend manual human review before making any decision.",
        ]

    matches = cand.skill_matches
    required = [m for m in matches if m.required]
    matched_req = [m for m in required if m.match_type != MatchType.MISSING]
    exact_syn = [
        m for m in matched_req
        if m.match_type in (MatchType.EXACT, MatchType.SYNONYM)
    ]
    missing_req = [m for m in required if m.match_type == MatchType.MISSING]
    partial_impl = [
        m for m in matches
        if m.match_type in (MatchType.PARTIAL, MatchType.IMPLICIT)
    ]

    bullets: list[str] = []

    # 1. Matched required skills.
    if required:
        highlight = _skill_names(exact_syn or matched_req)
        if highlight:
            bullets.append(
                f"Matches {len(matched_req)}/{len(required)} required skills "
                f"including {highlight}."
            )
        else:
            bullets.append(
                f"Matches 0/{len(required)} required skills directly."
            )

    # 2. Missing required skills.
    if missing_req:
        bullets.append(
            f"Missing {len(missing_req)} required skill(s): "
            f"{_skill_names(missing_req)}."
        )

    # 3. Most important conflict note (surfaced when present).
    if cand.conflict_notes:
        bullets.append(cand.conflict_notes[0])

    # 4. Notable partial/implicit match.
    if partial_impl:
        pm = partial_impl[0]
        evidence = f" (via {pm.evidence})" if pm.evidence else ""
        bullets.append(
            f"{pm.match_type.value.capitalize()} match on "
            f"{pm.jd_skill}{evidence}."
        )

    # 5. Project / experience relevance.
    n_projects = len(cand.resume.projects)
    n_experience = len(cand.resume.experience)
    if n_projects or n_experience:
        bullets.append(
            f"Portfolio: {n_projects} project(s) and {n_experience} "
            f"experience entr{'y' if n_experience == 1 else 'ies'} on file."
        )

    # 6. Grade vs the JD minimum CGPA.
    grade = cand.resume.normalized_grade
    if grade is None:
        bullets.append(
            "Grade not found on resume — could not verify against the CGPA gate."
        )
    elif jd.min_cgpa and jd.min_cgpa > 0:
        relation = "meets" if grade >= jd.min_cgpa else "is below"
        bullets.append(
            f"Normalized grade {grade:.1f}/10 {relation} the "
            f"{jd.min_cgpa:.1f} minimum CGPA."
        )
    else:
        bullets.append(f"Normalized grade {grade:.1f}/10 (no CGPA gate set).")

    # --- Normalize to EXACTLY three bullets (Req 16.2) ---
    pad = (
        f"Overall score {cand.score:.1f}/100 with "
        f"{cand.confidence or 'n/a'} confidence ({cand.tier_used.value} tier)."
    )
    while len(bullets) < 3:
        bullets.append(pad)
    return bullets[:3]


def summary_line(
    jd: Any,
    shortlist: list[ScoredCandidate],
    reserve: list[ScoredCandidate],
    failed: list[ScoredCandidate],
) -> str:
    """Return a one-line per-JD summary (task 10.2, Req 16.3).

    States the number of candidates evaluated (all scored + all failed), the
    number shortlisted, the score cutoff (the lowest score in the Shortlist, or
    ``n/a`` when the Shortlist is empty), and the number of parse failures.
    """
    evaluated = len(shortlist) + len(reserve) + len(failed)
    cutoff = f"{min(c.score for c in shortlist):.1f}" if shortlist else "n/a"
    title = getattr(jd, "title", None) or getattr(jd, "id", "?")
    return (
        f"JD '{title}': {evaluated} candidate(s) evaluated, "
        f"{len(shortlist)} shortlisted, score cutoff {cutoff}, "
        f"{len(failed)} parse failure(s)."
    )


def _candidate_name(cand: ScoredCandidate) -> str:
    """Return the display name for a candidate.

    Uses ``resume.full_name`` when present, otherwise falls back to the always-set
    ``resume.file_name`` so a candidate is never nameless in the output.
    """
    resume = cand.resume
    name = getattr(resume, "full_name", None)
    if name and str(name).strip():
        return str(name).strip()
    return resume.file_name


# ---------------------------------------------------------------------------
# Sample output (task 11.1) — Req 17.1, 16.1-16.3
# ---------------------------------------------------------------------------


def _flag_value(flag: Any) -> str:
    """Render a ``ParseFlag`` (or raw value) as its string name."""
    return flag.value if isinstance(flag, ParseFlag) else str(flag)


def _candidate_md(cand: ScoredCandidate, rank: int, jd: Any) -> list[str]:
    """Render one scored candidate as markdown lines (rank, score, reasoning)."""
    name = _candidate_name(cand)
    score = f"{cand.score:.1f}/100" if cand.score is not None else "n/a"
    confidence = cand.confidence or "n/a"
    flag = _flag_value(cand.resume.parse_flag)

    lines = [
        f"**{rank}. {name}** — {score} · confidence: {confidence} · "
        f"parse: {flag}",
    ]
    for bullet in build_reasoning(cand, jd):
        lines.append(f"   - {bullet}")
    lines.append("")
    return lines


def _failed_md(cand: ScoredCandidate) -> list[str]:
    """Render one failed-parse candidate as markdown lines."""
    name = _candidate_name(cand)
    reason = cand.resume.error_reason or "required fields could not be extracted"
    return [
        f"**{name}** ({cand.resume.file_name})",
        f"   - Reason: {reason}",
        f"   - {HUMAN_REVIEW_RECOMMENDATION}",
        "",
    ]


def _candidate_json(cand: ScoredCandidate, jd: Any) -> dict[str, Any]:
    """Build the machine-readable record for one scored candidate (Req 17.1)."""
    return {
        "name": _candidate_name(cand),
        "file_name": cand.resume.file_name,
        "score": cand.score,
        "confidence": cand.confidence,
        "quality": _flag_value(cand.resume.parse_flag),
        "reasons": build_reasoning(cand, jd),
    }


def _jd_result_json(result: dict[str, Any]) -> dict[str, Any]:
    """Build the per-JD JSON block matching the response contract (Req 17.1)."""
    jd = result["jd"]
    shortlist = result.get("shortlist", [])
    reserve = result.get("reserve", [])
    failed = result.get("failed", [])

    evaluated = len(shortlist) + len(reserve) + len(failed)
    cutoff = min((c.score for c in shortlist), default=None)

    return {
        "role": getattr(jd, "id", None),
        "title": getattr(jd, "title", None),
        "evaluated": evaluated,
        "shortlisted": len(shortlist),
        "cutoff": cutoff,
        "failures": len(failed),
        "shortlist": [_candidate_json(c, jd) for c in shortlist],
        "reserve": [_candidate_json(c, jd) for c in reserve],
        "failed": [
            {
                "name": _candidate_name(c),
                "file_name": c.resume.file_name,
                "reason": c.resume.error_reason
                or "required fields could not be extracted",
            }
            for c in failed
        ],
    }


def _jd_result_md(result: dict[str, Any]) -> list[str]:
    """Render one per-JD result block as markdown lines (Req 16.1-16.3, 17.1)."""
    jd = result["jd"]
    shortlist = result.get("shortlist", [])
    reserve = result.get("reserve", [])
    failed = result.get("failed", [])
    summary = result.get("summary") or summary_line(jd, shortlist, reserve, failed)

    title = getattr(jd, "title", None) or getattr(jd, "id", "?")
    lines: list[str] = [
        f"## {title}",
        "",
        f"_{summary}_",
        "",
        "### Shortlist",
        "",
    ]
    if shortlist:
        for rank, cand in enumerate(shortlist, start=1):
            lines.extend(_candidate_md(cand, rank, jd))
    else:
        lines.extend(["_No candidates shortlisted._", ""])

    lines.extend(["### Reserve", ""])
    if reserve:
        for rank, cand in enumerate(reserve, start=1):
            lines.extend(_candidate_md(cand, rank, jd))
    else:
        lines.extend(["_No reserve candidates._", ""])

    lines.extend(["### Failed Parse", ""])
    if failed:
        for cand in failed:
            lines.extend(_failed_md(cand))
    else:
        lines.extend(["_No parse failures._", ""])

    return lines


def write_sample_output(results: list[dict[str, Any]], out_dir: str) -> None:
    """Write the full multi-JD sample output (task 11.1, Req 17.1, 16.1-16.3).

    ``results`` is a list of per-JD dicts, each shaped like::

        {
            "jd": JobDescription,
            "shortlist": list[ScoredCandidate],   # ranked (rank_and_split)
            "reserve": list[ScoredCandidate],     # ranked (rank_and_split)
            "failed": list[ScoredCandidate],      # parse failures
            "summary": str,                       # summary_line (optional)
        }

    One call writes BOTH ``{out_dir}/sample_output.md`` (human-readable, with a
    per-JD heading, summary line, and Shortlist / Reserve / Failed Parse
    sections — each scored candidate showing rank, name, score/100, confidence,
    parse flag, and the three reasoning bullets from :func:`build_reasoning`) and
    ``{out_dir}/sample_output.json`` (machine-readable, matching the response
    contract per JD). Creates ``out_dir`` when missing.
    """
    os.makedirs(out_dir, exist_ok=True)

    md_lines: list[str] = ["# Sample Output", ""]
    for result in results:
        md_lines.extend(_jd_result_md(result))

    json_payload = [_jd_result_json(result) for result in results]

    md_path = os.path.join(out_dir, "sample_output.md")
    json_path = os.path.join(out_dir, "sample_output.json")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Parse quality report (task 11.2 — DISQUALIFICATION-CRITICAL) — Req 17.2
# ---------------------------------------------------------------------------

# Required-field set that drives the parse flag (design.md → ResumeData). Used
# to report exactly which required fields are missing on a Partial resume.
_REQUIRED_FIELDS: list[tuple[str, str]] = [
    ("name", "full_name"),
    ("email", "email"),
    ("degree", "degree"),
    ("grad_year", "grad_year"),
    ("skills", "skills"),
]


def _missing_required_fields(resume: ResumeData) -> list[str]:
    """Return the human-readable names of required fields absent on a resume.

    ``skills`` counts as present when the list holds ≥1 entry; every other field
    counts as present when it is not ``None`` (and, for strings, non-blank).
    """
    missing: list[str] = []
    for label, attr in _REQUIRED_FIELDS:
        value = getattr(resume, attr, None)
        if attr == "skills":
            if not value:
                missing.append(label)
        elif value is None or (isinstance(value, str) and not value.strip()):
            missing.append(label)
    return missing


def _quality_record(resume: ResumeData) -> dict[str, Any]:
    """Build one parse-quality record for a resume (Req 17.2)."""
    flag = resume.parse_flag
    flag_value = _flag_value(flag)

    record: dict[str, Any] = {
        "file_name": resume.file_name,
        "parse_flag": flag_value,
        "used_ocr": bool(resume.used_ocr),
    }

    if flag == ParseFlag.FAILED:
        record["status"] = "Failed"
        record["error_reason"] = (
            resume.error_reason or "required fields could not be extracted"
        )
    elif flag == ParseFlag.PARTIAL:
        record["status"] = "Partial"
        record["missing_required_fields"] = _missing_required_fields(resume)
    else:
        record["status"] = "Clean"

    return record


def _quality_markdown(records: list[dict[str, Any]]) -> str:
    """Render the parse-quality records as a clear, complete markdown report."""
    total = len(records)
    clean = sum(1 for r in records if r["parse_flag"] == ParseFlag.CLEAN.value)
    partial = sum(1 for r in records if r["parse_flag"] == ParseFlag.PARTIAL.value)
    failed = sum(1 for r in records if r["parse_flag"] == ParseFlag.FAILED.value)
    ocr_used = sum(1 for r in records if r.get("used_ocr"))

    lines: list[str] = [
        "# Parse Quality Report",
        "",
        f"Total resumes: {total}",
        f"- Clean: {clean}",
        f"- Partial: {partial}",
        f"- Failed: {failed}",
        f"- Used OCR: {ocr_used}",
        "",
        "Every resume below is listed exactly once with its parse flag and status.",
        "",
    ]

    for record in records:
        lines.append(f"## {record['file_name']}")
        lines.append(f"- Parse flag: {record['parse_flag']}")
        lines.append(f"- OCR used: {'yes' if record.get('used_ocr') else 'no'}")
        if record["status"] == "Failed":
            lines.append(f"- Status: Failed — {record['error_reason']}")
        elif record["status"] == "Partial":
            missing = record.get("missing_required_fields") or []
            missing_str = ", ".join(missing) if missing else "none identified"
            lines.append(
                f"- Status: Partial — missing required field(s): {missing_str}"
            )
        else:
            lines.append("- Status: Clean — all required fields recovered")
        lines.append("")

    return "\n".join(lines)


def write_parse_quality_report(
    all_resumes: list[ResumeData], out_dir: str
) -> None:
    """Write the separate parse-quality report (task 11.2, Req 17.2).

    DISQUALIFICATION-CRITICAL: a missing parse quality report is an instant DQ.
    Writes ``{out_dir}/parse_quality_report.md`` and ``.json`` listing EVERY
    resume exactly once with its ``file_name``, parse flag (Clean/Partial/
    Failed), whether OCR was used, and — for Partial/Failed — what failed or is
    missing (``error_reason`` for Failed; the missing required fields for
    Partial). A summary header reports the total resume count, the Clean/Partial/
    Failed counts, and how many resumes used OCR. Creates ``out_dir`` when
    missing.
    """
    os.makedirs(out_dir, exist_ok=True)

    records = [_quality_record(resume) for resume in all_resumes]

    md_path = os.path.join(out_dir, "parse_quality_report.md")
    json_path = os.path.join(out_dir, "parse_quality_report.json")

    summary = {
        "total": len(records),
        "clean": sum(1 for r in records if r["parse_flag"] == ParseFlag.CLEAN.value),
        "partial": sum(
            1 for r in records if r["parse_flag"] == ParseFlag.PARTIAL.value
        ),
        "failed": sum(
            1 for r in records if r["parse_flag"] == ParseFlag.FAILED.value
        ),
        "used_ocr": sum(1 for r in records if r.get("used_ocr")),
    }

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_quality_markdown(records))

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "resumes": records}, fh, indent=2)


def export_csv(cands: list[ScoredCandidate], path: str) -> None:
    """Export the shortlist as CSV (bonus).

    TODO(task 13.4): CSV export of the shortlist for the Streamlit UI and CLI
    (Req 20.1).
    """
    raise NotImplementedError("export_csv is implemented in task 13.4")
