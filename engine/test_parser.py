"""Unit tests for engine.parser task 2.6 — parse-quality classification and
the ``parse_resume`` orchestration contract (Req 3.1–3.4, 2.3, 7.2/7.3).

These tests cover the pure, dependency-free logic (``classify_parse_quality``)
exhaustively, plus the never-raise / always-Failed-on-error contract of
``parse_resume`` (which degrades gracefully even when the PDF/OCR stack is
unavailable, per Req 2.3 and 7.4).
"""

from engine.models import ParseFlag, ResumeData
from engine.parser import classify_parse_quality, parse_resume


def _resume(**overrides) -> ResumeData:
    """Build a ResumeData with sensible defaults, overriding named fields."""
    base = dict(
        file_name="cand.pdf",
        resume_hash="hash",
        raw_text="some resume text here",
    )
    base.update(overrides)
    return ResumeData(**base)


# --- classify_parse_quality: Clean (Req 3.2) -------------------------------


def test_clean_when_all_required_fields_present():
    data = _resume(
        full_name="Jane Doe",
        email="jane@example.com",
        degree="B.Tech",
        grad_year=2025,
        skills=["python"],
    )
    assert classify_parse_quality(data) == ParseFlag.CLEAN


def test_partial_when_one_required_field_missing():
    # Missing email — name + several other required fields present.
    data = _resume(
        full_name="Jane Doe",
        degree="B.Tech",
        grad_year=2025,
        skills=["python"],
    )
    assert classify_parse_quality(data) == ParseFlag.PARTIAL


# --- classify_parse_quality: Partial (Req 3.3) -----------------------------


def test_partial_name_plus_one_additional_field():
    data = _resume(full_name="Jane Doe", email="jane@example.com")
    assert classify_parse_quality(data) == ParseFlag.PARTIAL


def test_partial_name_plus_one_skill():
    data = _resume(full_name="Jane Doe", skills=["python"])
    assert classify_parse_quality(data) == ParseFlag.PARTIAL


# --- classify_parse_quality: Failed (Req 3.4) ------------------------------


def test_failed_when_raw_text_empty():
    data = _resume(
        raw_text="",
        full_name="Jane Doe",
        email="jane@example.com",
        degree="B.Tech",
        grad_year=2025,
        skills=["python"],
    )
    assert classify_parse_quality(data) == ParseFlag.FAILED


def test_failed_when_raw_text_only_whitespace():
    data = _resume(raw_text="   \n\t ", full_name="Jane Doe", email="j@x.com")
    assert classify_parse_quality(data) == ParseFlag.FAILED


def test_failed_when_only_name_recovered():
    # Name present but no other required field → not a usable partial (Req 3.4).
    data = _resume(full_name="Jane Doe")
    assert classify_parse_quality(data) == ParseFlag.FAILED


def test_failed_when_no_required_field_recovered():
    data = _resume()  # no name, email, degree, grad_year, or skills
    assert classify_parse_quality(data) == ParseFlag.FAILED


def test_failed_when_fields_present_but_no_name():
    # Additional fields but no name → cannot be Partial (name is mandatory).
    data = _resume(email="jane@example.com", degree="B.Tech", grad_year=2025)
    assert classify_parse_quality(data) == ParseFlag.FAILED


def test_returns_exactly_one_flag_member():
    data = _resume(full_name="Jane Doe", email="jane@example.com")
    assert classify_parse_quality(data) in (
        ParseFlag.CLEAN,
        ParseFlag.PARTIAL,
        ParseFlag.FAILED,
    )


# --- parse_resume: never-raise contract (Req 2.3, 7.2/7.3, 7.4) ------------


def test_parse_resume_never_raises_on_missing_file():
    # A nonexistent path must degrade to a visible Failed record, not raise.
    result = parse_resume("does_not_exist_12345.pdf")
    assert isinstance(result, ResumeData)
    assert result.parse_flag == ParseFlag.FAILED
    assert result.error_reason is not None


def test_parse_resume_sets_file_name_from_basename():
    result = parse_resume("/some/nested/dir/candidate_42.pdf")
    assert result.file_name == "candidate_42.pdf"
    assert result.parse_flag == ParseFlag.FAILED
