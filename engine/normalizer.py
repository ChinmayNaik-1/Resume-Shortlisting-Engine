"""Grade normalization for the Resume Shortlisting Engine (Stage 1, offline).

Converts an academic grade found in a resume (CGPA, percentage, or 4-point GPA)
onto a uniform 10-point scale so candidates using different grading systems can
be compared fairly.

Design references (Req 5 — Grade Normalization):
- Req 5.1 : A 10-point CGPA is recorded unchanged.
- Req 5.2 : A percentage is normalized as ``percentage / 9.5``.
- Req 5.3 : A 4-point GPA is normalized as ``gpa * 2.5``.
- Req 5.4 : When the scale is ambiguous, apply a *stated* assumption, record the
            assumption text, and assign a Low grade confidence.

Correctness Property 4 (Normalized grade always in [0, 10]): whenever a
non-null normalized grade is returned it is clamped to the closed interval
[0, 10].

These are pure functions with no side effects and no network access (Req 8.1),
depending only on the standard library, keeping Stage 1 fully offline.
"""

from __future__ import annotations

import re
from typing import Optional

# A grade token: an integer or decimal number such as ``8``, ``8.4`` or ``82``.
_NUMBER = r"\d+(?:\.\d+)?"

# An explicit "value / scale" fraction such as ``3.6/4``, ``8.4 / 10`` or
# ``3.6/4.0`` (optional surrounding whitespace).
_FRACTION_RE = re.compile(rf"({_NUMBER})\s*/\s*({_NUMBER})")

# First standalone number anywhere in the string.
_NUMBER_RE = re.compile(_NUMBER)

# Confidence levels (kept as plain strings to match the ResumeData contract).
_HIGH = "High"
_LOW = "Low"


def _clamp_10(value: float) -> float:
    """Clamp a normalized grade to the [0, 10] range (Property 4)."""
    return max(0.0, min(10.0, value))


def _round2(value: float) -> float:
    """Round to two decimals for stable, readable output (e.g. 82/9.5 -> 8.63)."""
    return round(value, 2)


def normalize_grade(
    raw_grade: str,
) -> tuple[Optional[float], Optional[str], str]:
    """Normalize a raw grade string onto a 10-point scale.

    Args:
        raw_grade: The grade text exactly as found in the resume, e.g.
            ``"8.7 CGPA"``, ``"82%"``, ``"3.6/4"`` or ``"CGPA - 8.9 / 10"``.

    Returns:
        A 3-tuple ``(normalized_10pt, assumption_note, confidence)`` where:
          * ``normalized_10pt`` is a float in [0, 10], or ``None`` if nothing
            parseable was found.
          * ``assumption_note`` is a human-readable note describing the
            assumption applied for an ambiguous scale, or ``None`` when the
            scale was explicit.
          * ``confidence`` is ``"High"`` for an explicit scale or ``"Low"`` for
            an assumed (ambiguous) scale. ``"Low"`` is also returned when
            nothing parseable is found.
    """
    if not raw_grade or not str(raw_grade).strip():
        return (None, None, _LOW)

    text = str(raw_grade).strip()
    lowered = text.lower()

    has_percent = "%" in text or "percent" in lowered
    has_cgpa = "cgpa" in lowered

    # --- Case: explicit percentage (Req 5.2). Checked first because '%' is the
    # least ambiguous scale marker. Guard against a fraction like "79/100".
    if has_percent and not _FRACTION_RE.search(text):
        num_match = _NUMBER_RE.search(text)
        if num_match:
            value = float(num_match.group(0))
            return (_round2(_clamp_10(value / 9.5)), None, _HIGH)

    # --- Case: explicit fraction "value / scale" (Req 5.1 for /10, Req 5.3 for
    # /4). Handles messy spacing like "CGPA - 8.9 / 10".
    frac_match = _FRACTION_RE.search(text)
    if frac_match:
        value = float(frac_match.group(1))
        denom = float(frac_match.group(2))
        if denom <= 0:
            # Degenerate scale — fall through to magnitude-based handling.
            pass
        elif abs(denom - 10.0) < 0.01:
            return (_round2(_clamp_10(value)), None, _HIGH)          # 10-pt CGPA
        elif abs(denom - 4.0) < 0.01:
            return (_round2(_clamp_10(value * 2.5)), None, _HIGH)     # 4-pt GPA
        else:
            # Any other explicit scale (e.g. /5, /100): scale linearly to 10.
            return (_round2(_clamp_10(value / denom * 10.0)), None, _HIGH)

    # --- Case: explicit 10-point CGPA by keyword, no fraction (Req 5.1).
    # e.g. "CGPA: 8.7", "8.7 CGPA".
    if has_cgpa:
        num_match = _NUMBER_RE.search(text)
        if num_match:
            value = float(num_match.group(0))
            return (_round2(_clamp_10(value)), None, _HIGH)

    # --- Ambiguity handling (Req 5.4): a bare number with no scale context.
    num_match = _NUMBER_RE.search(text)
    if not num_match:
        return (None, None, _LOW)

    value = float(num_match.group(0))

    if value <= 4.0:
        # Could plausibly be a 4-point GPA; assume so and multiply by 2.5.
        note = "Bare value <=4 assumed to be 4-point GPA; multiplied by 2.5"
        return (_round2(_clamp_10(value * 2.5)), note, _LOW)

    if value <= 10.0:
        # Between 4 and 10 with no marker: most likely a 10-point CGPA.
        note = "Bare value assumed to be 10-point CGPA"
        return (_round2(_clamp_10(value)), note, _LOW)

    if value <= 100.0:
        # Between 10 and 100: most likely a percentage; divide by 9.5.
        note = "Bare value assumed to be a percentage; divided by 9.5"
        return (_round2(_clamp_10(value / 9.5)), note, _LOW)

    # value > 100: out of any sane grade range. Treat as a percentage-like
    # figure and clamp, still flagged Low so the output makes the doubt visible.
    note = "Bare value >100 assumed to be a percentage; divided by 9.5 and clamped"
    return (_round2(_clamp_10(value / 9.5)), note, _LOW)


if __name__ == "__main__":  # pragma: no cover - quick smoke test
    # Explicit-scale cases (High confidence, no assumption note).
    assert normalize_grade("8.4") == (8.4, "Bare value assumed to be 10-point CGPA", _LOW)
    assert normalize_grade("8.4/10") == (8.4, None, _HIGH)
    assert normalize_grade("CGPA: 8.7") == (8.7, None, _HIGH)
    assert normalize_grade("8.7 CGPA") == (8.7, None, _HIGH)
    assert normalize_grade("CGPA - 8.9 / 10") == (8.9, None, _HIGH)

    # Percentage (Req 5.2): 82 / 9.5 = 8.63, 79 / 9.5 = 8.32.
    assert normalize_grade("82%") == (8.63, None, _HIGH)
    assert normalize_grade("82 %") == (8.63, None, _HIGH)
    assert normalize_grade("Percentage: 79") == (8.32, None, _HIGH)
    assert normalize_grade("79%") == (8.32, None, _HIGH)

    # 4-point GPA (Req 5.3): 3.6 * 2.5 = 9.0.
    assert normalize_grade("3.6/4") == (9.0, None, _HIGH)
    assert normalize_grade("GPA 3.6/4.0") == (9.0, None, _HIGH)

    # Ambiguous bare values (Req 5.4): Low confidence + stated assumption.
    val, note, conf = normalize_grade("3.6")
    assert (val, conf) == (9.0, _LOW) and note is not None
    val, note, conf = normalize_grade("88")
    assert (val, conf) == (round(88 / 9.5, 2), _LOW) and note is not None

    # Unparseable input.
    assert normalize_grade("") == (None, None, _LOW)
    assert normalize_grade("N/A grade") == (None, None, _LOW)

    # Property 4: every non-null result stays within [0, 10].
    for sample in ["150%", "999", "12/4", "10.5/10", "-", "0"]:
        v, _, _ = normalize_grade(sample)
        assert v is None or 0.0 <= v <= 10.0, sample

    print("normalizer smoke test passed")
