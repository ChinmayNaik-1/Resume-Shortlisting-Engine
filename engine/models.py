"""Shared data models for the Resume Shortlisting Engine.

Models are plain ``@dataclass`` objects (stdlib, no ORM). All optional fields
default to ``None``; absent = ``None``, never fabricated (Req 4.2, 4.3).

``JobDescription`` intentionally lives in ``jd.py`` (scoring layer), not here.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ParseFlag(str, Enum):
    CLEAN = "Clean"
    PARTIAL = "Partial"
    FAILED = "Failed"


class MatchType(str, Enum):
    EXACT = "exact"
    SYNONYM = "synonym"
    PARTIAL = "partial"
    IMPLICIT = "implicit"
    MISSING = "missing"


class Tier(str, Enum):
    RULES = "rules"
    CLOUD = "cloud"
    LOCAL = "local"


# ---------------------------------------------------------------------------
# Parsing-layer dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class Project:
    title: str
    description: Optional[str] = None      # one-line


@dataclass
class Experience:
    company: Optional[str] = None
    role: Optional[str] = None
    duration: Optional[str] = None


@dataclass
class ResumeData:
    file_name: str                          # always set (identity in output)
    resume_hash: str                        # sha256 of raw extracted text; cache key
    raw_text: str                           # reconstructed reading-order text

    # Identity / contact
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

    # Academic
    college: Optional[str] = None
    degree: Optional[str] = None
    branch: Optional[str] = None
    grad_year: Optional[int] = None
    raw_grade: Optional[str] = None         # as found, e.g. "8.7 CGPA", "82%"
    normalized_grade: Optional[float] = None  # 10-pt scale
    grade_assumption: Optional[str] = None  # stated assumption when ambiguous
    grade_confidence: str = "High"          # High | Low (Low when ambiguous)

    # Content
    skills: list[str] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)
    experience: list[Experience] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)

    # Provenance
    parse_flag: ParseFlag = ParseFlag.FAILED
    used_ocr: bool = False
    parse_notes: list[str] = field(default_factory=list)  # e.g. "OCR used", "2-column"
    error_reason: Optional[str] = None      # set only when Failed via exception


# ---------------------------------------------------------------------------
# Scoring-layer dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SkillMatch:
    jd_skill: str
    match_type: MatchType
    evidence: Optional[str] = None          # candidate skill or project snippet
    required: bool = True                   # required vs preferred
    credit: float = 0.0                     # 0..1 fraction of this skill's weight


@dataclass
class ScoredCandidate:
    resume: ResumeData
    jd_id: str
    score: Optional[float]                  # 0..100; None when parse Failed
    confidence: Optional[str]               # High | Medium | Low; None when Failed
    tier_used: Tier                         # highest tier that confirmed
    skill_matches: list[SkillMatch] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)  # exactly 3 bullets when scored
    human_review: bool = False              # True when Failed parse
    conflict_notes: list[str] = field(default_factory=list)
