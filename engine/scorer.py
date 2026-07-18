"""Scoring orchestration for the Resume Shortlisting Engine (scoring layer).

This module owns Stage 2 scoring. It is built incrementally across several
tasks; this file currently implements the **offline rules baseline** (the
guaranteed floor score, task 6.1) and leaves clearly-marked stubs for the
pieces that arrive in later tasks so imports never break:

- :func:`rules_baseline`      — task 6.1: offline exact/synonym floor.
- :func:`weighted_score`      — task 6.2: full weighted component formula.
- :func:`score_candidate`     — task 6.2 (interim rules-only) → 7.3 / 9.x: orchestrates.
- :func:`resolve_confidence`  — task 9.1: parse x tier confidence tiering (stub).
- :func:`apply_conflict_rules`— task 9.2: four documented conflict rules (stub).

Design references:
- Req 9.1  : Rules baseline uses vocabulary + synonym matching vs the JD.
- Req 9.2  : Baseline is produced with no network dependency (pure/offline).
- Req 11.1 : A candidate skill equal to a JD skill is an ``exact`` match.
- Req 11.2 : A candidate skill mapped through the synonym dictionary is a
             ``synonym`` match.
- Req 11.5 : Required skills are weighted higher than preferred skills — here
             satisfied by ``WEIGHT_REQUIRED`` (55) > ``WEIGHT_PREFERRED`` (15).

The rules baseline is the *floor*: it uses only exact/synonym evidence (no
implicit/partial, which come from the LLM tier later), so the LLM can only ever
raise a candidate's score above this guaranteed offline baseline.
"""

from __future__ import annotations

import re
from typing import Optional

from . import config
from . import extractor_llm
from . import synonyms
from .models import MatchType, ParseFlag, ScoredCandidate, ResumeData, SkillMatch, Tier
from .jd import JobDescription


# ---------------------------------------------------------------------------
# Confidence + conflict-resolution thresholds (design.md Confidence-Tier
# Decision Table and Signal Conflict Resolution table). Kept as named module
# constants so the numbers are documented and tunable in one place.
# ---------------------------------------------------------------------------

# Ordinal ranking of the three (and only three) confidence levels. Higher value
# = stronger confidence (Req 12.3).
_CONFIDENCE_ORDER: dict[str, int] = {"Low": 1, "Medium": 2, "High": 3}

# Signal-conflict thresholds (design.md Signal Conflict Resolution table).
_HIGH_CGPA: float = 8.5          # "High CGPA" — case 1.
_LOW_CGPA: float = 6.0           # "Below-average CGPA" — case 2.
_MANY_RELEVANT_PROJECTS: int = 2  # "many relevant projects" — case 2.
_STRONG_SKILL_FRACTION: float = 0.70  # "strong skill match" — case 3.
_HIGH_SCORE: float = 70.0        # "high score" — case 4 (partial parse).

# Alternatives inside a single JD skill string are separated by " or ",
# "/", or ",". A JD skill like "React.js or Next.js" or "Node.js / Python"
# or "SQL or NoSQL DB" is satisfied if ANY listed alternative matches a
# candidate skill. We split on the whole-word "or", slashes, and commas.
_ALT_SPLIT_RE = re.compile(r"\s+or\s+|/|,", flags=re.IGNORECASE)


def _split_alternatives(jd_skill: str) -> list[str]:
    """Split a JD skill string into its acceptable alternatives.

    ``"React.js or Next.js"`` -> ``["React.js", "Next.js"]``;
    ``"Node.js / Python / Java"`` -> ``["Node.js", "Python", "Java"]``;
    ``"SQL or NoSQL DB"`` -> ``["SQL", "NoSQL DB"]``. Empty fragments are
    dropped. A skill with no separator yields a single-element list.
    """
    parts = [p.strip() for p in _ALT_SPLIT_RE.split(jd_skill or "")]
    return [p for p in parts if p]


def _match_alternatives(
    alternatives: list[str],
    resume_by_canonical: dict[str, str],
) -> tuple[MatchType, str | None]:
    """Find the best match for a JD skill's alternatives among resume skills.

    ``resume_by_canonical`` maps each candidate skill's canonical form to its
    original surface string (for evidence). Returns the match type and the
    matching candidate skill (evidence), or ``(MISSING, None)`` if no
    alternative matches.

    An alternative is an ``exact`` match when its normalized surface form is
    identical to the candidate skill's normalized surface form; it is a
    ``synonym`` match when the two only agree after routing through the synonym
    dictionary (i.e. canonical forms equal but surface forms differ) (Req 11.1,
    11.2). Exact matches are preferred over synonym matches.
    """
    best: tuple[MatchType, str | None] | None = None
    for alt in alternatives:
        canon = synonyms.canonical(alt)
        if not canon:
            continue
        evidence = resume_by_canonical.get(canon)
        if evidence is None:
            continue
        # Distinguish a direct (exact) hit from a synonym-dictionary hit by
        # comparing normalized surface forms.
        if synonyms._normalize(alt) == synonyms._normalize(evidence):
            return MatchType.EXACT, evidence
        # Remember the synonym hit but keep scanning for a possible exact one.
        if best is None:
            best = (MatchType.SYNONYM, evidence)
    return best if best is not None else (MatchType.MISSING, None)


def _score_skill_group(
    jd_skills: list[str],
    resume_by_canonical: dict[str, str],
    required: bool,
) -> tuple[list[SkillMatch], float]:
    """Build SkillMatches for one JD skill group and return (matches, credit_sum).

    ``credit_sum`` is the sum of per-skill credit fractions (each in ``[0, 1]``)
    used by the caller to compute the weighted sub-score.
    """
    matches: list[SkillMatch] = []
    credit_sum = 0.0
    for jd_skill in jd_skills:
        alternatives = _split_alternatives(jd_skill)
        match_type, evidence = _match_alternatives(alternatives, resume_by_canonical)
        credit = config.CREDIT[match_type.value]
        credit_sum += credit
        matches.append(
            SkillMatch(
                jd_skill=jd_skill,
                match_type=match_type,
                evidence=evidence,
                required=required,
                credit=credit,
            )
        )
    return matches, credit_sum


def rules_baseline(
    resume: ResumeData, jd: JobDescription
) -> tuple[float, list[SkillMatch]]:
    """Compute the offline rules-tier floor score and skill matches (task 6.1).

    Matches the candidate's extracted skills against the JD's required and
    preferred skills using exact/synonym matching via :mod:`engine.synonyms`
    (Req 9.1). No network is used (Req 9.2). Each JD skill may list alternatives
    (e.g. ``"React.js or Next.js"``) and is satisfied if ANY alternative
    matches a candidate skill.

    Returns ``(floor_score, skill_matches)`` where ``floor_score`` is the
    required sub-score plus the preferred sub-score:

    - required sub-score = ``WEIGHT_REQUIRED * (Σ required credit) / n_required``
    - preferred sub-score = ``WEIGHT_PREFERRED * (Σ preferred credit) / n_preferred``

    Required skills are weighted above preferred skills by the 55-vs-15 weights
    (Req 11.5). Division-by-zero is guarded: an empty skill group contributes 0.
    This is the guaranteed offline floor (exact/synonym only — implicit/partial
    credit is added later by the LLM tier), so the LLM can only raise a score.
    """
    # Canonicalize the resume's extracted skills. Map canonical -> original
    # surface form so matches can cite the candidate's own wording as evidence.
    resume_by_canonical: dict[str, str] = {}
    for skill in resume.skills:
        canon = synonyms.canonical(skill)
        if canon and canon not in resume_by_canonical:
            resume_by_canonical[canon] = skill

    required_matches, required_credit = _score_skill_group(
        jd.required_skills, resume_by_canonical, required=True
    )
    preferred_matches, preferred_credit = _score_skill_group(
        jd.preferred_skills, resume_by_canonical, required=False
    )

    n_required = len(jd.required_skills)
    n_preferred = len(jd.preferred_skills)

    # Guard divide-by-zero: an empty group contributes a 0 sub-score.
    required_sub = (
        config.WEIGHT_REQUIRED * (required_credit / n_required) if n_required else 0.0
    )
    preferred_sub = (
        config.WEIGHT_PREFERRED * (preferred_credit / n_preferred)
        if n_preferred
        else 0.0
    )

    floor_score = required_sub + preferred_sub
    return floor_score, required_matches + preferred_matches


# ---------------------------------------------------------------------------
# Weighted scoring formula (task 6.2)
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    return max(low, min(high, value))


def _jd_canonical_skills(jd: JobDescription) -> set[str]:
    """Return the set of canonical forms for every JD skill alternative.

    Each JD skill may list alternatives (``"React.js or Next.js"``); every
    alternative is split out and canonicalized so downstream relevance checks
    can recognize any acceptable skill.
    """
    canon: set[str] = set()
    for jd_skill in list(jd.required_skills) + list(jd.preferred_skills):
        for alt in _split_alternatives(jd_skill):
            c = synonyms.canonical(alt)
            if c:
                canon.add(c)
    return canon


def _text_evidences_jd_skill(text: str | None, jd_canonical: set[str]) -> bool:
    """Return True if any canonical JD skill appears in ``text``.

    The free-form text (a project title+description or an experience
    role+company) is normalized and scanned as unigrams and bigrams, each
    routed through :func:`synonyms.canonical`, so surface variants like
    ``"Node.js"`` or ``"REST APIs"`` still resolve to the canonical JD skill.
    """
    if not text or not jd_canonical:
        return False
    norm = synonyms._normalize(text)
    if not norm:
        return False
    words = norm.split(" ")
    # Unigrams and bigrams cover single- and two-word canonical skills
    # (e.g. "react", "rest api").
    grams = list(words)
    grams += [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
    for gram in grams:
        if synonyms.canonical(gram) in jd_canonical:
            return True
    return False


# Degree/branch keywords that count as engineering/CS/IT for education
# relevance (Req 11.3). Matched as normalized substrings against the candidate's
# degree and branch fields.
_ENGINEERING_CS_IT_KEYWORDS: tuple[str, ...] = (
    "computer",
    "cs",
    "cse",
    "information technology",
    "it",
    "software",
    "engineering",
    "b tech",
    "btech",
    "b e",
    "m tech",
    "mtech",
    "electronics",
    "electrical",
    "information science",
    "data science",
)


def _is_engineering_cs_it_degree(resume: ResumeData) -> bool:
    """True if the candidate's degree/branch reads as engineering/CS/IT.

    Used by both education relevance and the "non-CS background" conflict rule
    (Req 11.3, 13.3). Matches normalized keyword substrings against the degree
    and branch fields.
    """
    haystack = synonyms._normalize(f"{resume.degree or ''} {resume.branch or ''}")
    if not haystack:
        return False
    return any(kw in haystack for kw in _ENGINEERING_CS_IT_KEYWORDS)


def _education_relevance_score(resume: ResumeData) -> float:
    """Education-relevance sub-score (0..WEIGHT_EDUCATION), per design.md.

    Full marks when the degree/branch is in the engineering/CS/IT set; a
    partial 2 when a degree is present but not clearly relevant (non-CS is not
    penalized hard); 0 when no degree is known.
    """
    degree = resume.degree
    branch = resume.branch
    if not degree and not branch:
        return 0.0
    if _is_engineering_cs_it_degree(resume):
        return float(config.WEIGHT_EDUCATION)
    # Degree present but not clearly relevant → partial credit, not zero.
    return 2.0 if degree else 0.0


def weighted_score(
    resume: ResumeData, jd: JobDescription, matches: list[SkillMatch]
) -> float:
    """Compute the full 0..100 weighted score from all five components (task 6.2).

    Components (weights sum to 100, all in :mod:`engine.config`):

    - required   = ``WEIGHT_REQUIRED  * (Σ credit over required) / max(1, n_req)``
    - preferred  = ``WEIGHT_PREFERRED * (Σ credit over preferred) / max(1, n_pref)`` (0 if none)
    - experience = ``WEIGHT_EXPERIENCE * min(1, relevant_items / 3)`` where a
      project/experience is relevant if its text evidences ≥1 JD skill; capped
      at 3 items to avoid quantity gaming.
    - academic   = ``WEIGHT_ACADEMIC * clamp(normalized_grade / ACADEMIC_FULL_MARKS_CGPA, 0, 1)`` (0 if no grade).
    - education  = degree/branch relevance (see :func:`_education_relevance_score`).

    Returns ``raw_score`` clamped to ``[0, 100]`` (Req 11.3, 11.5, 9.1).
    """
    # --- Skill sub-scores from the provided matches ---
    req_credit = sum(m.credit for m in matches if m.required)
    pref_credit = sum(m.credit for m in matches if not m.required)
    n_required = sum(1 for m in matches if m.required)
    n_preferred = sum(1 for m in matches if not m.required)

    req_sub = config.WEIGHT_REQUIRED * (req_credit / max(1, n_required))
    pref_sub = (
        config.WEIGHT_PREFERRED * (pref_credit / max(1, n_preferred))
        if n_preferred
        else 0.0
    )

    # --- Experience / projects relevance (capped at 3 relevant items) ---
    jd_canonical = _jd_canonical_skills(jd)
    relevant_items = 0
    for project in resume.projects:
        text = f"{project.title or ''} {project.description or ''}"
        if _text_evidences_jd_skill(text, jd_canonical):
            relevant_items += 1
    for exp in resume.experience:
        text = f"{exp.role or ''} {exp.company or ''}"
        if _text_evidences_jd_skill(text, jd_canonical):
            relevant_items += 1
    exp_sub = config.WEIGHT_EXPERIENCE * min(1.0, relevant_items / 3.0)

    # --- Academic (normalized grade) ---
    if resume.normalized_grade is None:
        acad_sub = 0.0
    else:
        acad_sub = config.WEIGHT_ACADEMIC * _clamp(
            resume.normalized_grade / config.ACADEMIC_FULL_MARKS_CGPA, 0.0, 1.0
        )

    # --- Education relevance ---
    edu_sub = _education_relevance_score(resume)

    raw_score = req_sub + pref_sub + exp_sub + acad_sub + edu_sub
    return _clamp(raw_score, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Confidence tiering (task 9.1)
# ---------------------------------------------------------------------------


def _min_confidence(a: str, b: str) -> str:
    """Return the lower of two confidence levels by ``_CONFIDENCE_ORDER``."""
    return a if _CONFIDENCE_ORDER[a] <= _CONFIDENCE_ORDER[b] else b


def _drop_one_level(level: str) -> str:
    """Drop a confidence level by one step, flooring at ``Low``."""
    return {"High": "Medium", "Medium": "Low", "Low": "Low"}[level]


def resolve_confidence(
    parse_flag: ParseFlag, tier: Tier, has_implicit_or_partial: bool
) -> Optional[str]:
    """Resolve the final three-level confidence (task 9.1).

    The result is the **lower of two independent ceilings** (design.md
    Confidence-Tier Decision Table; Req 10.9, 12.4):

    - **Parse-quality ceiling:** ``Clean`` → High, ``Partial`` → Medium,
      ``Failed`` → ``None`` (no score, human review) (Req 12.1, 12.2).
    - **Tier ceiling:** ``Cloud`` → High, ``Local`` → Medium, ``Rules`` → Medium
      (Req 10.5, 10.6, 9.3).

    Confidence is strictly three-level — High/Medium/Low, never "Medium-High"
    (Req 12.3). After taking the minimum of the two ceilings, if the deciding
    required-skill evidence is implicit or partial the level is dropped one step
    (min Low) and (Req 11.3, 11.4).
    """
    # Parse ceiling. A Failed parse yields no numeric score / no confidence.
    if parse_flag == ParseFlag.FAILED:
        return None
    parse_ceiling = "High" if parse_flag == ParseFlag.CLEAN else "Medium"

    # Tier ceiling: only the Cloud tier can reach High; Local and Rules cap at
    # Medium.
    tier_ceiling = "High" if tier == Tier.CLOUD else "Medium"

    confidence = _min_confidence(parse_ceiling, tier_ceiling)

    # Implicit/partial deciding evidence is a weaker signal → drop one level.
    if has_implicit_or_partial:
        confidence = _drop_one_level(confidence)

    return confidence


# ---------------------------------------------------------------------------
# Signal conflict resolution (task 9.2)
# ---------------------------------------------------------------------------


def _count_relevant_projects(resume: ResumeData, jd: JobDescription) -> int:
    """Count resume projects whose text evidences ≥1 JD skill.

    Used by the "low CGPA offset by projects" conflict rule (Req 13.2).
    """
    jd_canonical = _jd_canonical_skills(jd)
    count = 0
    for project in resume.projects:
        text = f"{project.title or ''} {project.description or ''}"
        if _text_evidences_jd_skill(text, jd_canonical):
            count += 1
    return count


def apply_conflict_rules(
    resume: ResumeData,
    jd: JobDescription,
    matches: list[SkillMatch],
    score: float,
) -> list[str]:
    """Apply the four documented signal-conflict rules (task 9.2, Req 13.1-13.4).

    Returns a list of reasoning notes for whichever of the four documented cases
    fire (design.md Signal Conflict Resolution table). The list may be empty.

    The cases:

    1. **High CGPA, zero projects** (Req 13.1): ``normalized_grade >= 8.5`` and
       no projects → strong academics but unproven applied skill.
    2. **Low CGPA, many relevant projects** (Req 13.2): ``normalized_grade`` is
       known and ``< 6.0`` and ≥2 relevant projects → portfolio offsets grade.
    3. **Strong skills, non-CS degree** (Req 13.3): ≥70% of required skills
       matched (non-missing) and the degree is not engineering/CS/IT.
    4. **Partial parse, high score** (Req 13.4): ``Partial`` parse and
       ``score >= 70`` → keep the score but verify extracted skills.

    .. note:: **Confidence coupling for case 4.** This function only produces
       reasoning notes; it does not mutate confidence. To keep the interface
       simple (a plain ``list[str]`` return), :func:`score_candidate`
       independently re-checks the identical partial-parse + high-score
       condition and, when it holds, forces confidence down to ``Medium`` and
       sets ``human_review=True`` (design.md: "force confidence to Medium and
       flag for review"). Re-checking a single boolean is cheaper and clearer
       than threading a tuple/flag back through the caller.
    """
    notes: list[str] = []

    # --- Case 1: high CGPA but zero projects (Req 13.1) ---
    if (
        resume.normalized_grade is not None
        and resume.normalized_grade >= _HIGH_CGPA
        and len(resume.projects) == 0
    ):
        notes.append(
            "Strong academics but no projects — applied skill unproven."
        )

    # --- Case 2: below-average CGPA offset by many relevant projects (Req 13.2) ---
    if (
        resume.normalized_grade is not None
        and resume.normalized_grade < _LOW_CGPA
        and _count_relevant_projects(resume, jd) >= _MANY_RELEVANT_PROJECTS
    ):
        notes.append(
            "Below-average CGPA offset by strong, relevant project portfolio."
        )

    # --- Case 3: strong skill match despite a non-CS/engineering degree (Req 13.3) ---
    required = [m for m in matches if m.required]
    total_required = len(required)
    matched_required = sum(1 for m in required if m.match_type != MatchType.MISSING)
    strong_skills = (
        total_required > 0
        and (matched_required / total_required) >= _STRONG_SKILL_FRACTION
    )
    if strong_skills and not _is_engineering_cs_it_degree(resume):
        notes.append(
            "Non-CS background but skills directly match role requirements."
        )

    # --- Case 4: high score on a partial parse (Req 13.4) ---
    if resume.parse_flag == ParseFlag.PARTIAL and score >= _HIGH_SCORE:
        notes.append(
            "High score on partial parse — verify extracted skills before deciding."
        )

    return notes


# ---------------------------------------------------------------------------
# LLM match merge (task 7.3)
# ---------------------------------------------------------------------------


def _merge_matches(
    rules_matches: list[SkillMatch], llm_matches: list[SkillMatch]
) -> list[SkillMatch]:
    """Merge the offline rules matches with the LLM's matches (task 7.3).

    Prefer the LLM's matches (they carry the partial/implicit evidence the rules
    tier cannot produce), but **never let a required JD skill drop below the
    credit the rules tier already awarded it**: per JD skill we keep whichever of
    the two matches has the higher credit (Req 10.4, 11.1-11.4). The winning
    match's ``required`` flag is taken from the authoritative rules match when
    one exists (the JD, not the LLM, defines required vs preferred).

    Rules matches establish the ordering (required group first, then preferred);
    any JD skill the LLM matched that the rules tier did not enumerate is
    appended afterwards.
    """
    by_skill: dict[str, SkillMatch] = {}
    order: list[str] = []

    for rm in rules_matches:
        by_skill[rm.jd_skill] = rm
        order.append(rm.jd_skill)

    for lm in llm_matches:
        rm = by_skill.get(lm.jd_skill)
        if rm is None:
            by_skill[lm.jd_skill] = lm
            order.append(lm.jd_skill)
            continue
        # Take the higher-credit match; the rules match wins ties so the
        # guaranteed offline floor is never undercut.
        if lm.credit > rm.credit:
            by_skill[lm.jd_skill] = SkillMatch(
                jd_skill=rm.jd_skill,
                match_type=lm.match_type,
                evidence=lm.evidence,
                required=rm.required,  # authoritative from the JD/rules tier
                credit=lm.credit,
            )
        # else: keep the rules match unchanged.

    return [by_skill[skill] for skill in order]


# ---------------------------------------------------------------------------
# Orchestrator — full three-tier cascade (task 7.3, wiring 9.1 + 9.2).
# ---------------------------------------------------------------------------


def score_candidate(resume: ResumeData, jd: JobDescription) -> ScoredCandidate:
    """Score one resume/JD pair through the full scoring cascade (task 7.3).

    Steps:

    1. A ``Failed`` parse short-circuits to a no-score, human-review candidate
       that still appears in the output (Req 2.2, 12.2).
    2. Compute the offline rules baseline — the guaranteed floor score plus its
       exact/synonym matches (Req 9.1).
    3. Attempt :func:`extractor_llm.enhance`. On an :class:`LLMResult`, merge the
       LLM's matches over the rules matches (never dropping a required skill
       below its rules credit), recompute the weighted score, and adopt the
       LLM's source tier. If ``enhance`` returns ``None``, stay on the rules tier
       and note the LLM was unavailable (Req 10.4).
    4. Resolve confidence from the parse-quality × tier ceilings (Req 10.9, 12.4).
    5. Apply the four conflict rules; if the partial-parse + high-score case
       fires, force confidence to ``Medium`` and flag for human review (Req 13.4).
    6. ``final_score = clamp(max(rules_floor, weighted), 0, 100)`` — the LLM can
       only raise a score above the offline floor (Req 10.4).
    """
    # --- 1. Failed parse: no score, flag for human review, still emitted ---
    if resume.parse_flag == ParseFlag.FAILED:
        cand = ScoredCandidate(
            resume=resume,
            jd_id=jd.id,
            score=None,
            confidence=None,
            tier_used=Tier.RULES,
            skill_matches=[],
            reasoning=["Parse failed — recommend human review"],
            human_review=True,
        )
        cand.rules_floor = 0.0  # type: ignore[attr-defined]
        return cand

    # --- 2. Offline rules baseline: guaranteed floor + exact/synonym matches ---
    floor_score, rules_matches = rules_baseline(resume, jd)

    # --- 3. Attempt LLM enhancement and merge ---
    conflict_free_notes: list[str] = []
    llm_result = extractor_llm.enhance(resume, jd)
    if llm_result is not None:
        merged = _merge_matches(rules_matches, llm_result.skill_matches)
        weighted = weighted_score(resume, jd, merged)
        tier_used = llm_result.source_tier
        # Deciding required-skill evidence is weak when the required match that
        # counts is implicit/partial (rules tier only produces exact/synonym).
        has_implicit_or_partial = any(
            m.required and m.match_type in (MatchType.IMPLICIT, MatchType.PARTIAL)
            for m in merged
        )
    else:
        # LLM unavailable → offline rules baseline only (Req 10.4).
        merged = rules_matches
        weighted = weighted_score(resume, jd, merged)
        tier_used = Tier.RULES
        has_implicit_or_partial = False  # rules tier is exact/synonym only
        conflict_free_notes.append(
            "LLM unavailable — scored on offline rules baseline only"
        )

    # --- 6. Final score: LLM can only raise above the offline floor ---
    final_score = _clamp(max(floor_score, weighted), 0.0, 100.0)

    # --- 4. Confidence from parse × tier ceilings ---
    confidence = resolve_confidence(
        resume.parse_flag, tier_used, has_implicit_or_partial
    )

    # --- 5. Conflict resolution notes ---
    conflict_notes = apply_conflict_rules(resume, jd, merged, final_score)

    # Case 4 coupling: a high score on a partial parse forces confidence down to
    # Medium and flags for human review (design.md Signal Conflict Resolution).
    human_review = False
    if resume.parse_flag == ParseFlag.PARTIAL and final_score >= _HIGH_SCORE:
        if confidence is not None:
            confidence = _min_confidence(confidence, "Medium")
        human_review = True

    matched_required = sum(
        1 for m in merged if m.required and m.match_type != MatchType.MISSING
    )
    total_required = sum(1 for m in merged if m.required)

    cand = ScoredCandidate(
        resume=resume,
        jd_id=jd.id,
        score=final_score,
        confidence=confidence,
        tier_used=tier_used,
        skill_matches=merged,
        # Basic interim reasoning; task 10.2 builds the final three bullets.
        reasoning=[
            f"Matched {matched_required}/{total_required} required skills "
            f"({tier_used.value} tier).",
            f"Score {final_score:.1f}/100 (offline floor {floor_score:.1f}); "
            f"confidence {confidence}.",
            (
                conflict_free_notes[0]
                if conflict_free_notes
                else "Enhanced with LLM skill matching."
            ),
        ],
        human_review=human_review,
        conflict_notes=conflict_notes + conflict_free_notes,
    )
    # Stash the floor so downstream steps can reason about the guaranteed floor.
    cand.rules_floor = floor_score  # type: ignore[attr-defined]
    return cand
