"""LLM extraction + matching with the Groq → Gemini → Ollama cascade.

Stage 2 optional enhancement layer. Given a parsed :class:`ResumeData` and a
:class:`JobDescription`, this module asks an LLM to (a) extract
skills/projects/experience from the raw resume text and (b) match those against
the JD's required + preferred skills, labelling each JD skill
exact/synonym/partial/implicit/missing with brief evidence.

Design references:
- Req 10.1 : Primary Cloud_Tier via Groq, temperature 0, JSON-only responses.
- Req 10.2 : On Groq failure/timeout, fall back to the Gemini backup.
- Req 10.3 : On both Cloud_Tier failures, fall back to Local_Tier (Ollama).
- Req 10.4 : If all tiers fail, return ``None`` so the scorer uses rules-only.
- Req 10.7 : Cache Cloud/Local responses keyed by resume_hash + jd_id and reuse
             them on repeat runs without a new API call.
- Req 10.8 : Uniform hard 3-second per-tier timeout (``config.LLM_TIMEOUT_S``),
             applied identically to every tier including Ollama; a timeout is
             treated exactly like a call failure.
- Req 18.1 : Null/timeout/partial-JSON responses are handled gracefully; the
             cascade advances and ``enhance`` never raises.
- Req 18.2 : Warm cache returns an identical result with zero network calls.
- Req 18.3 : temperature 0 on every tier for deterministic output.

The heavy third-party clients (``groq``, ``google.generativeai``, ``ollama``)
are imported lazily *inside* each tier function so this module imports cleanly
even when none of them are installed.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from . import config
from .jd import JobDescription
from .models import Experience, MatchType, Project, SkillMatch, Tier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class LLMResult:
    """Structured output of a successful LLM tier (Cloud or Local)."""

    skills: list[str]
    projects: list[Project]
    experience: list[Experience]
    skill_matches: list[SkillMatch]
    source_tier: Tier          # CLOUD (Groq/Gemini) or LOCAL (Ollama)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(resume, jd: JobDescription) -> str:
    """Build the single extract-and-match prompt sent to every tier.

    Asks for JSON-only output covering extraction (skills/projects/experience)
    and per-JD-skill matching with a label + brief evidence (Req 10.1, 11.x).
    """
    required = ", ".join(jd.required_skills) or "(none)"
    preferred = ", ".join(jd.preferred_skills) or "(none)"
    # Bound the raw text so a huge resume cannot blow the prompt budget; the
    # first ~6000 chars carry the identity/skills/experience in practice.
    raw_text = (resume.raw_text or "")[:6000]

    return (
        "You are an expert technical recruiter. Read the RAW RESUME TEXT and the "
        "JOB DESCRIPTION skills, then return ONLY a single JSON object (no prose, "
        "no code fences) with exactly these keys:\n"
        '  "skills": [string]            // all technical skills found in the resume\n'
        '  "projects": [{"title": string, "description": string}]\n'
        '  "experience": [{"company": string, "role": string, "duration": string}]\n'
        '  "skill_matches": [{"jd_skill": string, "match_type": string, '
        '"evidence": string, "required": boolean}]\n'
        "\n"
        "For skill_matches, produce one entry for EVERY JD skill listed below "
        "(both required and preferred). match_type must be one of: "
        '"exact" (candidate states the same skill), "synonym" (equivalent term), '
        '"partial" (related but not the same), "implicit" (demonstrated in a '
        'project/experience without naming the term), or "missing" (no evidence). '
        "Set required=true for required skills and required=false for preferred "
        "skills. Keep evidence to a short phrase quoting the resume. Do NOT "
        "fabricate skills that are not supported by the text.\n"
        "\n"
        f"REQUIRED SKILLS: {required}\n"
        f"PREFERRED SKILLS: {preferred}\n"
        "\n"
        "RAW RESUME TEXT:\n"
        f"{raw_text}\n"
    )


# ---------------------------------------------------------------------------
# JSON robustness (Req 18.1)
# ---------------------------------------------------------------------------


def safe_json(text: Optional[str]) -> Optional[dict]:
    """Best-effort parse of an LLM text response into a dict.

    Strips ```json fences and attempts one brace-repair (closing a single
    unbalanced trailing brace). Returns the parsed ``dict`` on success or
    ``None`` on any failure — never raises (Req 18.1).
    """
    if not text:
        return None

    cleaned = text.strip()

    # Strip Markdown code fences (```json ... ``` or ``` ... ```).
    if cleaned.startswith("```"):
        # Drop the opening fence line.
        newline = cleaned.find("\n")
        if newline != -1:
            cleaned = cleaned[newline + 1:]
        # Drop a trailing fence if present.
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()

    # If the model wrapped the object in extra prose, isolate the outermost
    # brace span.
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last >= first:
        candidate = cleaned[first:last + 1]
    else:
        candidate = cleaned

    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        # One repair attempt: balance braces by appending missing closers.
        opens = candidate.count("{")
        closes = candidate.count("}")
        if opens > closes:
            repaired = candidate + ("}" * (opens - closes))
            try:
                parsed = json.loads(repaired)
            except (ValueError, TypeError):
                return None
        else:
            return None

    if isinstance(parsed, dict):
        return parsed
    return None


# ---------------------------------------------------------------------------
# JSON → LLMResult mapping
# ---------------------------------------------------------------------------


def _parse_llm_result(data: dict, tier: Tier) -> LLMResult:
    """Map a parsed JSON dict to an :class:`LLMResult`, tolerating missing keys.

    Unknown/invalid match-type labels degrade to ``MISSING``; credit is read
    from ``config.CREDIT`` per label. Malformed sub-entries are skipped rather
    than raising so a partially-valid response still yields a usable result.
    """
    # --- skills ---
    skills: list[str] = []
    for item in data.get("skills") or []:
        if isinstance(item, str) and item.strip():
            skills.append(item.strip())

    # --- projects ---
    projects: list[Project] = []
    for item in data.get("projects") or []:
        if isinstance(item, dict):
            title = item.get("title")
            if isinstance(title, str) and title.strip():
                desc = item.get("description")
                projects.append(
                    Project(
                        title=title.strip(),
                        description=desc.strip() if isinstance(desc, str) and desc.strip() else None,
                    )
                )
        elif isinstance(item, str) and item.strip():
            projects.append(Project(title=item.strip()))

    # --- experience ---
    experience: list[Experience] = []
    for item in data.get("experience") or []:
        if isinstance(item, dict):
            company = item.get("company")
            role = item.get("role")
            duration = item.get("duration")
            experience.append(
                Experience(
                    company=company.strip() if isinstance(company, str) and company.strip() else None,
                    role=role.strip() if isinstance(role, str) and role.strip() else None,
                    duration=duration.strip() if isinstance(duration, str) and duration.strip() else None,
                )
            )

    # --- skill matches ---
    skill_matches: list[SkillMatch] = []
    for item in data.get("skill_matches") or []:
        if not isinstance(item, dict):
            continue
        jd_skill = item.get("jd_skill")
        if not isinstance(jd_skill, str) or not jd_skill.strip():
            continue

        label = str(item.get("match_type", "")).strip().lower()
        try:
            match_type = MatchType(label)
        except ValueError:
            match_type = MatchType.MISSING

        evidence = item.get("evidence")
        evidence = evidence.strip() if isinstance(evidence, str) and evidence.strip() else None

        required = item.get("required")
        required = bool(required) if isinstance(required, bool) else True

        credit = config.CREDIT.get(match_type.value, 0.0)

        skill_matches.append(
            SkillMatch(
                jd_skill=jd_skill.strip(),
                match_type=match_type,
                evidence=evidence,
                required=required,
                credit=credit,
            )
        )

    return LLMResult(
        skills=skills,
        projects=projects,
        experience=experience,
        skill_matches=skill_matches,
        source_tier=tier,
    )


# ---------------------------------------------------------------------------
# Disk cache (Req 10.7, 18.2)
# ---------------------------------------------------------------------------


def _cache_key(resume, jd: JobDescription) -> str:
    """Deterministic cache key = sha256(resume_hash :: jd.id) (Req 18.2)."""
    basis = f"{resume.resume_hash}::{jd.id}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(config.CACHE_DIR, f"{key}.json")


def _read_cache(resume, jd: JobDescription) -> Optional[LLMResult]:
    """Return a cached :class:`LLMResult` if present, else ``None``.

    Never raises: a missing/corrupt cache file is treated as a miss so a fresh
    call is made instead (Req 18.1).
    """
    path = _cache_path(_cache_key(resume, jd))
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    tier_value = data.get("source_tier", Tier.CLOUD.value)
    try:
        tier = Tier(tier_value)
    except ValueError:
        tier = Tier.CLOUD
    return _parse_llm_result(data, tier)


def _write_cache(resume, jd: JobDescription, result: LLMResult) -> None:
    """Serialize an :class:`LLMResult` to the disk cache (Req 10.7).

    Creates ``CACHE_DIR`` if missing. Never raises on write failure — the caller
    still returns the freshly computed result.
    """
    try:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        payload = {
            "skills": result.skills,
            "projects": [
                {"title": p.title, "description": p.description} for p in result.projects
            ],
            "experience": [
                {"company": e.company, "role": e.role, "duration": e.duration}
                for e in result.experience
            ],
            "skill_matches": [
                {
                    "jd_skill": m.jd_skill,
                    "match_type": m.match_type.value,
                    "evidence": m.evidence,
                    "required": m.required,
                }
                for m in result.skill_matches
            ],
            "source_tier": result.source_tier.value,
        }
        path = _cache_path(_cache_key(resume, jd))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("Failed to write LLM cache: %s", exc)


# ---------------------------------------------------------------------------
# Timeout wrapper (Req 10.8)
# ---------------------------------------------------------------------------


def _run_with_timeout(func, *args, **kwargs) -> Optional[str]:
    """Run ``func`` with a hard ``config.LLM_TIMEOUT_S`` timeout.

    Uses a thread pool + ``future.result(timeout=...)`` so a hanging call is
    abandoned (the worker thread is left to die on its own). Any timeout or
    exception yields ``None`` so the caller advances the cascade (Req 10.8,
    18.1). Applied uniformly to every tier including Ollama.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=config.LLM_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        logger.info("LLM tier timed out after %ss", config.LLM_TIMEOUT_S)
        return None
    except Exception as exc:  # noqa: BLE001 - any failure = tier failure
        logger.info("LLM tier raised: %s", exc)
        return None
    finally:
        # Do not block on a hung worker thread.
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Tier implementations (lazy imports inside each)
# ---------------------------------------------------------------------------


def _call_groq(prompt: str) -> Optional[str]:
    """Groq Cloud_Tier call — OpenAI-SDK-compatible client (Req 10.1)."""
    if not config.GROQ_API_KEY:
        logger.info("Groq tier skipped: no API key configured.")
        return None
    try:
        from groq import Groq
    except ImportError:
        logger.info("Groq tier skipped: 'groq' package not installed.")
        return None

    def _do() -> Optional[str]:
        client = Groq(api_key=config.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    return _run_with_timeout(_do)


def _call_gemini(prompt: str) -> Optional[str]:
    """Gemini backup Cloud_Tier call via google-generativeai (Req 10.2)."""
    if not config.GEMINI_API_KEY:
        logger.info("Gemini tier skipped: no API key configured.")
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        logger.info("Gemini tier skipped: 'google-generativeai' not installed.")
        return None

    def _do() -> Optional[str]:
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )
        return getattr(resp, "text", None)

    return _run_with_timeout(_do)


def _call_ollama(prompt: str) -> Optional[str]:
    """Local_Tier call via the ollama client (Req 10.3), same 3s timeout."""
    try:
        import ollama
    except ImportError:
        logger.info("Ollama tier skipped: 'ollama' package not installed.")
        return None

    def _do() -> Optional[str]:
        client = ollama.Client(host=config.OLLAMA_HOST)
        resp = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        # ollama returns a dict-like with message.content.
        message = resp.get("message") if isinstance(resp, dict) else None
        if isinstance(message, dict):
            return message.get("content")
        return getattr(getattr(resp, "message", None), "content", None)

    return _run_with_timeout(_do)


# Cascade order: (label, callable, tier). Groq → Gemini → Ollama.
_CASCADE = (
    ("groq", _call_groq, Tier.CLOUD),
    ("gemini", _call_gemini, Tier.CLOUD),
    ("ollama", _call_ollama, Tier.LOCAL),
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def enhance(resume, jd: JobDescription) -> Optional[LLMResult]:
    """Enhance a resume-JD pair with LLM extraction + matching.

    Cache-first (Req 10.7, 18.2): if a cached result exists for
    ``resume_hash + jd.id`` it is returned WITHOUT any network call. Otherwise
    the cascade runs Groq → Gemini → Ollama; the first tier that returns
    parseable JSON produces an :class:`LLMResult`, which is written to the cache
    and returned. If every tier fails (missing keys/libs, timeout, error, empty
    or unparseable response) this returns ``None`` and never raises (Req 10.4,
    18.1).
    """
    # 1) Cache lookup — no network on a hit.
    cached = _read_cache(resume, jd)
    if cached is not None:
        logger.info("LLM cache hit for %s / %s", resume.resume_hash[:8], jd.id)
        return cached

    # 2) Run the cascade.
    prompt = _build_prompt(resume, jd)
    for label, call, tier in _CASCADE:
        raw = call(prompt)
        data = safe_json(raw)
        if not data:
            logger.info("Tier %s produced no usable JSON; advancing cascade.", label)
            continue

        result = _parse_llm_result(data, tier)
        _write_cache(resume, jd, result)
        logger.info("Tier %s succeeded (source_tier=%s).", label, tier.value)
        return result

    # 3) All tiers failed — scorer falls back to the rules-only baseline.
    logger.info("All LLM tiers failed for %s / %s; returning None.", resume.resume_hash[:8], jd.id)
    return None
