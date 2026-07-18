"""Synonym data + canonicalization for skill matching (scoring-data layer).

This module is pure data plus pure functions. It MUST NOT import from any other
`engine` module so the scoring-data layer stays dependency-free and trivially
testable (design.md: dependency direction is one-way, `synonyms.py` is a leaf).

The rules tier (`scorer.py`) and the LLM merge step both route candidate and JD
skills through :func:`canonical` so that surface variants like ``"ReactJS"``,
``"React.js"`` and ``"React"`` collapse onto a single shared key (``"react"``).
Exact matching and synonym matching therefore compare canonical forms, giving
both tiers one consistent notion of "the same skill" (Req 9.1, 11.2).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Synonym dictionary
# ---------------------------------------------------------------------------
# Maps a *normalized* skill variant (see `_normalize`) to its canonical form.
# Keys are stored in normalized form (lowercase, punctuation/whitespace
# collapsed) so a raw input only needs `_normalize` applied before lookup.
#
# Canonical forms are the stable keys used everywhere downstream. Covers the
# common tech-resume variants relevant to the 5 JDs (Frontend, Backend,
# Full Stack, Database, API Integration) plus widely-seen tooling.

_RAW_SYNONYMS: dict[str, str] = {
    # --- Frontend frameworks / libraries ---
    "react.js": "react",
    "reactjs": "react",
    "react js": "react",
    "react": "react",
    "next.js": "nextjs",
    "nextjs": "nextjs",
    "next js": "nextjs",
    "redux": "redux",
    "zustand": "zustand",
    # --- Languages ---
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "py": "python",
    "python": "python",
    "golang": "go",
    "go": "go",
    # --- Node / backend runtimes & frameworks ---
    "node": "nodejs",
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "express": "express",
    "express.js": "express",
    "expressjs": "express",
    "flask": "flask",
    "django": "django",
    # --- Databases ---
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "postgre sql": "postgresql",
    "mysql": "mysql",
    "mongo": "mongodb",
    "mongodb": "mongodb",
    "redis": "redis",
    "firebase": "firebase",
    # --- API / integration ---
    "rest": "rest api",
    "rest api": "rest api",
    "restful": "rest api",
    "rest apis": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
    "graphql": "graphql",
    "jwt": "jwt",
    "oauth": "oauth",
    "oauth2": "oauth",
    "oauth 2": "oauth",
    "swagger": "swagger",
    "openapi": "swagger",
    "postman": "postman",
    # --- Markup / styling ---
    "html": "html",
    "html5": "html",
    "css": "css",
    "css3": "css",
    # --- DevOps / cloud / tooling ---
    "docker": "docker",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "git": "git",
    "github": "git",
    "ci/cd": "cicd",
    "ci cd": "cicd",
    "cicd": "cicd",
    "aws": "aws",
    "amazon web services": "aws",
    "gcp": "gcp",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "azure": "azure",
    "microsoft azure": "azure",
    # --- Testing ---
    "jest": "jest",
}


def _normalize(skill: str) -> str:
    """Normalize a raw skill string to a stable lookup/return form.

    Steps: lowercase, strip, then collapse runs of punctuation/whitespace into
    a single space. Dots inside tokens (``react.js``) become spaces too, so the
    normalized form is comparable across ``React.js`` / ``React js`` etc. A few
    combined-token forms are preserved via the synonym table (e.g. ``ci/cd``).
    """
    if skill is None:
        return ""
    s = skill.lower().strip()
    # Replace any run of characters that are not letters/digits with a single
    # space. This folds ".", "/", "-", "_", and repeated whitespace uniformly.
    s = re.sub(r"[^a-z0-9]+", " ", s)
    # Collapse whitespace and trim.
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Build the lookup table keyed by fully-normalized variants so that inputs like
# "CI/CD" and "ci cd" both resolve. We normalize each raw key with the same
# `_normalize` used on inputs to guarantee they line up.
_SYNONYMS: dict[str, str] = {_normalize(k): v for k, v in _RAW_SYNONYMS.items()}


def canonical(skill: str) -> str:
    """Return the canonical form of a skill for matching.

    Normalizes the input (lowercase, strip, collapse punctuation/whitespace)
    and looks it up in the synonym map. If the normalized form is a known
    variant, its canonical name is returned; otherwise the normalized form is
    returned unchanged. This gives exact matching and synonym matching a shared
    key, e.g. ``canonical("ReactJS") == canonical("React.js") == "react"``.
    """
    norm = _normalize(skill)
    if not norm:
        return ""
    return _SYNONYMS.get(norm, norm)


# All distinct canonical skill names, for full-text vocabulary scanning by the
# rules tier (Req 4.4, 11.2). This is the set the offline scanner treats as the
# known skill vocabulary.
SKILL_VOCABULARY: frozenset[str] = frozenset(_SYNONYMS.values())
