"""Job Description model and loader (scoring layer).

Loads a Job Description config from JSON or YAML into a :class:`JobDescription`.
Only ``id``, ``title``, and ``required_skills`` are required; every other field
has a documented default baked into both the dataclass and the loader, so a JD
missing optional fields still yields a valid ranked Shortlist/Reserve rather
than an error (Req 15.3). Unknown/extra keys are ignored so a judge can add a
6th JD file with extra fields and run it unchanged (Req 15.1, 15.2).

Design references:
- Req 15.1 : Process any JD (required/preferred skills, Slot_Count, min CGPA).
- Req 15.2 : An unseen JD produces a Shortlist/Reserve via the same cascade.
- Req 15.3 : Missing optional fields apply documented defaults, not failures.
- Req 19.3 : Accept a JD via a --jd file path (JSON or YAML).

Applied defaults are logged so the operator can see what was filled in.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class JobDescription:
    """A role specification used by the scorer.

    ``min_cgpa`` is on the normalized 10-point scale. Optional fields default to
    the documented values in :mod:`engine.config` (Req 15.3).
    """

    id: str
    title: str
    required_skills: list[str]
    preferred_skills: list[str] = field(default_factory=list)
    slots: int = config.DEFAULT_SLOTS       # Slot_Count N (default 5)
    min_cgpa: float = config.DEFAULT_MIN_CGPA  # on 10-pt normalized scale (default 0.0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# Fields that must be present in the config; absence is a hard error (Req 15.3
# only defaults *optional* fields).
_REQUIRED_KEYS = ("id", "title", "required_skills")

# Known optional fields and their documented defaults (Req 15.3). Kept here so
# the loader can log exactly which defaults were applied.
_OPTIONAL_DEFAULTS = {
    "preferred_skills": lambda: [],
    "slots": lambda: config.DEFAULT_SLOTS,
    "min_cgpa": lambda: config.DEFAULT_MIN_CGPA,
}


def _read_config_file(path: str) -> dict:
    """Read a JD config file, choosing the parser by extension.

    ``.json`` uses :func:`json.load`; ``.yaml``/``.yml`` uses
    ``yaml.safe_load``. YAML is only imported when needed so the engine does not
    hard-require PyYAML for JSON-only runs.
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as fh:
        if ext == ".json":
            data = json.load(fh)
        elif ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ImportError(
                    "PyYAML is required to load YAML Job Description files; "
                    "install it or provide a .json JD instead."
                ) from exc
            data = yaml.safe_load(fh)
        else:
            raise ValueError(
                f"Unsupported Job Description file extension {ext!r} for {path!r}; "
                "expected .json, .yaml, or .yml."
            )

    if not isinstance(data, dict):
        raise ValueError(
            f"Job Description file {path!r} must contain a mapping/object at the top level."
        )
    return data


def load_jd(path: str) -> JobDescription:
    """Load one Job Description from a JSON or YAML file.

    Only ``id``, ``title``, and ``required_skills`` are required. Missing
    optional fields (``preferred_skills``, ``slots``, ``min_cgpa``) receive their
    documented defaults, and unknown/extra keys are ignored so an augmented 6th
    JD still loads (Req 15.1, 15.2, 15.3). Applied defaults are logged.
    """
    raw = _read_config_file(path)

    missing = [key for key in _REQUIRED_KEYS if key not in raw or raw[key] is None]
    if missing:
        raise ValueError(
            f"Job Description file {path!r} is missing required field(s): "
            f"{', '.join(missing)}."
        )

    # Build the known-field set, applying documented defaults where absent.
    applied_defaults: list[str] = []
    values: dict = {
        "id": str(raw["id"]),
        "title": str(raw["title"]),
        "required_skills": list(raw["required_skills"]),
    }
    for key, default_factory in _OPTIONAL_DEFAULTS.items():
        if key in raw and raw[key] is not None:
            values[key] = raw[key]
        else:
            values[key] = default_factory()
            applied_defaults.append(f"{key}={values[key]!r}")

    # Coerce to declared types defensively (e.g. YAML may give ints/floats).
    values["preferred_skills"] = list(values["preferred_skills"])
    values["slots"] = int(values["slots"])
    values["min_cgpa"] = float(values["min_cgpa"])

    if applied_defaults:
        logger.info(
            "JD %r: applied default(s) for missing optional field(s): %s",
            values["id"],
            ", ".join(applied_defaults),
        )

    # Ignore unknown/extra keys (Req 15.1) — log them for transparency.
    known = set(_REQUIRED_KEYS) | set(_OPTIONAL_DEFAULTS)
    extra = [key for key in raw if key not in known]
    if extra:
        logger.info(
            "JD %r: ignoring unknown extra key(s): %s",
            values["id"],
            ", ".join(sorted(extra)),
        )

    return JobDescription(**values)


def load_jds(paths: list[str]) -> list[JobDescription]:
    """Load multiple Job Descriptions, preserving input order (Req 19.3)."""
    return [load_jd(path) for path in paths]


# ---------------------------------------------------------------------------
# Live JD parsing (bonus B — not implemented yet)
# ---------------------------------------------------------------------------


def parse_live_jd(text: str) -> JobDescription:
    """Parse an unstructured free-text Job Description into a JobDescription.

    TODO(bonus 13.2 / Req 20.2): LLM-assisted extraction of required/preferred
    skills and minimum CGPA from raw text, applying the default Slot_Count when
    none is stated. Intentionally unimplemented in the MVP path.
    """
    raise NotImplementedError(
        "Live JD parsing is a bonus feature (task 13.2, Req 20.2) and is not "
        "yet implemented; provide a JSON or YAML JD via load_jd instead."
    )
