"""Unit tests for engine.synonyms (canonicalization + synonym data)."""

from engine import synonyms as s


def test_react_variants_share_canonical():
    # Core requirement: exact + synonym matching share one key.
    assert s.canonical("ReactJS") == s.canonical("React.js") == s.canonical("React") == "react"


def test_language_abbreviations():
    assert s.canonical("JS") == s.canonical("JavaScript") == "javascript"
    assert s.canonical("TS") == s.canonical("TypeScript") == "typescript"
    assert s.canonical("py") == s.canonical("Python") == "python"
    assert s.canonical("golang") == s.canonical("Go") == "go"


def test_database_variants():
    assert s.canonical("postgres") == s.canonical("PostgreSQL") == "postgresql"
    assert s.canonical("mongo") == s.canonical("MongoDB") == "mongodb"


def test_rest_api_variants():
    for variant in ("REST", "REST API", "RESTful", "REST APIs"):
        assert s.canonical(variant) == "rest api"


def test_node_variants():
    for variant in ("Node", "Node.js", "NodeJS"):
        assert s.canonical(variant) == "nodejs"


def test_punctuation_and_whitespace_collapse():
    assert s.canonical("CI/CD") == "cicd"
    assert s.canonical("  html5  ") == "html"
    assert s.canonical("Amazon Web Services") == "aws"


def test_unknown_skill_returns_normalized_form():
    # Unknown skills fall through to their normalized lowercased form.
    assert s.canonical("  Kafka Streams ") == "kafka streams"


def test_empty_and_none_inputs():
    assert s.canonical("") == ""
    assert s.canonical("   ") == ""
    assert s.canonical(None) == ""


def test_vocabulary_contains_canonicals():
    assert "react" in s.SKILL_VOCABULARY
    assert "rest api" in s.SKILL_VOCABULARY
    # Every canonical value is present in the vocabulary set.
    for canon in s._SYNONYMS.values():
        assert canon in s.SKILL_VOCABULARY
