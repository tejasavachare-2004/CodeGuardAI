from enum import Enum

from orchestrator.graph import (
    _findings_from_report,
    _normalize_finding,
    build_review_graph,
)


class SampleSeverity(Enum):
    HIGH = "HIGH"


def test_normalization_adds_common_fields_and_preserves_agent_data():
    finding = {
        "title": "Unsafe query",
        "severity": SampleSeverity.HIGH,
        "category": "security",
        "file": "app.py",
        "line": 12,
        "evidence": "query = user_input",
        "cwe": "CWE-89",
        "owasp": "A03:2021-Injection",
        "attack_scenario": "Attacker changes the query.",
        "recommendation": "Use parameters.",
        "validation_status": "VERIFIED",
        "source": "static",
        "confidence": 0.75,
        "related_files": ["db.py"],
    }

    normalized = _normalize_finding("security_agent", finding)

    assert {
        "id",
        "reported_by",
        "category",
        "subcategory",
        "code",
        "message",
        "file",
        "line",
        "column",
        "severity",
        "confidence",
        "confidence_scale",
    } <= normalized.keys()
    assert normalized["reported_by"] == "security_agent"
    assert normalized["category"] == "security"
    assert normalized["code"] == "CWE-89"
    assert normalized["message"] == "Unsafe query"
    assert normalized["severity"] == "HIGH"
    assert normalized["confidence"] == 0.75
    assert normalized["confidence_scale"] == "0-1"
    assert normalized["evidence"] == finding["evidence"]
    assert normalized["owasp"] == finding["owasp"]
    assert normalized["attack_scenario"] == finding["attack_scenario"]
    assert normalized["recommendation"] == finding["recommendation"]
    assert normalized["validation_status"] == finding["validation_status"]
    assert normalized["related_files"] == finding["related_files"]


def test_normalization_handles_missing_optional_fields_without_confidence():
    normalized = _normalize_finding("style_agent", {"code": "D100"})

    assert normalized["category"] == "style"
    assert normalized["file"] is None
    assert normalized["line"] is None
    assert normalized["column"] is None
    assert normalized["confidence"] is None
    assert normalized["confidence_scale"] is None


def test_normalization_preserves_code_reviewer_confidence_scale():
    normalized = _normalize_finding(
        "code_reviewer", {"confidence": 75.0, "title": "Possible bug"}
    )

    assert normalized["confidence"] == 75.0
    assert normalized["confidence_scale"] == "0-100"


def test_finding_id_is_stable_for_the_same_finding():
    finding = {
        "file": "app.py",
        "line": 12,
        "code": "D100",
        "message": "Missing docstring",
    }

    first = _normalize_finding("style_agent", finding)
    second = _normalize_finding("style_agent", finding)

    assert first["id"] == second["id"]


def test_architecture_category_and_related_files_are_preserved():
    finding = {
        "category": "layering",
        "related_files": ["app/repositories/user.py"],
        "source": "static",
        "validation_status": "VERIFIED",
    }

    normalized = _normalize_finding("architecture_agent", finding)

    assert normalized["category"] == "layering"
    assert normalized["related_files"] == finding["related_files"]
    assert normalized["source"] == "static"
    assert normalized["validation_status"] == "VERIFIED"
    assert normalized["confidence"] is None


def test_malformed_findings_are_preserved_as_raw_without_crashing():
    warnings = []

    findings = _findings_from_report(
        "security_agent", {"findings": [None, {"title": "Partial"}]}, warnings
    )

    assert len(findings) == 2
    assert findings[0]["raw"] is None
    assert findings[0]["reported_by"] == "security_agent"
    assert findings[1]["title"] == "Partial"
    assert warnings == [
        "security_agent: finding 0 is not an object; preserved as raw data."
    ]


def test_build_review_graph_returns_compiled_graph():
    graph = build_review_graph()

    assert graph is not None
    assert callable(graph.invoke)
