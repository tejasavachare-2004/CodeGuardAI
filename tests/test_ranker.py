
from orchestrator.ranker import (
    calculate_priority,
    rank_findings,
)


def test_critical_finding_has_high_priority():
    finding = {
        "severity": "CRITICAL",
        "confidence": 0.95,
        "impact": "CRITICAL",
    }

    assert calculate_priority(finding) == 98.5


def test_low_severity_has_lower_priority():
    finding = {
        "severity": "LOW",
        "confidence": 0.9,
        "impact": "LOW",
    }

    assert calculate_priority(finding) == 46.0


def test_confidence_scales_are_normalized():
    first = {
        "severity": "HIGH",
        "confidence": 0.9,
        "impact": "HIGH",
    }

    second = {
        "severity": "HIGH",
        "confidence": 90,
        "impact": "HIGH",
    }

    assert calculate_priority(first) == calculate_priority(second)


def test_original_confidence_is_preserved():
    finding = {
        "severity": "HIGH",
        "confidence": 0.9,
    }

    rank_findings([finding])

    assert finding["confidence"] == 0.9


def test_findings_are_sorted_by_priority():
    findings = [
        {"severity": "LOW", "confidence": 0.8},
        {"severity": "CRITICAL", "confidence": 0.95},
        {"severity": "MEDIUM", "confidence": 0.8},
    ]

    result = rank_findings(findings)

    scores = [item["priority_score"] for item in result]

    assert scores == sorted(scores, reverse=True)


def test_empty_findings():
    assert rank_findings([]) == []
