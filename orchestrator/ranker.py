
"""
Rank findings based on severity, confidence, and impact.
"""


SEVERITY_SCORES = {
    "CRITICAL": 100,
    "HIGH": 75,
    "MEDIUM": 50,
    "LOW": 25,
    "INFO": 10,
}

IMPACT_SCORES = {
    "CRITICAL": 100,
    "HIGH": 80,
    "MEDIUM": 60,
    "LOW": 30,
    "INFO": 10,
}


def _normalize_confidence(confidence):
    """Convert confidence to a common 0-100 scale."""

    if confidence is None:
        return 50.0

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return 50.0

    if 0 <= confidence <= 1:
        return confidence * 100

    if 0 <= confidence <= 100:
        return confidence

    return 50.0


def calculate_priority(finding):
    """
    Calculate priority using severity, confidence, and impact.

    Severity: 40%
    Confidence: 30%
    Impact: 30%
    """

    severity = str(finding.get("severity", "MEDIUM")).upper()
    impact = str(finding.get("impact", severity)).upper()

    severity_score = SEVERITY_SCORES.get(severity, 50)
    confidence_score = _normalize_confidence(
        finding.get("confidence")
    )
    impact_score = IMPACT_SCORES.get(impact, 50)

    priority = (
        severity_score * 0.40
        + confidence_score * 0.30
        + impact_score * 0.30
    )

    return round(priority, 2)


def rank_findings(findings):
    """
    Add priority scores and sort findings from highest to lowest.
    """

    ranked = []

    for finding in findings:
        if not isinstance(finding, dict):
            continue

        item = finding.copy()
        item["priority_score"] = calculate_priority(item)

        ranked.append(item)

    ranked.sort(
        key=lambda finding: finding["priority_score"],
        reverse=True,
    )

    return ranked
