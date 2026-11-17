
import re
from difflib import SequenceMatcher
from typing import Any


def _normalize_text(value: Any) -> str:
    """Normalize text for duplicate comparison."""
    if not value:
        return ""

    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _similarity(first: str, second: str) -> float:
    """Calculate similarity between two strings."""
    if not first or not second:
        return 0.0

    return SequenceMatcher(None, first, second).ratio()


def _is_duplicate(first: dict, second: dict) -> bool:
    """Determine whether two findings describe the same issue."""

    file1 = _normalize_text(first.get("file"))
    file2 = _normalize_text(second.get("file"))

    if not file1 or file1 != file2:
        return False

    line1 = first.get("line")
    line2 = second.get("line")

    if not isinstance(line1, int) or not isinstance(line2, int):
        return False

    if abs(line1 - line2) > 2:
        return False

    code1 = _normalize_text(first.get("code"))
    code2 = _normalize_text(second.get("code"))

    cwe1 = _normalize_text(first.get("cwe"))
    cwe2 = _normalize_text(second.get("cwe"))

    title1 = _normalize_text(first.get("title") or first.get("message"))
    title2 = _normalize_text(second.get("title") or second.get("message"))

    evidence1 = _normalize_text(first.get("evidence"))
    evidence2 = _normalize_text(second.get("evidence"))

    if cwe1 and cwe1 == cwe2:
        return True

    if code1 and code1 == code2:
        return True

    if len(title1) < 20 or len(title2) < 20:
        return False

    title_similarity = _similarity(title1, title2)
    evidence_similarity = _similarity(evidence1, evidence2)

    return (
        title_similarity >= 0.90
        or (
            title_similarity >= 0.70
            and evidence_similarity >= 0.85
            and bool(evidence1)
            and bool(evidence2)
        )
    )


def _get_agents(finding: dict) -> list:
    """Get all agent names associated with a finding."""

    agents = finding.get("reported_by") or finding.get("agent")

    if isinstance(agents, list):
        return [agent for agent in agents if agent]

    return [agents] if agents else []


def _preserve_agent_information(target: dict, source: dict) -> None:
    """Preserve agent attribution and individual confidence values."""

    agents = _get_agents(source)

    if not isinstance(target.get("reported_by"), list):
        target["reported_by"] = []

    for agent in agents:
        if agent not in target["reported_by"]:
            target["reported_by"].append(agent)

    confidence = source.get("confidence")

    if confidence is not None:
        agent_confidences = target.setdefault("agent_confidences", {})

        for agent in agents:
            agent_confidences.setdefault(agent, confidence)


def deduplicate_findings(findings: list[dict]) -> list[dict]:
    """
    Remove duplicate findings while preserving information
    and attribution from contributing agents.
    """

    if not findings:
        return []

    result = []

    for finding in findings:
        if not isinstance(finding, dict):
            continue

        duplicate = None

        for existing in result:
            if _is_duplicate(existing, finding):
                duplicate = existing
                break

        if duplicate is None:
            merged = finding.copy()

            _preserve_agent_information(merged, finding)

            result.append(merged)
            continue

        # Preserve contributing agents and their confidence values.
        _preserve_agent_information(duplicate, finding)

        # Preserve additional metadata without overwriting existing data.
        for key, value in finding.items():
            if key not in duplicate or duplicate[key] in (None, "", [], {}):
                if key not in ("reported_by", "agent_confidences"):
                    duplicate[key] = value

    return result
