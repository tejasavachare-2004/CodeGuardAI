
from orchestrator.deduplicator import deduplicate_findings


def make_finding(agent, title, line=10, file="app.py"):
    return {
        "reported_by": agent,
        "file": file,
        "line": line,
        "title": title,
        "message": title,
        "severity": "HIGH",
        "confidence": 0.9,
    }


def test_exact_duplicates():
    findings = [
        make_finding("security_agent", "SQL injection vulnerability"),
        make_finding("code_reviewer", "SQL injection vulnerability"),
    ]

    result = deduplicate_findings(findings)

    assert len(result) == 1
    assert set(result[0]["reported_by"]) == {
        "security_agent",
        "code_reviewer",
    }


def test_different_files_are_not_duplicates():
    findings = [
        make_finding("security_agent", "SQL injection", file="a.py"),
        make_finding("code_reviewer", "SQL injection", file="b.py"),
    ]

    assert len(deduplicate_findings(findings)) == 2


def test_different_lines_are_not_duplicates():
    findings = [
        make_finding("security_agent", "SQL injection", line=10),
        make_finding("code_reviewer", "SQL injection", line=20),
    ]

    assert len(deduplicate_findings(findings)) == 2


def test_different_issues_on_same_line():
    findings = [
        make_finding("security_agent", "SQL injection vulnerability"),
        make_finding("style_agent", "Missing function docstring"),
    ]

    assert len(deduplicate_findings(findings)) == 2


def test_empty_input():
    assert deduplicate_findings([]) == []


def test_missing_optional_fields():
    findings = [
        {"file": "app.py", "line": 10, "message": "Issue A"},
        {"file": "app.py", "line": 10, "message": "Issue B"},
    ]

    result = deduplicate_findings(findings)

    assert len(result) == 2


def test_deterministic_output():
    findings = [
        make_finding("security_agent", "SQL injection"),
        make_finding("code_reviewer", "SQL injection"),
    ]

    first = deduplicate_findings(findings)
    second = deduplicate_findings(findings)

    assert first == second


def test_duplicate_findings_merge_integration():
    findings = [
        {
            "id": "finding-1",
            "reported_by": "security_agent",
            "file": "app.py",
            "line": 10,
            "title": "SQL injection vulnerability detected",
            "message": "SQL injection vulnerability detected",
            "severity": "HIGH",
            "confidence": 0.9,
            "cwe": "CWE-89",
        },
        {
            "id": "finding-2",
            "reported_by": "code_reviewer",
            "file": "app.py",
            "line": 10,
            "title": "SQL injection vulnerability detected",
            "message": "SQL injection vulnerability detected",
            "severity": "HIGH",
            "confidence": 85,
            "cwe": "CWE-89",
        },
    ]

    result = deduplicate_findings(findings)

    assert len(result) == 1
    assert set(result[0]["reported_by"]) == {
        "security_agent",
        "code_reviewer",
    }
    assert result[0]["confidence"] == 0.9
    assert result[0]["agent_confidences"] == {
        "security_agent": 0.9,
        "code_reviewer": 85,
    }
