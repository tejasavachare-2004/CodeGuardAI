
from orchestrator.report_generator import (
    generate_markdown_report,
    generate_repository_markdown_report,
)


def test_report_contains_summary():
    review = {
        "repo_root": "project",
        "filename": "app.py",
        "summary": {
            "total_findings": 1,
            "successful_agents": 4,
            "failed_agents": 0,
            "findings_by_severity": {
                "MEDIUM": 1
            },
        },
        "all_findings": [],
        "errors": {},
    }

    report = generate_markdown_report(review)

    assert "# CodeGuard AI - Code Review Report" in report
    assert "Total Findings:** 1" in report
    assert "Successful Agents:** 4" in report


def test_report_contains_finding_details():
    review = {
        "filename": "app.py",
        "summary": {
            "total_findings": 1,
            "findings_by_severity": {"HIGH": 1},
        },
        "all_findings": [
            {
                "message": "SQL injection vulnerability",
                "severity": "HIGH",
                "priority_score": 95.0,
                "file": "app.py",
                "line": 10,
                "code": "CWE-89",
                "reported_by": ["security_agent"],
            }
        ],
        "errors": {},
    }

    report = generate_markdown_report(review)

    assert "SQL injection vulnerability" in report
    assert "Priority Score:** 95.0" in report
    assert "security_agent" in report


def test_empty_report():
    review = {
        "summary": {
            "total_findings": 0,
            "findings_by_severity": {},
        },
        "all_findings": [],
        "errors": {},
    }

    report = generate_markdown_report(review)

    assert "No issues found" in report


def test_failed_agent_status():
    review = {
        "summary": {},
        "all_findings": [],
        "errors": {
            "security_agent": "Connection failed"
        },
    }

    report = generate_markdown_report(review)

    assert "security_agent:** FAILED" in report


def test_partial_agent_status_is_not_reported_as_failed():
    review = {
        "summary": {
            "successful_agents": 2,
            "partial_agents": 1,
            "failed_agents": 1,
            "agent_statuses": {
                "code_reviewer": "PARTIAL",
                "security_agent": "FAILED",
                "style_agent": "SUCCESS",
                "architecture_agent": "SUCCESS",
            },
        },
        "all_findings": [],
        "errors": {
            "code_reviewer": "PartialReviewError: bounded AST-selected context",
            "security_agent": "RuntimeError: scanner failed",
        },
    }

    report = generate_markdown_report(review)

    assert "Partial Agents:** 1" in report
    assert "code_reviewer:** PARTIAL" in report
    assert "Static analysis processed the full source" in report
    assert "security_agent:** FAILED" in report
    assert "style_agent:** COMPLETED" in report


def test_repository_markdown_report():
    result = {
        "repo_root": "sample_project",
        "total_files": 2,
        "reviewed_files": 2,
        "failed_files": 0,
        "file_results": {
            "main.py": {},
            "utils.py": {},
        },
        "all_findings": [
            {
                "file": "main.py",
                "severity": "HIGH",
                "priority_score": 85,
                "line": 10,
                "message": "Possible SQL injection",
                "code": "SQL001",
            },
            {
                "file": "utils.py",
                "severity": "MEDIUM",
                "priority_score": 53,
                "line": 5,
                "message": "Missing docstring",
                "code": "D103",
            },
        ],
        "summary": {
            "total_findings": 2,
            "findings_by_severity": {
                "HIGH": 1,
                "MEDIUM": 1,
            },
            "successful_agents": 8,
            "partial_agents": 0,
            "failed_agents": 0,
        },
    }

    report = generate_repository_markdown_report(result)

    assert "# CodeGuard AI - Repository Review Report" in report
    assert "main.py" in report
    assert "utils.py" in report
    assert "Possible SQL injection" in report
    assert "Missing docstring" in report
    assert "Repository Health Score: 77/100" in report
    assert "No file-level failures." in report


def test_repository_markdown_report_shows_agent_status_counts_and_partials():
    result = {
        "repo_root": "sample_project",
        "total_files": 1,
        "reviewed_files": 1,
        "failed_files": 0,
        "partial_files": 1,
        "file_results": {
            "large.py": {
                "errors": {
                    "code_reviewer": "PartialReviewError: bounded context",
                    "security_agent": "RuntimeError: unavailable",
                },
                "summary": {
                    "agent_statuses": {
                        "code_reviewer": "PARTIAL",
                        "security_agent": "FAILED",
                        "style_agent": "SUCCESS",
                        "architecture_agent": "SUCCESS",
                    },
                },
            },
        },
        "all_findings": [],
        "summary": {
            "total_findings": 0,
            "findings_by_severity": {},
            "successful_agents": 2,
            "partial_agents": 1,
            "failed_agents": 1,
        },
    }

    report = generate_repository_markdown_report(result)

    assert "Successful Agent Runs:** 2" in report
    assert "Partial Agent Runs:** 1" in report
    assert "Failed Agent Runs:** 1" in report
    assert "Files with Partial Reviews:** 1" in report
    assert "large.py | code_reviewer:** PARTIAL" in report
    assert "Static analysis processed the full source" in report
    assert "large.py | security_agent:** FAILED" in report
