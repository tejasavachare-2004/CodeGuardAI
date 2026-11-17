from __future__ import annotations

import json
from pathlib import Path

import pytest

import orchestrator.graph as review_graph
from agents.architecture_agent import (
    ArchitectureAgent,
    ArchLLMFinding,
    ArchLLMReview,
)
from agents.code_reviewer import (
    MAX_CODE_CHARS,
    CodeReviewer,
    LLMFinding,
    LLMReview,
)
from agents.code_reviewer import (
    Category as CodeCategory,
)
from agents.code_reviewer import (
    Severity as CodeSeverity,
)
from agents.security_agent import (
    SecurityAgent,
    SecurityLLMFinding,
    SecurityLLMReview,
)
from agents.security_agent import (
    Severity as SecuritySeverity,
)
from orchestrator.report_generator import generate_repository_markdown_report


class CapturingChain:
    def __init__(self, result):
        self.result = result
        self.inputs = None

    def invoke(self, inputs):
        self.inputs = inputs
        return self.result


def make_large_source(target_chars: int = 60_000) -> str:
    lines = ["def risky():", "    eval(input())", ""]
    index = 0
    while len("\n".join(lines)) < target_chars:
        lines.extend([
            f"def helper_{index}():",
            f"    return {index}  # {'x' * 90}",
            "",
        ])
        index += 1
    return "\n".join(lines)


def test_slightly_oversized_file_gets_bounded_ast_context():
    lines = ["eval(input())"]
    index = 0
    while len("\n".join(lines)) <= MAX_CODE_CHARS:
        lines.append(f"value_{index} = {'x' * 60!r}")
        index += 1
    code = "\n".join(lines)
    assert MAX_CODE_CHARS < len(code) < MAX_CODE_CHARS + 1_000

    chain = CapturingChain(LLMReview())
    reviewer = CodeReviewer()
    reviewer._chain = chain

    report = reviewer.review(code)

    assert chain.inputs is not None
    assert len(chain.inputs["numbered_code"]) <= MAX_CODE_CHARS
    assert report.llm_context_complete is False
    assert report.findings[0].line == 1


def test_small_file_still_sends_complete_context():
    code = "def add(a, b):\n    return a + b\n"
    chain = CapturingChain(LLMReview())
    reviewer = CodeReviewer()
    reviewer._chain = chain

    report = reviewer.review(code)

    assert chain.inputs["numbered_code"] == "   1| def add(a, b):\n   2|     return a + b"
    assert report.llm_context_complete is True


def test_large_file_static_scans_keep_original_line_numbers(tmp_path: Path):
    security_code = "\n".join(f"value_{i} = {i}" for i in range(6_000))
    expected_security_line = 6_001
    security_code += "\neval(input())\n"

    security_report = SecurityAgent(use_llm=False).review(security_code, "large.py")
    security_finding = next(
        finding for finding in security_report.findings
        if finding.title == "Dynamic Code Execution (eval/exec)"
    )
    assert security_finding.line == expected_security_line
    assert security_report.llm_context_complete is None

    architecture_code = "".join(
        f"def function_{i}():\n    value = '{'x' * 2_000}'\n    return value\n\n"
        for i in range(30)
    )
    architecture_code += "from missing_module import *\n"
    assert len(architecture_code) > 50_000
    target = tmp_path / "large.py"
    target.write_text(architecture_code, encoding="utf-8")

    architecture_report = ArchitectureAgent(
        str(tmp_path), use_llm=False
    ).review("large.py")
    wildcard = next(
        finding for finding in architecture_report.findings
        if finding.title == "Wildcard Import"
    )
    assert wildcard.line == len(architecture_code.splitlines())
    assert architecture_report.llm_context_complete is None


def test_all_llm_agents_use_bounded_context_and_original_lines(tmp_path: Path):
    code = make_large_source()
    assert len(code) > 50_000

    code_chain = CapturingChain(LLMReview(findings=[
        LLMFinding(
            title="Potential bug",
            severity=CodeSeverity.LOW,
            category=CodeCategory.BUG,
            line=2,
            evidence="eval(input())",
            description="d",
            recommendation="r",
            confidence=80,
        )
    ]))
    reviewer = CodeReviewer()
    reviewer._chain = code_chain
    code_report = reviewer.review(code)

    security_chain = CapturingChain(SecurityLLMReview(findings=[
        SecurityLLMFinding(
            title="Unsafe execution",
            severity=SecuritySeverity.HIGH,
            line=2,
            evidence="eval(input())",
            cwe="CWE-22",
            owasp="A03:2021-Injection",
            description="d",
            attack_scenario="a",
            recommendation="r",
            confidence=0.9,
        )
    ]))
    security_agent = SecurityAgent(use_llm=True)
    security_agent._chain = security_chain
    security_report = security_agent.review(code, "large.py")

    (tmp_path / "large.py").write_text(code, encoding="utf-8")
    architecture_chain = CapturingChain(ArchLLMReview(findings=[
        ArchLLMFinding(
            title="Design concern",
            severity=CodeSeverity.MEDIUM,
            category="design",
            line=2,
            evidence="eval(input())",
            description="d",
            recommendation="r",
            confidence=0.9,
        )
    ]))
    architecture_agent = ArchitectureAgent(str(tmp_path), use_llm=True)
    architecture_agent._chain = architecture_chain
    architecture_report = architecture_agent.review("large.py")

    for chain, limit in (
        (code_chain, MAX_CODE_CHARS),
        (security_chain, MAX_CODE_CHARS),
        (architecture_chain, 12_000),
    ):
        context = chain.inputs["numbered_code"]
        assert len(context) <= limit
        assert "omitted" in context
        assert "   2|     eval(input())" in context

    assert code_report.llm_context_complete is False
    assert security_report.llm_context_complete is False
    assert architecture_report.llm_context_complete is False
    assert next(f for f in code_report.findings if f.source.value == "llm").line == 2
    assert next(f for f in security_report.findings if f.source.value == "llm").line == 2
    assert next(f for f in architecture_report.findings if f.source.value == "llm").line == 2


def test_security_and_architecture_static_modes_accept_large_files(tmp_path: Path):
    code = make_large_source(55_000)
    (tmp_path / "large.py").write_text(code, encoding="utf-8")

    security = SecurityAgent(use_llm=False).review(code, "large.py")
    architecture = ArchitectureAgent(str(tmp_path), use_llm=False).review("large.py")

    assert security.syntax_valid
    assert architecture.syntax_valid


def test_unrepresentable_ast_statement_fails_explicitly():
    code = "value = '" + ("x" * MAX_CODE_CHARS) + "'"
    reviewer = CodeReviewer()
    reviewer._chain = CapturingChain(LLMReview())

    with pytest.raises(ValueError, match="no complete top-level AST statement"):
        reviewer.review(code)


def test_repository_result_marks_partial_llm_review_as_agent_error(tmp_path, monkeypatch):
    class FakeReport:
        llm_context_complete = False

        def model_dump(self, **_kwargs):
            return {"findings": []}

    class FakeCodeReviewer:
        def __init__(self, **_kwargs):
            pass

        def review(self, *_args, **_kwargs):
            return FakeReport()

    class FakeSecurityAgent:
        def __init__(self, **_kwargs):
            pass

        def review(self, *_args, **_kwargs):
            return {"findings": []}

    class FakeStyleAgent:
        duration_seconds = 0.0

        def analyze(self, _path):
            return []

    class FakeArchitectureAgent:
        def __init__(self, **_kwargs):
            pass

        def review(self, **_kwargs):
            return {"findings": []}

    monkeypatch.setattr(review_graph, "CodeReviewer", FakeCodeReviewer)
    monkeypatch.setattr(review_graph, "SecurityAgent", FakeSecurityAgent)
    monkeypatch.setattr(review_graph, "StyleAgent", FakeStyleAgent)
    monkeypatch.setattr(review_graph, "ArchitectureAgent", FakeArchitectureAgent)
    (tmp_path / "sample.py").write_text("value = 1\n", encoding="utf-8")

    result = review_graph.run_review(str(tmp_path), "sample.py", use_llm=False)

    assert "bounded AST-selected source context" in result["errors"]["code_reviewer"]
    assert result["code_report"]["findings"] == []
    assert result["summary"]["agent_statuses"]["code_reviewer"] == "PARTIAL"
    assert result["summary"]["successful_agents"] == 3
    assert result["summary"]["partial_agents"] == 1
    assert result["summary"]["failed_agents"] == 0


def test_actual_agent_error_is_failed_while_other_agents_continue(tmp_path, monkeypatch):
    class FakeCodeReviewer:
        def __init__(self, **_kwargs):
            pass

        def review(self, *_args, **_kwargs):
            return {"findings": []}

    class FakeSecurityAgent:
        def __init__(self, **_kwargs):
            pass

        def review(self, *_args, **_kwargs):
            raise RuntimeError("security scanner failed")

    class FakeStyleAgent:
        duration_seconds = 0.0

        def analyze(self, _path):
            return []

    class FakeArchitectureAgent:
        def __init__(self, **_kwargs):
            pass

        def review(self, **_kwargs):
            return {"findings": []}

    monkeypatch.setattr(review_graph, "CodeReviewer", FakeCodeReviewer)
    monkeypatch.setattr(review_graph, "SecurityAgent", FakeSecurityAgent)
    monkeypatch.setattr(review_graph, "StyleAgent", FakeStyleAgent)
    monkeypatch.setattr(review_graph, "ArchitectureAgent", FakeArchitectureAgent)
    (tmp_path / "sample.py").write_text("value = 1\n", encoding="utf-8")

    result = review_graph.run_review(str(tmp_path), "sample.py", use_llm=False)

    assert result["summary"]["agent_statuses"]["security_agent"] == "FAILED"
    assert result["summary"]["successful_agents"] == 3
    assert result["summary"]["partial_agents"] == 0
    assert result["summary"]["failed_agents"] == 1
    assert "security scanner failed" in result["errors"]["security_agent"]


def test_repository_aggregates_agent_statuses_and_findings(tmp_path, monkeypatch):
    agent_names = review_graph.AGENT_NAMES
    for index in range(19):
        (tmp_path / f"file_{index:02}.py").write_text(
            f"value = {index}\n", encoding="utf-8"
        )

    def fake_run_review(repo_root, file_path, **_kwargs):
        relative = Path(file_path).relative_to(repo_root).as_posix()
        index = int(Path(relative).stem.split("_")[1])
        errors = {}
        if index < 4:
            errors[agent_names[index]] = "PartialReviewError: bounded LLM context"
        statuses = review_graph._agent_statuses(errors)
        summary = {
            "agent_statuses": statuses,
            **review_graph._status_counts(statuses),
        }
        findings = []
        if index == 0:
            findings = [
                {
                    "reported_by": "security_agent",
                    "file": relative,
                    "line": 1,
                    "title": "Potential command injection vulnerability",
                    "message": "Repeated issue",
                    "severity": "HIGH",
                    "confidence": 0.9,
                },
                {
                    "reported_by": "code_reviewer",
                    "file": relative,
                    "line": 1,
                    "title": "Potential command injection vulnerability",
                    "message": "Repeated issue",
                    "severity": "HIGH",
                    "confidence": 85,
                },
            ]
        elif index == 1:
            findings = [{
                "reported_by": "security_agent",
                "file": relative,
                "line": 1,
                "message": "Critical issue",
                "severity": "CRITICAL",
                "confidence": 0.9,
            }]
        return {
            "all_findings": findings,
            "summary": summary,
            "errors": errors,
        }

    monkeypatch.setattr(review_graph, "run_review", fake_run_review)

    result = review_graph.run_repository_review(str(tmp_path), use_llm=False)

    assert result["total_files"] == 19
    assert result["reviewed_files"] == 19
    assert result["failed_files"] == 0
    assert result["partial_files"] == 4
    assert result["summary"]["agent_runs"] == 76
    assert result["summary"]["successful_agents"] == 72
    assert result["summary"]["partial_agents"] == 4
    assert result["summary"]["failed_agents"] == 0
    assert len(result["all_findings"]) == 2
    assert result["all_findings"][0]["severity"] == "CRITICAL"

    encoded_result = json.loads(json.dumps(result))
    assert encoded_result["summary"]["successful_agents"] == 72
    assert encoded_result["summary"]["partial_agents"] == 4
    assert encoded_result["summary"]["failed_agents"] == 0
    markdown = generate_repository_markdown_report(encoded_result)
    assert "Successful Agent Runs:** 72" in markdown
    assert "Partial Agent Runs:** 4" in markdown
    assert "Failed Agent Runs:** 0" in markdown
    assert "PARTIAL" in markdown


def test_repository_scanner_handles_large_files(tmp_path, monkeypatch):
    source = "value = 1\n" * 7_000
    (tmp_path / "large.py").write_text(source, encoding="utf-8")
    assert len(source) > 50_000

    def successful_review(repo_root, file_path, **_kwargs):
        statuses = {agent: "SUCCESS" for agent in review_graph.AGENT_NAMES}
        summary = {
            "agent_statuses": statuses,
            **review_graph._status_counts(statuses),
        }
        return {"all_findings": [], "summary": summary, "errors": {}}

    monkeypatch.setattr(review_graph, "run_review", successful_review)

    result = review_graph.run_repository_review(str(tmp_path), use_llm=False)

    assert result["total_files"] == 1
    assert result["reviewed_files"] == 1
    assert result["failed_files"] == 0
    assert result["summary"]["agent_runs"] == 4
    assert result["summary"]["successful_agents"] == 4


def test_repository_file_failure_counts_agent_runs_as_failed(tmp_path, monkeypatch):
    (tmp_path / "unreadable.py").write_text("value = 1\n", encoding="utf-8")

    def failed_review(*_args, **_kwargs):
        raise OSError("file could not be loaded")

    monkeypatch.setattr(review_graph, "run_review", failed_review)

    result = review_graph.run_repository_review(str(tmp_path), use_llm=False)

    assert result["total_files"] == 1
    assert result["reviewed_files"] == 0
    assert result["failed_files"] == 1
    assert result["summary"]["agent_runs"] == 4
    assert result["summary"]["successful_agents"] == 0
    assert result["summary"]["partial_agents"] == 0
    assert result["summary"]["failed_agents"] == 4
    assert set(result["file_results"]["unreadable.py"]["summary"]["agent_statuses"].values()) == {
        "FAILED"
    }
