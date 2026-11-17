"""CodeGuard AI - LangGraph orchestrator (version 1: sequential).

Flow:
    START -> code_reviewer -> security_agent -> style_agent
          -> architecture_agent -> combine_findings -> END

Run from the project root (the folder that contains `agents/`):
    python -m orchestrator.graph
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.architecture_agent import ArchitectureAgent
from agents.code_reviewer import CodeReviewer
from agents.security_agent import SecurityAgent
from agents.style_agent import StyleAgent
from orchestrator.deduplicator import deduplicate_findings
from orchestrator.ranker import rank_findings
from orchestrator.report_generator import generate_markdown_report

AGENT_NAMES = ["code_reviewer", "security_agent", "style_agent", "architecture_agent"]


# ---------------------------------------------------------------------------
# Step A: shared state
# ---------------------------------------------------------------------------
class ReviewState(TypedDict, total=False):
    """Everything the graph passes between nodes. Nodes fill it in step by step."""

    repo_root: str            # absolute path of the repository
    file_path: str            # absolute path of the file being reviewed
    filename: str             # path relative to repo_root, with forward slashes
    code: str                 # source code of the file
    code_report: dict         # Code Reviewer report (model_dump)
    security_report: dict     # Security Agent report (model_dump)
    style_findings: list      # Style Agent findings (list of dicts)
    style_duration_seconds: float  # Style Agent execution time
    architecture_report: dict # Architecture Agent report (model_dump)
    all_findings: list        # every finding from every agent
    errors: dict              # agent name -> error message
    summary: dict             # counts produced by combine_findings


def _agent_statuses(errors: dict) -> dict[str, str]:
    """Classify every configured agent as successful, partial, or failed."""
    statuses = {}
    for agent in AGENT_NAMES:
        message = errors.get(agent)
        if message is None:
            statuses[agent] = "SUCCESS"
        elif str(message).startswith("PartialReviewError:"):
            statuses[agent] = "PARTIAL"
        else:
            statuses[agent] = "FAILED"
    return statuses


def _status_counts(statuses: dict[str, str]) -> dict[str, int]:
    return {
        "successful_agents": sum(status == "SUCCESS" for status in statuses.values()),
        "partial_agents": sum(status == "PARTIAL" for status in statuses.values()),
        "failed_agents": sum(status == "FAILED" for status in statuses.values()),
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _add_error(state: ReviewState, agent: str, exc: Exception) -> dict:
    """Return a new errors dict that includes the failure of `agent`."""
    errors = dict(state.get("errors", {}))
    errors[agent] = f"{type(exc).__name__}: {exc}"
    return errors


def _partial_context_error(state: ReviewState, agent: str) -> dict:
    """Mark a bounded LLM review as partial without dropping its findings."""
    errors = dict(state.get("errors", {}))
    errors[agent] = (
        "PartialReviewError: LLM analysis used bounded AST-selected source context; "
        "the full file was not reviewed by the LLM."
    )
    return errors


def _to_dict(report: Any) -> Any:
    """Pydantic report -> plain dict. mode="json" turns enums/paths into plain values."""
    if report is None:
        return None
    if hasattr(report, "model_dump"):
        return report.model_dump(mode="json")
    return report


def _json_safe(value: Any) -> Any:
    """Convert finding values to JSON-compatible values without dropping fields."""
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_finding(reported_by: str, finding: Any) -> dict:
    """Preserve agent fields and add a common, JSON-safe finding structure."""
    original = finding if isinstance(finding, dict) else {"raw": finding}
    original = _json_safe(original)

    confidence = original.get("confidence")
    confidence_scales = {
        "code_reviewer": "0-100",
        "security_agent": "0-1",
        "architecture_agent": "0-1",
    }

    category = original.get("category")
    if category is None and reported_by == "style_agent":
        category = "style"

    code = original.get("code")
    if code is None:
        code = original.get("cwe")

    message = original.get("message")
    if message is None:
        message = original.get("description")
    if message is None:
        message = original.get("title")

    identity = {
        "reported_by": reported_by,
        "file": original.get("file"),
        "line": original.get("line"),
        "code": code,
        "title": original.get("title"),
        "message": message,
    }
    identity_json = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    finding_id = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()

    normalized = {
        **original,
        "id": finding_id,
        "reported_by": reported_by,
        "category": category,
        "subcategory": original.get("subcategory"),
        "code": code,
        "message": message,
        "file": original.get("file"),
        "line": original.get("line"),
        "column": original.get("column"),
        "severity": original.get("severity"),
        "confidence": confidence,
        "confidence_scale": confidence_scales.get(reported_by)
        if confidence is not None
        else None,
    }
    return _json_safe(normalized)


def _findings_from_report(
    agent: str, report: Any, warnings: list[str]
) -> list[dict]:
    """Pull the 'findings' list out of a report dict.

    ASSUMPTION: Code Reviewer, Security and Architecture reports keep their
    findings in a list under the key "findings". If that key is missing we do
    NOT silently drop anything - we add a warning to the summary instead.
    """
    if not report:
        return []
    findings = report.get("findings") if isinstance(report, dict) else None
    if not isinstance(findings, list):
        keys = list(report.keys()) if isinstance(report, dict) else type(report).__name__
        warnings.append(f"{agent}: no 'findings' list in report. Keys found: {keys}")
        return []
    result = []
    for index, item in enumerate(findings):
        if not isinstance(item, dict):
            warnings.append(
                f"{agent}: finding {index} is not an object; preserved as raw data."
            )
        try:
            result.append(_normalize_finding(agent, item))
        except (AttributeError, RecursionError, TypeError, ValueError) as exc:
            warnings.append(
                f"{agent}: finding {index} could not be normalized "
                f"({type(exc).__name__}: {exc}); preserved as raw data."
            )
            result.append(_normalize_finding(agent, {"raw": repr(item)}))
    return result


# ---------------------------------------------------------------------------
# Step E (part 1): build the graph
# ---------------------------------------------------------------------------
def build_review_graph(
    repo_root: str = ".",
    model: str = "qwen2.5-coder:3b",
    base_url: str = "http://localhost:11434",
    use_llm: bool = False,
):
    """Create the agents, wire up the StateGraph and return the compiled graph."""
    repo_root = str(Path(repo_root).resolve())

    # Agents are created once and reused by the nodes below.
    # Code Reviewer has no use_llm switch: it always uses its LLM.
    code_reviewer = CodeReviewer(model=model, base_url=base_url)
    security_agent = SecurityAgent(model=model, base_url=base_url, use_llm=use_llm)
    style_agent = StyleAgent()
    architecture_agent = ArchitectureAgent(
        repo_root=repo_root, model=model, base_url=base_url, use_llm=use_llm
    )

    # ----- Step B: one node per agent. Each returns ONLY the fields it changes. -----
    def run_code_reviewer(state: ReviewState) -> dict:
        try:
            report = code_reviewer.review(state["code"], filename=state["filename"])
            result = {"code_report": _to_dict(report)}
            if getattr(report, "llm_context_complete", None) is False:
                result["errors"] = _partial_context_error(state, "code_reviewer")
            return result
        except Exception as exc:  # noqa: BLE001 - isolate agent failures
            return {"errors": _add_error(state, "code_reviewer", exc)}

    def run_security_agent(state: ReviewState) -> dict:
        try:
            report = security_agent.review(state["code"], filename=state["filename"])
            result = {"security_report": _to_dict(report)}
            if getattr(report, "llm_context_complete", None) is False:
                result["errors"] = _partial_context_error(state, "security_agent")
            return result
        except Exception as exc:  # noqa: BLE001 - isolate agent failures
            return {"errors": _add_error(state, "security_agent", exc)}

    def run_style_agent(state: ReviewState) -> dict:
        try:
            findings = style_agent.analyze(state["file_path"])

            if not isinstance(findings, list):
                raise TypeError(
                    f"StyleAgent.analyze() must return a list, got {type(findings).__name__}"
                )

            return {
                "style_findings": findings,
                "style_duration_seconds": style_agent.duration_seconds,
            }

        except Exception as exc:  # noqa: BLE001 - isolate agent failures
            return {"errors": _add_error(state, "style_agent", exc)}

    def run_architecture_agent(state: ReviewState) -> dict:
        try:
            report = architecture_agent.review(
                file_path=state["file_path"],
                code=state["code"],
            )
            result = {"architecture_report": _to_dict(report)}
            if getattr(report, "llm_context_complete", None) is False:
                result["errors"] = _partial_context_error(state, "architecture_agent")
            return result
        except Exception as exc:  # noqa: BLE001 - isolate agent failures
            return {"errors": _add_error(state, "architecture_agent", exc)}

    # ----- Step C: combine everything -----
    def combine_findings(state: ReviewState) -> dict:
        warnings: list[str] = []
        all_findings: list[dict] = []

        all_findings += _findings_from_report(
            "code_reviewer", state.get("code_report"), warnings
        )
        all_findings += _findings_from_report(
            "security_agent", state.get("security_report"), warnings
        )

        style_findings = state.get("style_findings", [])

        if not isinstance(style_findings, list):
            warnings.append("style_agent: findings value is not a list.")
            style_findings = []

        for index, item in enumerate(style_findings):
            if not isinstance(item, dict):
                warnings.append(
                    f"style_agent: finding {index} is not an object; preserved as raw data."
                )

            try:
                all_findings.append(_normalize_finding("style_agent", item))
            except (AttributeError, RecursionError, TypeError, ValueError) as exc:
                warnings.append(
                    f"style_agent: finding {index} could not be normalized "
                    f"({type(exc).__name__}: {exc}); preserved as raw data."
                )
                all_findings.append(
                    _normalize_finding("style_agent", {"raw": repr(item)})
                )

        all_findings += _findings_from_report(
            "architecture_agent", state.get("architecture_report"), warnings
        )

        # Remove duplicate findings after normalization.
        all_findings = deduplicate_findings(all_findings)
        all_findings = rank_findings(all_findings)

        # Calculate severity counts using deduplicated findings.
        by_severity: dict[str, int] = {}

        for finding in all_findings:
            severity = str(finding.get("severity", "UNKNOWN")).upper()
            by_severity[severity] = by_severity.get(severity, 0) + 1

        agent_statuses = _agent_statuses(state.get("errors", {}))
        summary = {
            "total_findings": len(all_findings),
            "findings_by_severity": by_severity,
            "agent_statuses": agent_statuses,
            **_status_counts(agent_statuses),
            "warnings": warnings,
        }
        return {
            "all_findings": all_findings,
            "summary": summary,
        }

    # ----- Step D: nodes and edges -----
    graph = StateGraph(ReviewState)
    graph.add_node("code_reviewer", run_code_reviewer)
    graph.add_node("security_agent", run_security_agent)
    graph.add_node("style_agent", run_style_agent)
    graph.add_node("architecture_agent", run_architecture_agent)
    graph.add_node("combine_findings", combine_findings)

    graph.add_edge(START, "code_reviewer")
    graph.add_edge("code_reviewer", "security_agent")
    graph.add_edge("security_agent", "style_agent")
    graph.add_edge("style_agent", "architecture_agent")
    graph.add_edge("architecture_agent", "combine_findings")
    graph.add_edge("combine_findings", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Step E (part 2): public entry point
# ---------------------------------------------------------------------------
def run_review(
    repo_root: str,
    file_path: str,
    model: str = "qwen2.5-coder:3b",
    base_url: str = "http://localhost:11434",
    use_llm: bool = False,
) -> ReviewState:
    """Validate inputs, read the file, run the graph and return the final state."""
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository root not found: {root}")

    target = Path(file_path)
    if not target.is_absolute():
        target = root / target          # relative paths are relative to repo_root
    target = target.resolve()

    if not target.is_file():
        raise FileNotFoundError(f"File not found: {target}")
    if target.suffix != ".py":
        raise ValueError(f"Only .py files are supported: {target}")
    try:
        relative = target.relative_to(root)
    except ValueError:
        raise ValueError(f"{target} is not inside repository root {root}") from None

    initial_state: ReviewState = {
        "repo_root": str(root),
        "file_path": str(target),
        "filename": relative.as_posix(),   # forward slashes, also on Windows
        "code": target.read_text(encoding="utf-8"),
        "errors": {},
    }

    graph = build_review_graph(
        repo_root=str(root), model=model, base_url=base_url, use_llm=use_llm
    )
    return graph.invoke(initial_state)


def run_repository_review(
    repo_root: str,
    model: str = "qwen2.5-coder:3b",
    base_url: str = "http://localhost:11434",
    use_llm: bool = False,
) -> dict:
    """Review every Python file in a repository."""

    root = Path(repo_root).resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Repository root not found: {root}")

    ignored_directories = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "dist",
        "build",
    }

    python_files = []
    for path in sorted(root.rglob("*.py")):
        relative_parts = path.relative_to(root).parts
        if any(part in ignored_directories for part in relative_parts):
            continue
        try:
            with path.open(encoding="utf-8", errors="ignore") as source_file:
                if not any(line.strip() for line in source_file):
                    continue
        except OSError:
            # Keep unreadable files in the result so their failure is reported per file.
            pass
        python_files.append(path)

    if not python_files:
        raise FileNotFoundError("No non-empty Python files found in repository.")

    repository_findings = []
    file_results = {}
    total_successful_agents = 0
    total_partial_agents = 0
    total_failed_agents = 0

    for file_path in python_files:
        relative_path = file_path.relative_to(root).as_posix()

        print(f"\nReviewing: {relative_path}")

        try:
            result = run_review(
                repo_root=str(root),
                file_path=str(file_path),
                model=model,
                base_url=base_url,
                use_llm=use_llm,
            )

            repository_findings.extend(result.get("all_findings", []))
            file_results[relative_path] = result

            total_successful_agents += result["summary"]["successful_agents"]
            total_partial_agents += result["summary"].get("partial_agents", 0)
            total_failed_agents += result["summary"]["failed_agents"]

        except Exception as exc:  # noqa: BLE001 - record file-specific failures
            print(f"Failed to review {relative_path}: {exc}")
            file_errors = {
                agent: f"FileReviewError: {type(exc).__name__}: {exc}"
                for agent in AGENT_NAMES
            }
            statuses = {agent: "FAILED" for agent in AGENT_NAMES}
            file_results[relative_path] = {
                "error": str(exc),
                "errors": file_errors,
                "summary": {
                    "agent_statuses": statuses,
                    **_status_counts(statuses),
                },
            }
            total_failed_agents += len(AGENT_NAMES)

    repository_findings = deduplicate_findings(repository_findings)
    repository_findings = rank_findings(repository_findings)

    severity_counts = {}

    for finding in repository_findings:
        severity = str(finding.get("severity", "UNKNOWN")).upper()
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    partial_files = sum(
        1 for result in file_results.values()
        if result.get("summary", {}).get("partial_agents", 0) > 0
    )

    return {
        "repo_root": str(root),
        "total_files": len(python_files),
        "reviewed_files": sum(
            1 for result in file_results.values() if "error" not in result
        ),
        "failed_files": sum(
            1 for result in file_results.values() if "error" in result
        ),
        "file_results": file_results,
        "all_findings": repository_findings,
        "summary": {
            "total_findings": len(repository_findings),
            "findings_by_severity": severity_counts,
            "successful_agents": total_successful_agents,
            "partial_agents": total_partial_agents,
            "failed_agents": total_failed_agents,
            "agent_runs": len(python_files) * len(AGENT_NAMES),
        },
        "partial_files": partial_files,
    }


# ---------------------------------------------------------------------------
# Quick manual test:  python -m orchestrator.graph
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_review(
        repo_root=".",
        file_path="agents/style_agent.py",
        model="qwen2.5-coder:3b",
        base_url="http://localhost:11434",
        use_llm=False,
    )

    print("SUMMARY:", json.dumps(result["summary"], indent=2))
    print("ERRORS:", json.dumps(result["errors"], indent=2))
    print("REPORT KEYS:")

    for key in ("code_report", "security_report", "architecture_report"):
        report = result.get(key)
        print(f"  {key}:", list(report.keys()) if isinstance(report, dict) else report)

    if result.get("all_findings"):
        print("FIRST FINDING:", json.dumps(result["all_findings"][0], indent=2, default=str))

    # Add Style Agent execution time to the report
    result["style_report"] = {
        "duration_seconds": result.get("style_duration_seconds")
    }

    # Save complete JSON output
    Path("review_output.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print("Full result saved to review_output.json")

    # Generate and save Markdown report
    markdown_report = generate_markdown_report(result)

    Path("review_report.md").write_text(
        markdown_report, encoding="utf-8"
    )
    print("Markdown report saved to review_report.md")

# Save complete JSON output
    # Save complete JSON output
    Path("review_output.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print("Full result saved to review_output.json")

    # Generate and save Markdown report
    markdown_report = generate_markdown_report(result)

    Path("review_report.md").write_text(
        markdown_report, encoding="utf-8"
    )
    print("Markdown report saved to review_report.md")