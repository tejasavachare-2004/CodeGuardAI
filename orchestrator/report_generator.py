
"""
Generate readable Markdown reports from CodeGuard AI results.
"""


def calculate_health_score(severity_counts):
    """Calculate a heuristic repository health score."""

    penalties = {
        "CRITICAL": 25,
        "HIGH": 15,
        "MEDIUM": 8,
        "LOW": 3,
        "INFO": 1,
    }

    total_penalty = sum(
        severity_counts.get(severity, 0) * penalty
        for severity, penalty in penalties.items()
    )

    return max(0, 100 - total_penalty)


AGENT_NAMES = (
    "code_reviewer",
    "security_agent",
    "style_agent",
    "architecture_agent",
)
STATUS_LABELS = {
    "SUCCESS": "COMPLETED",
    "PARTIAL": "PARTIAL",
    "FAILED": "FAILED",
}
PARTIAL_REVIEW_NOTE = (
    "Static analysis processed the full source; the LLM did not review the entire "
    "file because it received bounded AST-selected context."
)


def _agent_status(agent, errors, agent_statuses):
    status = agent_statuses.get(agent)
    if status in STATUS_LABELS:
        return status

    error = errors.get(agent)
    if error is None:
        return "SUCCESS"
    if str(error).startswith("PartialReviewError:"):
        return "PARTIAL"
    return "FAILED"


def generate_markdown_report(review_result):
    """Generate a Markdown report for a single-file review."""

    summary = review_result.get("summary", {})
    findings = review_result.get("all_findings", [])
    severity_counts = summary.get("findings_by_severity", {})

    health_score = calculate_health_score(severity_counts)

    lines = []

    lines.append("# CodeGuard AI - Code Review Report")
    lines.append("")

    lines.append("## Repository Information")
    lines.append("")
    lines.append(f"- **Repository:** {review_result.get('repo_root', 'Unknown')}")
    lines.append(f"- **File:** {review_result.get('filename', 'Unknown')}")
    lines.append("")

    lines.append("## Review Summary")
    lines.append("")
    lines.append(f"**Repository Health Score: {health_score}/100**")
    lines.append("")
    lines.append(
        f"- **Total Findings:** {summary.get('total_findings', len(findings))}"
    )
    lines.append(
        f"- **Successful Agents:** {summary.get('successful_agents', 0)}"
    )
    lines.append(
        f"- **Partial Agents:** {summary.get('partial_agents', 0)}"
    )
    lines.append(
        f"- **Failed Agents:** {summary.get('failed_agents', 0)}"
    )
    lines.append("")

    lines.append("## Findings by Severity")
    lines.append("")

    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        lines.append(
            f"- **{severity}:** {severity_counts.get(severity, 0)}"
        )

    lines.append("")
    lines.append("## Detailed Findings")
    lines.append("")

    if not findings:
        lines.append("No issues found. Great work!")
    else:
        for index, finding in enumerate(findings, start=1):
            lines.append(
                f"### {index}. {finding.get('message', 'Unknown issue')}"
            )
            lines.append("")
            lines.append(f"- **Severity:** {finding.get('severity', 'UNKNOWN')}")
            lines.append(
                f"- **Priority Score:** {finding.get('priority_score', 'N/A')}"
            )
            lines.append(f"- **File:** {finding.get('file', 'Unknown')}")
            lines.append(f"- **Line:** {finding.get('line', 'Unknown')}")
            lines.append(f"- **Code:** {finding.get('code', 'N/A')}")

            reported_by = finding.get("reported_by", [])
            if isinstance(reported_by, list):
                reported_by = ", ".join(reported_by)

            lines.append(f"- **Reported By:** {reported_by}")

            if finding.get("confidence") is not None:
                lines.append(f"- **Confidence:** {finding['confidence']}")

            if finding.get("evidence"):
                lines.append(f"- **Evidence:** {finding['evidence']}")

            lines.append("")

    lines.append("## Agent Status")
    lines.append("")

    errors = review_result.get("errors", {})

    agent_reports = {
        "code_reviewer": review_result.get("code_report", {}),
        "security_agent": review_result.get("security_report", {}),
        "style_agent": review_result.get("style_report", {}),
        "architecture_agent": review_result.get("architecture_report", {}),
    }
    agent_statuses = summary.get("agent_statuses", {})

    for agent, report in agent_reports.items():
        agent_status = _agent_status(agent, errors, agent_statuses)
        status = STATUS_LABELS[agent_status]
        duration = report.get("duration_seconds")

        if duration is not None:
            lines.append(
                f"- **{agent}:** {status} | **Duration:** {duration:.2f}s"
            )
        else:
            lines.append(f"- **{agent}:** {status}")
        if agent_status == "PARTIAL":
            lines.append(f"  - {PARTIAL_REVIEW_NOTE}")
        elif agent_status == "FAILED" and agent in errors:
            lines.append(f"  - Error: {errors[agent]}")

    lines.extend(["", "---", "*Generated by CodeGuard AI*"])

    return "\n".join(lines)


def generate_repository_markdown_report(repository_result):
    """Generate a Markdown report for a complete repository review."""

    summary = repository_result.get("summary", {})
    findings = repository_result.get("all_findings", [])
    file_results = repository_result.get("file_results", {})

    severity_counts = summary.get("findings_by_severity", {})
    health_score = calculate_health_score(severity_counts)

    lines = []

    lines.append("# CodeGuard AI - Repository Review Report")
    lines.append("")

    lines.append("## Repository Information")
    lines.append("")
    lines.append(
        f"- **Repository:** {repository_result.get('repo_root', 'Unknown')}"
    )
    lines.append(
        f"- **Total Python Files:** {repository_result.get('total_files', 0)}"
    )
    lines.append(
        f"- **Reviewed Files:** {repository_result.get('reviewed_files', 0)}"
    )
    lines.append(
        f"- **Failed Files:** {repository_result.get('failed_files', 0)}"
    )
    lines.append(
        f"- **Files with Partial Reviews:** {repository_result.get('partial_files', 0)}"
    )
    lines.append("")

    lines.append("## Repository Summary")
    lines.append("")
    lines.append(f"**Repository Health Score: {health_score}/100**")
    lines.append("")
    lines.append(
        f"- **Total Findings:** {summary.get('total_findings', len(findings))}"
    )
    lines.append(
        f"- **Successful Agent Runs:** {summary.get('successful_agents', 0)}"
    )
    lines.append(
        f"- **Partial Agent Runs:** {summary.get('partial_agents', 0)}"
    )
    lines.append(
        f"- **Failed Agent Runs:** {summary.get('failed_agents', 0)}"
    )
    lines.append("")

    lines.append("## Findings by Severity")
    lines.append("")

    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        lines.append(
            f"- **{severity}:** {severity_counts.get(severity, 0)}"
        )

    lines.append("")

    lines.append("## Findings by File")
    lines.append("")

    if not findings:
        lines.append("No issues found. Great work!")
    else:
        grouped_findings = {}

        for finding in findings:
            filename = finding.get("file", "Unknown")
            grouped_findings.setdefault(filename, []).append(finding)

        for filename, file_findings in sorted(grouped_findings.items()):
            lines.append(f"### {filename}")
            lines.append("")

            for finding in file_findings:
                lines.append(
                    f"- **{finding.get('severity', 'UNKNOWN')}** "
                    f"(Priority: {finding.get('priority_score', 'N/A')}) "
                    f"- Line {finding.get('line', 'Unknown')}: "
                    f"{finding.get('message', 'Unknown issue')} "
                    f"[{finding.get('code', 'N/A')}]"
                )

            lines.append("")

    lines.append("## Agent Status by File")
    lines.append("")

    for filename, result in sorted(file_results.items()):
        file_summary = result.get("summary", {})
        agent_statuses = file_summary.get("agent_statuses", {})
        errors = result.get("errors", {})
        if "error" in result and not agent_statuses:
            agent_statuses = {agent: "FAILED" for agent in AGENT_NAMES}

        for agent in AGENT_NAMES:
            agent_status = _agent_status(agent, errors, agent_statuses)
            detail = ""
            if agent_status == "PARTIAL":
                detail = f" | {PARTIAL_REVIEW_NOTE}"
            elif agent_status == "FAILED":
                detail = f" | {errors.get(agent, result.get('error', 'Review failed.'))}"
            lines.append(
                f"- **{filename} | {agent}:** {STATUS_LABELS[agent_status]}{detail}"
            )

    lines.append("")
    lines.append("## Files That Failed")
    lines.append("")

    failed_files = [
        (filename, result)
        for filename, result in file_results.items()
        if "error" in result
    ]

    if not failed_files:
        lines.append("No file-level failures.")
    else:
        for filename, result in failed_files:
            lines.append(f"- **{filename}:** {result['error']}")

    lines.extend(["", "---", "*Generated by CodeGuard AI*"])

    return "\n".join(lines)
