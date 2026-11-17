
import json
from pathlib import Path

from orchestrator.graph import run_repository_review
from orchestrator.report_generator import generate_repository_markdown_report


if __name__ == "__main__":

    print("\nStarting CodeGuard AI Repository Review...\n")

    result = run_repository_review(
        repo_root=".",
        model="qwen2.5-coder:3b",
        base_url="http://localhost:11434",
        use_llm=False,
    )

    print("\nREPOSITORY SUMMARY:")
    print(json.dumps(result["summary"], indent=2))

    print("\nFILES:")
    print("Total:", result["total_files"])
    print("Reviewed:", result["reviewed_files"])
    print("Failed:", result["failed_files"])

    # Save JSON results
    Path("repository_review.json").write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8",
    )

    # Generate Markdown report
    markdown_report = generate_repository_markdown_report(result)

    Path("repository_review.md").write_text(
        markdown_report,
        encoding="utf-8",
    )

    print("\nRepository review completed!")
    print("JSON saved: repository_review.json")
    print("Markdown saved: repository_review.md")
