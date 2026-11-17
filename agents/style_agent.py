
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


class StyleAgent:

    def __init__(self):
        self.name = "Style & Code Quality Agent"
        self.duration_seconds = 0.0

    def analyze(self, file_path):

        start_time = time.perf_counter()

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        command = [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--output-format",
            "json",
            "--select",
            "I,N,D,C90",
            str(file_path)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode not in (0, 1):
            raise RuntimeError(
                f"Ruff encountered an error:\n{result.stderr}"
            )

        raw_findings = json.loads(result.stdout or "[]")

        findings = []

        for finding in raw_findings:

            location = finding.get("location", {})

            findings.append({
                "agent": "style",
                "file": str(file_path),
                "code": finding.get("code", "UNKNOWN"),
                "message": finding.get("message", "No description"),
                "line": location.get("row"),
                "column": location.get("column"),
                "severity": "MEDIUM"
            })

        self.duration_seconds = round(
            time.perf_counter() - start_time, 4
        )

        return findings


def main():

    parser = argparse.ArgumentParser(
        description="CodeGuard AI Style Agent"
    )

    parser.add_argument(
        "path",
        help="Python file to analyze"
    )

    args = parser.parse_args()

    agent = StyleAgent()
    findings = agent.analyze(args.path)

    print(f"\nRunning {agent.name}...")
    print(f"File: {args.path}")
    print(f"Duration: {agent.duration_seconds:.4f}s")

    if not findings:
        print("\nNo style issues found!")
        return

    print(f"\nFound {len(findings)} style issue(s):\n")

    for index, finding in enumerate(findings, start=1):
        print(f"[{index}] {finding['code']}")
        print(f"    Issue: {finding['message']}")
        print(
            f"    Location: Line {finding['line']}, "
            f"Column {finding['column']}"
        )
        print()


if __name__ == "__main__":
    main()
