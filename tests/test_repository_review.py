
from orchestrator.graph import run_repository_review


def test_repository_review_returns_expected_structure(tmp_path):
    # Create two sample Python files
    (tmp_path / "file1.py").write_text(
        "def hello():\n    print('Hello')\n",
        encoding="utf-8",
    )

    (tmp_path / "file2.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (tmp_path / "empty.py").write_text("", encoding="utf-8")
    (tmp_path / "whitespace.py").write_text(" \n\t\n", encoding="utf-8")

    result = run_repository_review(
        repo_root=str(tmp_path),
        use_llm=False,
    )

    assert result["total_files"] == 2
    assert result["reviewed_files"] == 2
    assert result["failed_files"] == 0
    assert "summary" in result
    assert "all_findings" in result
    assert len(result["file_results"]) == 2
