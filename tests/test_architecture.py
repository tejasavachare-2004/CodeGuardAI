"""
Tests for the Architecture Agent. No Ollama needed: the LLM is replaced by a fake.

Each test builds a tiny fake project in a temporary folder (pytest's `tmp_path`),
then reviews one file and checks the result.

Run:  pytest -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from architecture_agent import (
    ArchCategory,
    ArchitectureAgent,
    ArchLLMFinding,
    ArchLLMReview,
    Severity,
    detect_layer,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_repo(root: Path, files: Dict[str, str]) -> Path:
    """Create a fake project: {relative path: file content}."""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def static_agent(root: Path) -> ArchitectureAgent:
    return ArchitectureAgent(str(root), use_llm=False)


class FakeChain:
    """Pretends to be the LLM chain and returns the findings we give it."""

    def __init__(self, findings):
        self._findings = findings

    def invoke(self, _inputs):
        return ArchLLMReview(findings=self._findings)


def llm_finding(line, evidence, confidence=0.9, category="design"):
    return ArchLLMFinding(
        title="Business logic in controller", severity=Severity.MEDIUM, category=category,
        line=line, evidence=evidence, description="d", recommendation="r", confidence=confidence,
    )


def titles(report) -> set:
    return {f.title for f in report.findings}


# A small, well-structured layered project used by many tests.
LAYERED = {
    "app/__init__.py": "",
    "app/models/__init__.py": "",
    "app/models/user.py": "class User:\n    pass\n",
    "app/repositories/__init__.py": "",
    "app/repositories/user_repository.py":
        "from app.models.user import User\n\ndef get_user(uid):\n    return User()\n",
    "app/services/__init__.py": "",
    "app/services/user_service.py":
        "from app.repositories.user_repository import get_user\n\ndef load(uid):\n    return get_user(uid)\n",
    "app/controllers/__init__.py": "",
    "app/controllers/user_controller.py":
        "from app.services.user_service import load\n\ndef show(uid):\n    return load(uid)\n",
}


# --------------------------------------------------------------------------- #
# Layer detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path, layer", [
    ("app/controllers/user.py", "controller"),
    ("app/services/user.py", "service"),
    ("app/repositories/user.py", "repository"),
    ("app/models/user.py", "model"),
    ("app/user_service.py", "service"),     # file name wins over folder
    ("app/main.py", None),
])
def test_detect_layer(path, layer):
    assert detect_layer(path) == layer


# --------------------------------------------------------------------------- #
# Layering rules
# --------------------------------------------------------------------------- #
def test_clean_layered_project_has_no_findings(tmp_path):
    repo = make_repo(tmp_path, LAYERED)
    for rel in ["app/controllers/user_controller.py", "app/services/user_service.py",
                "app/repositories/user_repository.py"]:
        assert static_agent(repo).review(rel).total_findings == 0, rel


def test_controller_calling_repository_directly_is_flagged(tmp_path):
    files = dict(LAYERED)
    files["app/controllers/user_controller.py"] = (
        "from app.repositories.user_repository import get_user\n\ndef show(uid):\n    return get_user(uid)\n")
    report = static_agent(make_repo(tmp_path, files)).review("app/controllers/user_controller.py")
    finding = next(f for f in report.findings if f.title == "Layer Skipped")
    assert finding.severity == Severity.MEDIUM
    assert finding.line == 1
    assert finding.related_files == ["app/repositories/user_repository.py"]


def test_repository_importing_service_is_upward_dependency(tmp_path):
    files = dict(LAYERED)
    files["app/repositories/user_repository.py"] = (
        "from app.services.user_service import load\n\ndef get_user(uid):\n    return load(uid)\n")
    report = static_agent(make_repo(tmp_path, files)).review("app/repositories/user_repository.py")
    assert "Upward Layer Dependency" in titles(report)
    assert report.findings[0].severity == Severity.HIGH


def test_relative_imports_are_resolved(tmp_path):
    files = dict(LAYERED)
    files["app/repositories/user_repository.py"] = (
        "from ..services import user_service\n\ndef get_user(uid):\n    return user_service.load(uid)\n")
    report = static_agent(make_repo(tmp_path, files)).review("app/repositories/user_repository.py")
    assert "Upward Layer Dependency" in titles(report)


def test_layer_rules_are_off_for_non_layered_projects(tmp_path):
    repo = make_repo(tmp_path, {"main.py": "import helpers\n", "helpers.py": "X = 1\n"})
    assert static_agent(repo).review("main.py").total_findings == 0


def test_database_code_in_controller_is_flagged(tmp_path):
    files = dict(LAYERED)
    files["app/controllers/user_controller.py"] = (
        "import sqlite3\n\ndef show(uid):\n    cur = sqlite3.connect('x.db').cursor()\n"
        "    cur.execute('SELECT 1')\n    return cur.fetchall()\n")
    report = static_agent(make_repo(tmp_path, files)).review("app/controllers/user_controller.py")
    flagged = [f for f in report.findings if f.title == "Data Access Inside Controller"]
    assert {f.line for f in flagged} >= {1, 5}   # the import and the .execute() call


# --------------------------------------------------------------------------- #
# Circular imports
# --------------------------------------------------------------------------- #
def test_circular_import_is_detected(tmp_path):
    repo = make_repo(tmp_path, {"a.py": "import b\n", "b.py": "import a\n"})
    report = static_agent(repo).review("a.py")
    finding = next(f for f in report.findings if f.title == "Circular Dependency")
    assert "a -> b -> a" in finding.description
    assert finding.severity == Severity.HIGH


def test_three_module_cycle_is_detected(tmp_path):
    repo = make_repo(tmp_path, {"a.py": "import b\n", "b.py": "import c\n", "c.py": "import a\n"})
    finding = next(f for f in static_agent(repo).review("a.py").findings
                   if f.title == "Circular Dependency")
    assert "a -> b -> c -> a" in finding.description


def test_import_inside_function_does_not_count_as_cycle(tmp_path):
    repo = make_repo(tmp_path, {"a.py": "import b\n", "b.py": "def f():\n    import a\n    return a\n"})
    assert "Circular Dependency" not in titles(static_agent(repo).review("a.py"))


def test_type_checking_import_does_not_count_as_cycle(tmp_path):
    repo = make_repo(tmp_path, {
        "a.py": "import b\n",
        "b.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import a\n",
    })
    assert "Circular Dependency" not in titles(static_agent(repo).review("a.py"))


# --------------------------------------------------------------------------- #
# Coupling, size, duplication
# --------------------------------------------------------------------------- #
def test_too_many_project_imports_is_tight_coupling(tmp_path):
    files = {f"m{i}.py": "X = 1\n" for i in range(9)}
    files["big.py"] = "".join(f"import m{i}\n" for i in range(9))
    report = static_agent(make_repo(tmp_path, files)).review("big.py")
    assert "Tight Coupling (too many project imports)" in titles(report)


def test_wildcard_and_private_imports(tmp_path):
    repo = make_repo(tmp_path, {
        "lib.py": "_secret = 1\nX = 2\n",
        "use.py": "from lib import *\nfrom lib import _secret\n",
    })
    assert {"Wildcard Import", "Import of a Private Name"} <= titles(static_agent(repo).review("use.py"))


def test_god_class_is_flagged(tmp_path):
    methods = "".join(f"    def m{i}(self):\n        return {i}\n" for i in range(21))
    repo = make_repo(tmp_path, {"god.py": f"class Everything:\n{methods}"})
    report = static_agent(repo).review("god.py")
    assert "God Class (too many responsibilities)" in titles(report)


def test_copy_pasted_function_is_detected(tmp_path):
    body = ("def total(items):\n    result = 0\n    for item in items:\n"
            "        if item > 0:\n            result += item\n    return result\n")
    repo = make_repo(tmp_path, {"one.py": body, "two.py": body.replace("total", "sum_positive")})
    report = static_agent(repo).review("two.py")
    finding = next(f for f in report.findings if f.title == "Duplicated Functionality")
    assert finding.related_files == ["one.py"]


def test_different_functions_are_not_duplicates(tmp_path):
    repo = make_repo(tmp_path, {
        "one.py": "def a(x):\n    y = x + 1\n    z = y * 2\n    w = z - 3\n    return w\n",
        "two.py": "def b(x):\n    y = x - 1\n    z = y / 2\n    w = z + 3\n    return w\n",
    })
    assert static_agent(repo).review("two.py").total_findings == 0


# --------------------------------------------------------------------------- #
# Input handling and robustness
# --------------------------------------------------------------------------- #
def test_test_files_are_skipped(tmp_path):
    repo = make_repo(tmp_path, {"tests/test_x.py": "import a\nimport b\n", "a.py": "", "b.py": ""})
    report = static_agent(repo).review("tests/test_x.py")
    assert report.total_findings == 0 and "Test files" in report.summary


def test_syntax_error_does_not_crash(tmp_path):
    repo = make_repo(tmp_path, {"bad.py": "def broken(:\n    pass\n"})
    report = static_agent(repo).review("bad.py")
    assert report.syntax_valid is False


def test_file_outside_repo_is_rejected(tmp_path):
    repo = make_repo(tmp_path / "repo", {"a.py": "X = 1\n"})
    outside = tmp_path / "other.py"
    outside.write_text("X = 1\n")
    with pytest.raises(ValueError):
        static_agent(repo).review(str(outside))


def test_pr_version_of_code_can_be_reviewed_instead_of_disk(tmp_path):
    repo = make_repo(tmp_path, LAYERED)
    pr_code = "from app.repositories.user_repository import get_user\n\ndef show(uid):\n    return get_user(uid)\n"
    report = static_agent(repo).review("app/controllers/user_controller.py", code=pr_code)
    assert "Layer Skipped" in titles(report)


def test_src_layout_projects_resolve_imports(tmp_path):
    files = {f"src/{k}": v for k, v in LAYERED.items()}
    files["src/app/repositories/user_repository.py"] = (
        "from app.services.user_service import load\n\ndef get_user(uid):\n    return load(uid)\n")
    report = static_agent(make_repo(tmp_path, files)).review("src/app/repositories/user_repository.py")
    assert "Upward Layer Dependency" in titles(report)


def test_changed_lines_filter(tmp_path):
    repo = make_repo(tmp_path, {"use.py": "from lib import *\nfrom lib import _s\n", "lib.py": "_s = 1\n"})
    report = static_agent(repo).review("use.py", changed_lines={2})
    assert [f.line for f in report.findings] == [2]


def test_health_score_drops_with_findings(tmp_path):
    clean = static_agent(make_repo(tmp_path / "c", LAYERED)).review("app/services/user_service.py")
    files = dict(LAYERED)
    files["app/repositories/user_repository.py"] = "from app.services.user_service import load\n"
    bad = static_agent(make_repo(tmp_path / "b", files)).review("app/repositories/user_repository.py")
    assert clean.health_score == 100 and bad.health_score < 100


def test_context_for_llm_mentions_imports_and_dependents(tmp_path):
    repo = make_repo(tmp_path, LAYERED)
    context = static_agent(repo).describe_context("app/services/user_service.py")
    assert "Target layer: service" in context
    assert "app/repositories/user_repository.py" in context     # something it imports
    assert "app/controllers/user_controller.py" in context      # something that imports it


# --------------------------------------------------------------------------- #
# Anti-hallucination (uses the fake LLM)
# --------------------------------------------------------------------------- #
CODE = "def show(uid):\n    if uid > 10:\n        price = uid * 1.2\n    return price\n"


def agent_with_fake_llm(tmp_path, findings) -> ArchitectureAgent:
    make_repo(tmp_path, {"app.py": CODE})
    agent = static_agent(tmp_path)
    agent._chain = FakeChain(findings)
    return agent


def test_grounded_llm_finding_is_kept(tmp_path):
    report = agent_with_fake_llm(tmp_path, [llm_finding(3, "price = uid * 1.2")]).review("app.py")
    assert report.total_findings == 1 and report.discarded_findings == 0


def test_hallucinated_llm_finding_is_discarded(tmp_path):
    report = agent_with_fake_llm(tmp_path, [llm_finding(3, "db.session.add(order)")]).review("app.py")
    assert report.total_findings == 0 and report.discarded_findings == 1


def test_wrong_line_number_is_relocated(tmp_path):
    report = agent_with_fake_llm(tmp_path, [llm_finding(40, "price = uid * 1.2")]).review("app.py")
    assert report.findings[0].line == 3
    assert report.findings[0].validation_status.value == "RELOCATED"


def test_low_confidence_llm_finding_is_discarded(tmp_path):
    report = agent_with_fake_llm(tmp_path, [llm_finding(3, "price = uid * 1.2", confidence=0.2)]).review("app.py")
    assert report.total_findings == 0


def test_unknown_llm_category_falls_back_to_design(tmp_path):
    finding = llm_finding(3, "price = uid * 1.2", category="spaghetti")
    assert finding.category == ArchCategory.DESIGN


def test_markdown_comment_contains_key_facts(tmp_path):
    repo = make_repo(tmp_path, {"a.py": "import b\n", "b.py": "import a\n"})
    comment = static_agent(repo).review("a.py").findings[0].to_markdown_comment()
    assert "Circular Dependency" in comment and "Suggested refactoring" in comment
