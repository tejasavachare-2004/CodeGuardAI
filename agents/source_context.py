"""Build bounded, syntax-aware source context for LLM reviews."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class NumberedSourceContext:
    """Bounded numbered source and whether it covers the complete file."""

    text: str
    complete: bool


def _numbered_line(line_number: int, text: str) -> str:
    return f"{line_number:>4}| {text}"


def _render_context(lines: list[str], ranges: list[tuple[int, int]]) -> str:
    parts: list[str] = []
    next_line = 1

    for start, end in sorted(ranges):
        if next_line < start:
            parts.append(f"... [original lines {next_line}-{start - 1} omitted] ...")
        parts.extend(_numbered_line(number, lines[number - 1])
                     for number in range(start, end + 1))
        next_line = end + 1

    if next_line <= len(lines):
        parts.append(f"... [original lines {next_line}-{len(lines)} omitted] ...")
    return "\n".join(parts)


def _node_range(node: ast.stmt) -> tuple[int, int]:
    decorators = getattr(node, "decorator_list", [])
    start = min([node.lineno] + [decorator.lineno for decorator in decorators])
    return start, getattr(node, "end_lineno", node.lineno)


def _node_priority(node: ast.stmt) -> int:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return 0
    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)):
        return 1
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return 2
    return 3


def build_numbered_source_context(
    code: str, max_chars: int, agent_name: str
) -> NumberedSourceContext:
    """Return original-line-numbered source, selecting whole AST statements if needed."""
    lines = code.splitlines()
    full_context = "\n".join(
        _numbered_line(number, line) for number, line in enumerate(lines, 1)
    )
    if len(full_context) <= max_chars:
        return NumberedSourceContext(full_context, complete=True)

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(
            f"{agent_name} cannot safely create bounded LLM context for oversized "
            f"syntactically invalid source (line {exc.lineno or 1})."
        ) from exc

    # Reserve ample space for omission markers, including their original line ranges.
    marker_budget = 80
    selected: list[tuple[int, int]] = []
    selected_source_chars = 0
    candidates = sorted(tree.body, key=lambda node: (_node_priority(node), node.lineno))

    for node in candidates:
        start, end = _node_range(node)
        block_chars = sum(
            len(_numbered_line(number, lines[number - 1]))
            for number in range(start, end + 1)
        ) + max(0, end - start)
        proposed_count = len(selected) + 1
        if (selected_source_chars + block_chars
                + marker_budget * (proposed_count + 1) > max_chars):
            continue
        selected.append((start, end))
        selected_source_chars += block_chars

    if not selected:
        raise ValueError(
            f"{agent_name} cannot safely create bounded LLM context: no complete "
            f"top-level AST statement fits within the {max_chars}-character limit."
        )

    context = _render_context(lines, selected)
    if len(context) > max_chars:
        raise ValueError(
            f"{agent_name} could not safely fit complete AST statements within the "
            f"{max_chars}-character LLM context limit."
        )
    return NumberedSourceContext(context, complete=False)
