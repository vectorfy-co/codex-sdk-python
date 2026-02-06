"""Docstring coverage checks.

These tests ensure the SDK remains consistently documented in Google docstring style.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Tuple


def _iter_python_modules(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        yield path


def _collect_docstring_issues(root: Path) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    non_google: List[str] = []

    for path in _iter_python_modules(root):
        module = ast.parse(path.read_text(encoding="utf-8"))

        if ast.get_docstring(module) is None:
            missing.append(f"{path}: module docstring missing")

        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ds = ast.get_docstring(node)
                if ds is None:
                    missing.append(
                        f"{path}: {node.__class__.__name__} {node.name} docstring missing"
                    )
                elif "Parameters:" in ds:
                    non_google.append(
                        f"{path}: {node.name} uses 'Parameters:' (use Google-style 'Args:')"
                    )

                if isinstance(node, ast.ClassDef):
                    for sub in node.body:
                        if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        # Keep private helpers flexible, but enforce public + dunder methods.
                        if sub.name.startswith("_") and not sub.name.startswith("__"):
                            continue
                        mds = ast.get_docstring(sub)
                        if mds is None:
                            missing.append(
                                f"{path}: {node.name}.{sub.name} docstring missing"
                            )
                        elif "Parameters:" in mds:
                            non_google.append(
                                f"{path}: {node.name}.{sub.name} uses 'Parameters:' (use Google-style 'Args:')"
                            )

    return missing, non_google


def test_docstring_coverage() -> None:
    missing, non_google = _collect_docstring_issues(Path("src/codex_sdk"))
    assert not missing, "Missing docstrings:\n" + "\n".join(missing)
    assert not non_google, "Non-Google docstrings:\n" + "\n".join(non_google)
