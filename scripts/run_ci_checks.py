#!/usr/bin/env python3
"""Run the same local checks that gate CI before pushing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


def run_command(cmd: Sequence[str], cwd: Path) -> None:
    print(f"==> {' '.join(cmd)}")
    subprocess.run(list(cmd), cwd=cwd, check=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    commands = [
        ["uv", "sync", "--locked", "--all-extras", "--dev"],
        [
            "uv",
            "run",
            "black",
            "--check",
            "src",
            "tests",
            "examples",
            "scripts/install_git_hooks.py",
            "scripts/run_ci_checks.py",
            "scripts/setup_binary.py",
        ],
        [
            "uv",
            "run",
            "isort",
            "--check-only",
            "src",
            "tests",
            "examples",
            "scripts/install_git_hooks.py",
            "scripts/run_ci_checks.py",
            "scripts/setup_binary.py",
        ],
        [
            "uv",
            "run",
            "flake8",
            "src",
            "tests",
            "scripts/install_git_hooks.py",
            "scripts/run_ci_checks.py",
            "scripts/setup_binary.py",
        ],
        ["uv", "run", "mypy", "src"],
        [sys.executable, "scripts/setup_binary.py", "--verify-only"],
        ["uv", "run", "pytest", "--cov=codex_sdk", "--cov-report=term-missing"],
    ]

    try:
        for cmd in commands:
            run_command(cmd, repo_root)
    except subprocess.CalledProcessError as exc:
        print(f"\nCI-style validation failed with exit code {exc.returncode}.")
        return exc.returncode

    print("\nAll CI-style validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
