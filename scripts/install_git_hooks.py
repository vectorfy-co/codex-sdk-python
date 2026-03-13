#!/usr/bin/env python3
"""Install the repository-managed Git hooks for this clone."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    hooks_dir = repo_root / ".githooks"
    pre_push_hook = hooks_dir / "pre-push"

    if not pre_push_hook.exists():
        print(f"Missing hook file: {pre_push_hook}", file=sys.stderr)
        return 1

    current_mode = pre_push_hook.stat().st_mode
    pre_push_hook.chmod(current_mode | 0o111)

    subprocess.run(
        ["git", "config", "core.hooksPath", str(hooks_dir.relative_to(repo_root))],
        cwd=repo_root,
        check=True,
    )

    print(f"Installed repository Git hooks from {hooks_dir}")
    print("You can skip the pre-push hook for one push with SKIP_PRE_PUSH_CI=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
