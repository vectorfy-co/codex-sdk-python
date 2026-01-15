#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


VERSION_LINE_RE = re.compile(
    r"^\s*(?:##\s*)?(?:version\s+)?v?\d+\.\d+\.\d+(?:[a-z0-9.+-]+)?\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_version(tag: str) -> str:
    tag = tag.strip()
    return tag[1:] if tag.startswith("v") else tag


def _find_section(text: str, version: str) -> str | None:
    header_pattern = re.compile(
        rf"^\s*(?:##\s*)?(?:version\s+)?v?{re.escape(version)}\b.*$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = header_pattern.search(text)
    if not match:
        return None

    next_match = VERSION_LINE_RE.search(text, match.end())
    end = next_match.start() if next_match else len(text)
    return text[match.start() : end].strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a version section from a changelog.",
    )
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v0.81.0")
    parser.add_argument(
        "--input",
        default="CHANGELOG_SDK.md",
        help="Path to the changelog file",
    )
    parser.add_argument(
        "--output",
        default="release_notes.md",
        help="Path to write extracted release notes",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    version = _normalize_version(args.tag)
    changelog_path = Path(args.input)
    if not changelog_path.exists():
        print(f"::error::Changelog not found: {changelog_path}", file=sys.stderr)
        return 1

    text = changelog_path.read_text(encoding="utf-8")
    section = _find_section(text, version)
    if not section:
        print(
            f"::error::Version {version} not found in {changelog_path}",
            file=sys.stderr,
        )
        return 1

    output_path = Path(args.output)
    output_path.write_text(section + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
