#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["keepachangelog>=2.0.0"]
# ///
"""Extract release notes for a tag from CHANGELOG_SDK.md.

Usage:
    uv run scripts/extract_changelog.py --tag v0.89.0 --input CHANGELOG_SDK.md --output release_notes.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import keepachangelog  # pyright: ignore[reportMissingImports]


def _normalize_version(tag: str) -> str:
    """Remove leading 'v' from version tag if present."""
    return tag[1:] if tag.startswith("v") else tag


def _format_section_to_markdown(version: str, data: dict) -> str:
    """Convert a keepachangelog section dict back to markdown format."""
    lines: list[str] = []

    # Header with version and date (release_date may be top-level or in metadata)
    release_date = data.get("release_date") or data.get("date")
    if not release_date:
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            release_date = metadata.get("release_date") or metadata.get("date")
    if release_date:
        lines.append(f"## [{version}] - {release_date}")
    else:
        lines.append(f"## [{version}]")
    lines.append("")

    # Standard changelog categories in order
    categories = [
        ("added", "Added"),
        ("changed", "Changed"),
        ("deprecated", "Deprecated"),
        ("removed", "Removed"),
        ("fixed", "Fixed"),
        ("security", "Security"),
        ("updated", "Updated"),
        ("notes", "Notes"),
    ]

    for key, title in categories:
        items = data.get(key)
        if items:
            lines.append(f"### {title}")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

    return "\n".join(lines).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a version section from a changelog."
    )
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v0.89.0")
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
    args = parser.parse_args()

    version = _normalize_version(args.tag)
    changelog_path = Path(args.input)

    if not changelog_path.exists():
        print(f"::error::Changelog not found: {changelog_path}", file=sys.stderr)
        return 1

    try:
        changelog = keepachangelog.to_dict(str(changelog_path))
    except Exception as e:
        print(f"::error::Failed to parse changelog: {e}", file=sys.stderr)
        return 1

    if not changelog:
        print(f"::error::No versions found in {changelog_path}", file=sys.stderr)
        return 1

    # Try to find the requested version
    if version in changelog:
        section_data = changelog[version]
        body = _format_section_to_markdown(version, section_data)
    else:
        # Fallback to latest version
        versions = list(changelog.keys())
        latest_version = versions[0]
        print(
            f"::warning::Version {version} not found in {changelog_path}; "
            f"using latest version {latest_version} instead.",
            file=sys.stderr,
        )
        print(
            f"::notice::Available versions: {', '.join(versions[:10])}"
            + (" ..." if len(versions) > 10 else ""),
            file=sys.stderr,
        )
        section_data = changelog[latest_version]
        body = _format_section_to_markdown(latest_version, section_data)

    output_path = Path(args.output)
    output_path.write_text(body + "\n", encoding="utf-8")
    print(f"Wrote release notes to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
