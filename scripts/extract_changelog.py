#!/usr/bin/env python3
"""Extract release notes for a tag from CHANGELOG_SDK.md."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def extract_section(text: str, match: re.Match[str]) -> str:
    start = match.start()
    remainder = text[match.end() :]
    next_header = re.search(r"^\s*##\s+\S+.*$", remainder, re.MULTILINE)
    end = match.end() + (next_header.start() if next_header else len(remainder))
    return text[start:end].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract changelog section.")
    parser.add_argument("--tag", required=True, help="Release tag (e.g. v0.81.0)")
    parser.add_argument("--input", required=True, help="Path to CHANGELOG_SDK.md")
    parser.add_argument("--output", required=True, help="Output release notes file")
    args = parser.parse_args()

    tag = args.tag
    version = tag[1:] if tag.startswith("v") else tag
    path = Path(args.input)
    text = path.read_text(encoding="utf-8")

    header_pattern = re.compile(rf"^\s*##\s+v?{re.escape(version)}\b.*$", re.MULTILINE)
    match = header_pattern.search(text)
    if match:
        body = extract_section(text, match)
    else:
        first_header = re.search(r"^\s*##\s+\S+.*$", text, re.MULTILINE)
        if first_header:
            print(
                f"::warning::Version {version} not found in {path}; "
                "using the latest changelog section instead."
            )
            body = extract_section(text, first_header)
        else:
            print(
                f"::warning::No changelog sections found in {path}; "
                "using the full changelog contents instead."
            )
            body = text.strip()

    Path(args.output).write_text(body + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


VERSION_RE = r"(?P<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)"

HEADER_WITH_VERSION_RE = re.compile(
    rf"^\s*(?![-*+]\s)(?:#{{1,6}}\s*)?(?:version|release)?\s*"
    rf"[:\-]?\s*[\[\(]?\s*v?{VERSION_RE}\s*[\]\)]?.*$",
    re.IGNORECASE | re.MULTILINE,
)

MARKDOWN_HEADER_RE = re.compile(r"^\s*#{1,6}\s+.+$", re.MULTILINE)


def _normalize_version(value: str) -> str:
    value = value.strip()
    return value[1:] if value.startswith("v") else value


def _collect_headers(text: str) -> list[dict[str, int | str]]:
    headers: list[dict[str, int | str]] = []
    for match in HEADER_WITH_VERSION_RE.finditer(text):
        version_raw = match.group("version")
        if not version_raw:
            continue
        headers.append(
            {
                "version": _normalize_version(version_raw),
                "start": match.start(),
                "line": match.group(0),
            }
        )
    for index, header in enumerate(headers):
        next_start = headers[index + 1]["start"] if index + 1 < len(headers) else len(
            text
        )
        header["end"] = next_start
    return headers


def _format_detected_versions(headers: list[dict[str, int | str]]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for header in headers:
        version = str(header["version"])
        if version not in seen:
            ordered.append(version)
            seen.add(version)
    return ", ".join(ordered) if ordered else "none"


def _find_section(text: str, version: str) -> str | None:
    headers = _collect_headers(text)
    for header in headers:
        if header["version"] == version:
            start = int(header["start"])
            end = int(header["end"])
            return text[start:end].strip()
    return None


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
        headers = _collect_headers(text)
        detected = _format_detected_versions(headers)
        print(
            f"::error::Version {version} not found in {changelog_path}",
            file=sys.stderr,
        )
        print(f"::error::Detected versions: {detected}", file=sys.stderr)
        print("::group::Changelog header preview", file=sys.stderr)
        for line in text.splitlines()[:20]:
            print(line, file=sys.stderr)
        print("::endgroup::", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.write_text(section + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
