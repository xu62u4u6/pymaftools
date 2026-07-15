"""Extract one version section from CHANGELOG.md for a GitHub Release."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
VERSION_HEADING_PATTERN = re.compile(
    r"^## .*?\bVersion\s+([0-9]+\.[0-9]+\.[0-9]+)(?:\s|$)"
)


def extract_release_notes(changelog: str, version: str) -> str:
    """Return the changelog body for exactly one semantic version."""
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"Version must match X.Y.Z: {version}")

    lines = changelog.splitlines()
    matches = [
        index
        for index, line in enumerate(lines)
        if (match := VERSION_HEADING_PATTERN.match(line)) and match.group(1) == version
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one changelog section for {version}, found {len(matches)}"
        )

    start = matches[0] + 1
    end = next(
        (
            index
            for index in range(start, len(lines))
            if VERSION_HEADING_PATTERN.match(lines[index])
        ),
        len(lines),
    )
    section = lines[start:end]

    while section and not section[-1].strip():
        section.pop()
    if section and section[-1].strip() == "---":
        section.pop()
    while section and not section[-1].strip():
        section.pop()
    while section and not section[0].strip():
        section.pop(0)

    notes = "\n".join(section)
    if not notes:
        raise ValueError(f"Changelog section for {version} is empty")
    return f"{notes}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="Release version without the v prefix")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    notes = extract_release_notes(
        args.changelog.read_text(encoding="utf-8"), args.version
    )
    args.output.write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
