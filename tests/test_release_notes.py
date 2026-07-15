from __future__ import annotations

import pytest

from scripts.extract_release_notes import extract_release_notes


CHANGELOG = """# Changelog

## Version 1.2.0 (July 2026)

Summary.

### Changes
* Added a feature.

---

## Version 1.1.0 (June 2026)

Previous notes.
"""


def test_extract_release_notes_returns_only_requested_version() -> None:
    notes = extract_release_notes(CHANGELOG, "1.2.0")

    assert notes == "Summary.\n\n### Changes\n* Added a feature.\n"


@pytest.mark.parametrize("version", ["v1.2.0", "1.2", "latest"])
def test_extract_release_notes_rejects_invalid_version(version: str) -> None:
    with pytest.raises(ValueError, match="must match X.Y.Z"):
        extract_release_notes(CHANGELOG, version)


def test_extract_release_notes_rejects_missing_version() -> None:
    with pytest.raises(ValueError, match="found 0"):
        extract_release_notes(CHANGELOG, "2.0.0")


def test_extract_release_notes_rejects_duplicate_version() -> None:
    duplicate = f"{CHANGELOG}\n## Version 1.2.0\n\nDuplicate.\n"

    with pytest.raises(ValueError, match="found 2"):
        extract_release_notes(duplicate, "1.2.0")


def test_extract_release_notes_rejects_empty_section() -> None:
    changelog = "## Version 2.0.0\n\n---\n\n## Version 1.0.0\n\nOld.\n"

    with pytest.raises(ValueError, match="is empty"):
        extract_release_notes(changelog, "2.0.0")
