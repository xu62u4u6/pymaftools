"""Atomic filesystem output helpers."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator


@contextmanager
def atomic_output_path(target: str | Path) -> Iterator[Path]:
    """Yield a temporary sibling path and atomically replace target on success."""
    target = Path(target)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        yield temporary_path
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
