"""One place for deprecation policy.

0.6.0 unified public names (snake_case, ``to_h5``/``read_h5`` everywhere,
project-specific names off the top level). Old names keep working and only
warn; they are removed in :data:`REMOVAL_VERSION`. Every deprecation message
names the replacement and that version, so callers see one consistent story.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

REMOVAL_VERSION = "1.0"
"""Release in which names deprecated in 0.6.0 (and earlier) are removed."""

F = TypeVar("F", bound=Callable[..., Any])


def deprecation_message(old: str, new: str, extra: str = "") -> str:
    """Return the standard message for a deprecated name."""
    message = (
        f"{old} is deprecated and will be removed in pymaftools "
        f"{REMOVAL_VERSION}; use {new} instead."
    )
    return f"{message} {extra}".rstrip()


def warn_deprecated(old: str, new: str, extra: str = "", stacklevel: int = 3) -> None:
    """Emit the standard :class:`DeprecationWarning` for ``old``."""
    warnings.warn(
        deprecation_message(old, new, extra), DeprecationWarning, stacklevel=stacklevel
    )


def deprecated_alias(func: F, old: str, new: str) -> F:
    """Return a wrapper of ``func`` that warns that ``old`` is deprecated.

    Works for plain functions, methods and (when ``func`` is the underlying
    function) classmethods; wrap the result in ``classmethod`` yourself.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warn_deprecated(old, new)
        return func(*args, **kwargs)

    wrapper.__doc__ = (
        f"Deprecated alias of :obj:`{new}`; removed in {REMOVAL_VERSION}.\n\n"
        f".. deprecated:: 0.6.0\n    Use :obj:`{new}`."
    )
    return wrapper  # type: ignore[return-value]


class DeprecatedClassAttribute:
    """Class-attribute alias that warns on access (class or instance)."""

    def __init__(self, target: str, old: str, new: str) -> None:
        self.target = target
        self.old = old
        self.new = new

    def __get__(self, instance: Any, owner: type) -> Any:
        warn_deprecated(self.old, self.new, stacklevel=2)
        return getattr(owner, self.target)
