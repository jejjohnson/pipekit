"""An opt-in guard that reports data moving between NumPy and JAX.

NumPy and JAX interoperate by default: a NumPy array next to a JAX one joins
it, and a NumPy-only step gets its data on the host. Nothing says so, because
a move is correct and most programs do not care. An app on a GPU does: inside
``guard("warn")`` each move warns, and inside ``guard("raise")`` it raises, so
a test finds the step that moves data.

Code that moves data calls `report_move`. With no guard that is one read of a
context variable.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import sys
import warnings
from collections.abc import Iterator
from typing import Literal, get_args

from pipekit_ops.adapters import BackendError


__all__ = ["BackendWarning", "GuardLevel", "guard", "guard_level", "report_move"]

GuardLevel = Literal["off", "warn", "raise"]

_LEVEL: contextvars.ContextVar[GuardLevel] = contextvars.ContextVar(
    "pipekit_guard", default="off"
)

#: the frames a warning skips, so that it names the caller's own line
_OWN_PACKAGES = ("pipekit_ops", "pipekit_array")


class BackendWarning(UserWarning):
    """Data moved between NumPy and JAX inside ``guard("warn")``."""


@contextlib.contextmanager
def guard(level: GuardLevel) -> Iterator[None]:
    """Report each move between NumPy and JAX inside the ``with`` block.

    The level holds in this thread and in pipekit's thread and async
    executors, which copy the context. A thread pool of your own does not,
    unless it runs ``contextvars.copy_context().run``; a process never does.

    Args:
        level: ``"warn"`` for a `BackendWarning` per move, ``"raise"`` for a
            `BackendError` at the first one, ``"off"`` (the default outside
            any guard) to report nothing inside another guard.

    Raises:
        ValueError: ``level`` is not one of the three.

    Example:
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as caught, guard("warn"):
        ...     warnings.simplefilter("always")
        ...     report_move("Smooth", "JAX data copied to the host")
        >>> str(caught[0].message)
        'Smooth: JAX data copied to the host'
        >>> report_move("Smooth", "JAX data copied to the host")  # no guard
    """
    if level not in get_args(GuardLevel):
        raise ValueError(f"guard: level {level!r} is not one of {get_args(GuardLevel)}")
    token = _LEVEL.set(level)
    try:
        yield
    finally:
        _LEVEL.reset(token)


def guard_level() -> GuardLevel:
    """The level of the innermost `guard`; ``"off"`` outside any.

    Example:
        >>> guard_level()
        'off'
        >>> with guard("raise"):
        ...     guard_level()
        'raise'
    """
    return _LEVEL.get()


def report_move(what: str, move: str) -> None:
    """Tell the guard that ``what`` moved data; called by the code that does.

    Args:
        what: The operator or function that moved the data.
        move: What happened, e.g. ``"JAX data copied to the host"``.

    Raises:
        BackendError: Inside ``guard("raise")``.

    Example:
        >>> with guard("raise"):  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     report_move("combine", "NumPy data joined JAX")
        Traceback (most recent call last):
        ...
        BackendError: combine: NumPy data joined JAX
    """
    level = _LEVEL.get()
    if level == "off":
        return
    message = f"{what}: {move}"
    if level == "raise":
        raise BackendError(message)
    warnings.warn(message, BackendWarning, skip_file_prefixes=_own_files())


def _own_files() -> tuple[str, ...]:
    """The directories of pipekit's packages that are imported."""
    files = (getattr(sys.modules.get(p), "__file__", None) for p in _OWN_PACKAGES)
    return tuple(os.path.dirname(file) for file in files if file)
