"""Where a saved pipeline says that an operator lives."""

from __future__ import annotations

from collections.abc import Callable


def recorded[T: type](*former: str) -> Callable[[T], T]:
    """Record an operator under ``pipekit_array``, wherever its code is.

    The module of an operator is part of its saved state and of its
    `pipekit_ops.pipeline_hash`, so code that moves would change both. A state
    saved under a former module still loads.

    Args:
        *former: The modules that states were saved under before.

    Returns:
        The class decorator.
    """

    def decorate(cls: T) -> T:
        cls.__module__ = "pipekit_array"
        setattr(cls, "__former_modules__", former)  # noqa: B010 (a type has none)
        return cls

    return decorate
