"""Checks of an argument that also work under ``jax.jit``.

A check needs values, and a traced array has none. `check_values` raises at once
on a concrete array of any library, and when the compiled code runs on a traced
one, through ``equinox.error_if``. equinox is imported under a trace only
(ADR A28).
"""

from __future__ import annotations

from typing import Any

import array_api_compat
import numpy as np
from pipekit_ops import import_optional, is_concrete

from pipekit_array.arrays._backend import concrete, namespace_of
from pipekit_array.arrays._host import is_host


__all__ = ["check_values", "require_floating"]


def check_values(
    array: Any, bad: Any, message: str, *, error: type[Exception] = ValueError
) -> Any:
    """An array, or an error if its values are wrong.

    The check belongs to the array that is returned: under a trace, use that one,
    because JAX drops a check whose result nothing uses.

    Args:
        array: The array that the check is about.
        bad: Whether the check fails, a boolean scalar.
        message: What must hold, naming the argument (``"'x' must be > 0"``).
        error: The exception to raise on concrete arrays, such as a library's
            own. Under a trace it is equinox's, with the same message.

    Returns:
        ``array``.

    Raises:
        ValueError: ``error``, if ``bad`` holds on concrete arrays; its message
            ends with the array.
        ImportError: Under a trace, if equinox is not installed.

    Example:
        >>> import numpy as np
        >>> x = np.asarray([1.0, 2.0])
        >>> check_values(x, np.any(x <= 0), "'x' must be positive")
        array([1., 2.])
        >>> check_values(x, np.any(x > 1), "'x' must be ≤ 1")
        Traceback (most recent call last):
            ...
        ValueError: 'x' must be ≤ 1, got array([1., 2.])
    """
    if not is_concrete(array, bad):
        equinox = import_optional(
            "equinox", install="equinox", needed_by="check_values under a JAX trace"
        )
        return equinox.error_if(array, bad, message)
    if bool(bad):
        raise error(f"{message}, got {_shown(array)!r}")
    return array


def require_floating(
    array: Any, name: str, *, error: type[Exception] = ValueError
) -> None:
    """Raise unless an array has a real floating dtype; nothing is cast.

    Args:
        array: An array of any library, concrete or traced, a sequence or a
            number.
        name: What the argument is called, for the error.
        error: The exception to raise.

    Raises:
        ValueError: ``error``, if the dtype is not real floating.
        BackendError: If ``array`` is not an array of a known library.

    Example:
        >>> import numpy as np
        >>> require_floating(np.zeros(3), "x")
        >>> require_floating(np.arange(3), "x")
        Traceback (most recent call last):
            ...
        ValueError: 'x' must have a floating dtype, got int64
    """
    xp = namespace_of(array)
    dtype = np.asarray(array).dtype if is_host(array) else array.dtype
    if not xp.isdtype(dtype, "real floating"):
        msg = f"'{name}' must have a floating dtype, got {dtype}"
        raise error(msg)


def _shown(array: Any) -> Any:
    """What an error shows of an array: its values, without where they are."""
    known = array_api_compat.is_jax_array(array) or array_api_compat.is_torch_array(
        array
    )
    return concrete(array) if known or is_host(array) else array
