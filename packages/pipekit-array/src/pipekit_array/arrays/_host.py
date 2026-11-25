"""Values of the host in the library, the dtype and the device of an array.

Python numbers, sequences and NumPy arrays live on the host. `like` is for a
constant, `constants_like` for a table of numbers, `table_like` for a table that
keeps its dtype (an index, a mask) and `boolean_like` for a boolean argument.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import array_api_compat
import array_api_compat.numpy as numpy_namespace
import numpy as np


__all__ = ["boolean_like", "constants_like", "like", "table_like"]


# Values that live on the host and join whatever library the others are from.
_HOST_TYPES = (bool, int, float, complex, np.ndarray, np.generic, list, tuple)


def is_host(value: Any) -> bool:
    """Whether a value is ``None``, a Python number, a sequence or NumPy's."""
    return value is None or isinstance(value, _HOST_TYPES)


def is_number(value: Any) -> bool:
    """A Python real number, which adopts the dtype of the arrays.

    A NumPy scalar carries its own dtype, though ``np.float64`` is a float.
    """
    return isinstance(value, (bool, int, float)) and not isinstance(value, np.generic)


def device_of(value: Any, xp: ModuleType) -> Any:
    """The device that new arrays join ``value`` on, or ``None`` for the default.

    A JAX tracer has no device, and an uncommitted JAX scalar joins an array
    on any device, so JAX is left to place its own arrays.
    """
    if is_host(value) or array_api_compat.is_jax_namespace(xp):
        return None
    return array_api_compat.device(value)


def place(value: Any, xp: ModuleType, dtype: Any, device: Any) -> Any:
    """``value`` as an array of ``dtype``: cast if it is one already, else built.

    ``device`` is where a host value goes; an array stays on its own.
    """
    if not is_host(value):
        # asarray would detach a torch tensor from its graph (torch 2.5, 2.10)
        return xp.astype(value, dtype, copy=False)
    if device is None:
        return xp.asarray(value, dtype=dtype)
    return xp.asarray(value, dtype=dtype, device=device)


def like(value: Any, ref: Any, xp: ModuleType) -> Any:
    """``value`` (a number or a 0-d array) in ``ref``'s dtype, on its device.

    Args:
        value: The constant.
        ref: An array of the namespace ``xp``.
        xp: The namespace of the call.

    Returns:
        A 0-d array, or ``value`` itself if nothing changes.

    Example:
        >>> import numpy as np
        >>> like(0.1, np.zeros(2, dtype=np.float32), np).dtype
        dtype('float32')
    """
    return place(value, xp, ref.dtype, device_of(ref, xp))


def table_like(table: np.ndarray, ref: Any, xp: ModuleType) -> Any:
    """A NumPy table (an index, a mask) in ``ref``'s library, on its device.

    Unlike `like`, it keeps its own shape and dtype.

    Args:
        table: The table.
        ref: An array of the namespace ``xp``.
        xp: The namespace of the call.

    Returns:
        The table as an array of ``xp``.

    Example:
        >>> import numpy as np
        >>> table_like(np.arange(3), np.zeros(2, dtype=np.float32), np).tolist()
        [0, 1, 2]
    """
    device = device_of(ref, xp)
    return xp.asarray(table) if device is None else xp.asarray(table, device=device)


def constants_like(table: Any, ref: Any, xp: ModuleType) -> Any:
    """A table of numbers in ``ref``'s library and dtype, on its device.

    The numbers are constants of the problem (weights, the spacing of a grid),
    and no request for a precision, so a float64 table leaves a float32 field in
    float32. A table that is an array of ``xp`` already, such as the metric of a
    grid that is traced with its field, keeps its gradient.

    Args:
        table: The numbers: an array, a sequence or a number.
        ref: An array of the namespace ``xp``. If its dtype is not floating, the
            table keeps its own, so that an integer field does not truncate it.
        xp: The namespace of the call.

    Returns:
        The table as an array of ``xp``, in its own shape.

    Example:
        >>> import numpy as np
        >>> constants_like([0.5, 1.5], np.zeros(3, dtype=np.float32), np)
        array([0.5, 1.5], dtype=float32)
        >>> constants_like([0.5, 1.5], np.zeros(3, dtype=np.int64), np)
        array([0.5, 1.5])
    """
    if xp.isdtype(ref.dtype, ("real floating", "complex floating")):
        return place(table, xp, ref.dtype, device_of(ref, xp))
    return table_like(np.asarray(table), ref, xp) if is_host(table) else table


def boolean_like(value: Any, ref: Any, xp: ModuleType, *, name: str = "valid") -> Any:
    """A boolean argument (a mask) in ``ref``'s library, on its device.

    Args:
        value: A boolean, a sequence of them, or a boolean array. A value on the
            host joins ``ref``; any other array must be of its library already.
        ref: An array of the namespace ``xp``.
        xp: The namespace of the call.
        name: What the argument is called, for the error.

    Returns:
        The mask as an array of ``xp``, in its own shape.

    Raises:
        ValueError: If ``value`` is not boolean.

    Example:
        >>> import numpy as np
        >>> boolean_like([True, False], np.zeros(2), np).tolist()
        [True, False]
        >>> boolean_like([1, 0], np.zeros(2), np)
        Traceback (most recent call last):
            ...
        ValueError: 'valid' must be boolean (True = use), got dtype int64
    """
    on_host = is_host(value)
    if on_host:
        value = np.asarray(value)
    # a NumPy dtype is one of every namespace that NumPy arrays join
    namespace = numpy_namespace if on_host else xp
    if not namespace.isdtype(value.dtype, "bool"):
        msg = f"'{name}' must be boolean (True = use), got dtype {value.dtype}"
        raise ValueError(msg)
    return table_like(value, ref, xp) if on_host else value
