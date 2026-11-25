"""A copy of an array with some entries changed, or with zeros around it.

The array API standard has no functional update. On JAX data each function is
JAX's own call, and lowers to the same jaxpr; elsewhere it is a plain equivalent
(ADR A27). `pipekit_array.named.pad` is the one for named dimensions.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import array_api_compat
import numpy as np
from pipekit_ops import BackendError

from pipekit_array.arrays._backend import namespace_of
from pipekit_array.arrays._host import is_host, like, table_like


__all__ = ["add_at", "multiply_at", "pad", "set_at"]


def set_at(x: Any, index: Any, values: Any) -> Any:
    """A copy of an array with some entries replaced: y = x, then y[index] = v.

    On JAX it is ``x.at[index].set(values)``.

    Args:
        x: The array, which is not changed.
        index: What ``x[index]`` accepts: integers, slices, an ellipsis, a
            boolean mask or an integer array. An entry that an integer array
            names twice takes one of its two values.
        values: The new entries, which broadcast to ``x[index]`` and take the
            dtype of ``x``.

    Returns:
        y, in ``x``'s library, shape and dtype.

    Raises:
        BackendError: If the array library does not take that kind of index, or
            if two libraries other than NumPy meet in one call.
        IndexError: If the index is out of bounds. JAX does not report it, and
            leaves the array as it is.

    Example:
        >>> import numpy as np
        >>> set_at(np.zeros(3), slice(1, None), [1.0, 2.0])
        array([0., 1., 2.])
    """
    xp, x, index = _update_inputs(x, index, values)
    if array_api_compat.is_jax_namespace(xp):
        return x.at[index].set(values)
    return _assign("set_at", xp, x, index, values)


def add_at(x: Any, index: Any, values: Any) -> Any:
    """A copy of an array with some entries added to: y = x, then y[index] += v.

    An entry that the index names twice is added to twice, which ``y[index] += v``
    does not do on NumPy. On JAX it is ``x.at[index].add(values)``.

    Args:
        x: The array, which is not changed.
        index: As in `set_at`; an integer array needs NumPy or JAX.
        values: What to add, which broadcasts to ``x[index]``.

    Returns:
        y, in ``x``'s library, shape and dtype.

    Raises:
        BackendError: As `set_at`.
        IndexError: As `set_at`.

    Example:
        >>> import numpy as np
        >>> add_at(np.zeros(4), np.asarray([0, 2, 2]), [1.0, 10.0, 100.0])
        array([  1.,   0., 110.,   0.])
    """
    xp, x, index = _update_inputs(x, index, values)
    if array_api_compat.is_jax_namespace(xp):
        return x.at[index].add(values)
    if _has_integer_array(xp, index):
        return _accumulate("add_at", np.add, xp, x, index, values)
    # the index names an entry once
    return _assign("add_at", xp, x, index, x[index] + like(values, x, xp))


def multiply_at(x: Any, index: Any, values: Any) -> Any:
    """A copy of an array with some entries multiplied: y = x, then y[index] *= v.

    An entry that the index names twice is multiplied twice. On JAX it is
    ``x.at[index].multiply(values)``.

    Args:
        x: The array, which is not changed.
        index: As in `set_at`; an integer array needs NumPy or JAX.
        values: The factors, which broadcast to ``x[index]``.

    Returns:
        y, in ``x``'s library, shape and dtype.

    Raises:
        BackendError: As `set_at`.
        IndexError: As `set_at`.

    Example:
        >>> import numpy as np
        >>> multiply_at(np.ones(3), np.asarray([1, 1]), [2.0, 3.0])
        array([1., 6., 1.])
    """
    xp, x, index = _update_inputs(x, index, values)
    if array_api_compat.is_jax_namespace(xp):
        return x.at[index].multiply(values)
    if _has_integer_array(xp, index):
        return _accumulate("multiply_at", np.multiply, xp, x, index, values)
    return _assign("multiply_at", xp, x, index, x[index] * like(values, x, xp))


def pad(x: Any, widths: tuple[tuple[int, int], ...]) -> Any:
    """An array with zeros around it.

    Args:
        x: The array.
        widths: How many zeros before and after, for each axis.

    Returns:
        The larger array, in ``x``'s library and dtype, on its device.

    Raises:
        BackendError: If ``x`` is not an array of a known library.
        ValueError: If ``widths`` is not one pair per axis.

    Example:
        >>> import numpy as np
        >>> pad(np.ones(2), ((1, 2),))
        array([0., 1., 1., 0., 0.])
    """
    xp = namespace_of(x)
    x = xp.asarray(x) if is_host(x) else x
    if len(widths) != x.ndim:
        msg = f"pad needs one pair of widths per axis, got {len(widths)} for {x.ndim}"
        raise ValueError(msg)
    native = getattr(xp, "pad", None)
    if native is not None:
        return native(x, widths)

    def zeros(width: int, axis: int) -> Any:
        shape = (*x.shape[:axis], width, *x.shape[axis + 1 :])
        return xp.zeros(shape, dtype=x.dtype, device=array_api_compat.device(x))

    for axis, (before, after) in enumerate(widths):
        x = xp.concat([zeros(before, axis), x, zeros(after, axis)], axis=axis)
    return x


def _update_inputs(x: Any, index: Any, values: Any) -> tuple[ModuleType, Any, Any]:
    """The namespace, the array in it, and the index with its arrays in it."""
    parts = index if isinstance(index, tuple) else (index,)
    arrays = [part for part in parts if hasattr(part, "dtype")]
    xp = namespace_of(x, values, *arrays)
    x = xp.asarray(x) if is_host(x) else x
    if array_api_compat.is_jax_namespace(xp):
        return xp, x, index
    parts = tuple(
        table_like(part, x, xp) if isinstance(part, np.ndarray) else part
        for part in parts
    )
    return xp, x, parts if isinstance(index, tuple) else parts[0]


def _has_integer_array(xp: ModuleType, index: Any) -> bool:
    """Whether an index may name an entry twice."""
    parts = index if isinstance(index, tuple) else (index,)
    return any(
        isinstance(part, list)
        or (getattr(part, "ndim", 0) > 0 and xp.isdtype(part.dtype, "integral"))
        for part in parts
    )


def _assign(name: str, xp: ModuleType, x: Any, index: Any, values: Any) -> Any:
    """A copy of ``x`` with ``values`` at the index."""
    # astype, because asarray would detach a torch tensor from its graph
    y = xp.astype(x, x.dtype, copy=True)
    try:
        y[index] = like(values, x, xp)
    except (IndexError, NotImplementedError) as err:
        strict = array_api_compat.is_array_api_strict_namespace(xp)
        if isinstance(err, NotImplementedError) or (
            strict and _has_integer_array(xp, index)
        ):
            raise _refusal(name, xp) from err
        raise
    return y


def _accumulate(
    name: str, ufunc: np.ufunc, xp: ModuleType, x: Any, index: Any, values: Any
) -> Any:
    """A copy of ``x``, updated once for each time the index names an entry."""
    if not array_api_compat.is_numpy_namespace(xp):
        raise _refusal(name, xp)
    y = np.array(x)
    ufunc.at(y, index, like(values, x, xp))
    return y


def _refusal(name: str, xp: ModuleType) -> BackendError:
    library = xp.__name__.removeprefix("array_api_compat.").partition(".")[0]
    return BackendError(f"{name} does not take this index on {library}")
