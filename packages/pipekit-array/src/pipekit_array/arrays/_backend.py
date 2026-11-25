"""The namespace, the dtype and the device that a function of plain arrays runs in.

The host rule (ADR A23): Python numbers, sequences and NumPy arrays live on
the host, and join the library and the device of the other arrays. No other
array is moved, and two libraries other than NumPy in one call raise.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import array_api_compat
import array_api_compat.numpy as numpy_namespace
import numpy as np
from pipekit_ops import BackendError, guard_level, is_traced, report_move

from pipekit_array.arrays._host import device_of, is_host, is_number, place


__all__ = ["as_inexact", "concrete", "floating_inputs", "is_dataarray", "namespace_of"]


def namespace_of(*values: Any) -> ModuleType:
    """The array namespace that a computation on ``values`` runs in.

    Python numbers, ``None``, sequences and NumPy arrays join the library of
    the other arrays; on their own they resolve to NumPy.

    Args:
        *values: Arguments of the function being dispatched.

    Returns:
        The namespace, such as ``array_api_compat.numpy`` or ``jax.numpy``.

    Raises:
        BackendError: If a value is not an array of a known library, or if two
            libraries other than NumPy meet in one call.

    Example:
        >>> import numpy as np
        >>> namespace_of(np.zeros(3), 2.0, None).__name__
        'array_api_compat.numpy'
    """
    found: dict[str, ModuleType] = {}
    for value in values:
        if is_host(value):
            continue
        try:
            namespace = array_api_compat.array_namespace(value)
        except TypeError as err:
            msg = f"{type(value).__name__} is not an array of a known library"
            raise BackendError(msg) from err
        found[namespace.__name__] = namespace
    if len(found) > 1:
        msg = f"arrays of several libraries in one call: {', '.join(sorted(found))}"
        raise BackendError(msg)
    return next(iter(found.values()), numpy_namespace)


def floating_inputs(
    *values: Any, params: tuple[Any, ...] = ()
) -> tuple[ModuleType, list[Any]]:
    """The namespace of a call and its array inputs in one floating dtype.

    The dtype is the common one of the floating arrays (float32 stays
    float32). Python numbers and integer or boolean arrays adopt it; with no
    floating array, everything takes the namespace's default float. Values on
    the host join the device of the first array that is not.

    Args:
        *values: The array inputs: arrays, sequences or Python numbers.
        params: Other parameters, Python floats or 0-d arrays. They help to
            choose the namespace and the device, so a traced JAX scalar
            selects JAX, and never decide the dtype.

    Returns:
        The namespace, and ``values`` as arrays of one floating dtype.

    Raises:
        BackendError: As `namespace_of`.
        ValueError: If an input is complex, a date or not numeric.

    Example:
        >>> import numpy as np
        >>> xp, (x, y) = floating_inputs(np.zeros(2, dtype=np.float32), 1.5)
        >>> x.dtype, y.dtype
        (dtype('float32'), dtype('float32'))
    """
    xp = namespace_of(*values, *params)
    devices = (device_of(value, xp) for value in (*values, *params))
    device = next((found for found in devices if found is not None), None)
    floating = []
    for value in values:
        if is_number(value):
            continue
        dtype = xp.asarray(value).dtype if is_host(value) else value.dtype
        if xp.isdtype(dtype, "real floating"):
            floating.append(dtype)
        elif not xp.isdtype(dtype, ("bool", "integral")):
            msg = f"inputs must be real numbers, got dtype {dtype}"
            raise ValueError(msg)
    dtype = xp.result_type(*floating) if floating else xp.asarray(0.0).dtype
    if guard_level() != "off":
        _report_moves((*values, *params), xp, dtype)
    return xp, [place(value, xp, dtype, device) for value in values]


def as_inexact(x: Any, xp: ModuleType | None = None) -> Any:
    """An array as it is if floating or complex, else in the default float.

    Unlike `floating_inputs`, it takes one array, and complex numbers. The
    default float is the namespace's: float64 on NumPy, and on JAX what
    ``jax_enable_x64`` says.

    Args:
        x: An array, a sequence or a number.
        xp: The namespace of the call; that of ``x`` if not given.

    Returns:
        An array of ``xp``, on ``x``'s device.

    Raises:
        BackendError: If ``x`` is not an array of a known library.
        ValueError: If ``x`` is not numeric, such as a date.

    Example:
        >>> import numpy as np
        >>> as_inexact(np.arange(3)).dtype, as_inexact([1j]).dtype
        (dtype('float64'), dtype('complex128'))
    """
    xp = namespace_of(x) if xp is None else xp
    array = xp.asarray(x) if is_host(x) else x
    if xp.isdtype(array.dtype, ("real floating", "complex floating")):
        return array
    if not xp.isdtype(array.dtype, ("bool", "integral")):
        msg = f"the input must be numeric, got dtype {array.dtype}"
        raise ValueError(msg)
    return xp.astype(array, xp.asarray(0.0).dtype)


def concrete(x: Any) -> np.ndarray | None:
    """The values of an array on the host, or ``None`` if it has none yet.

    For a check or a table that needs values and is skipped under ``jax.jit``.
    No gradient goes through the copy, and a dask array is computed.

    Args:
        x: An array, a sequence or a number.

    Returns:
        The values as a NumPy array, or ``None`` for a JAX tracer.

    Example:
        >>> concrete([1.0, 2.0])
        array([1., 2.])
    """
    if is_traced(x):
        return None
    if array_api_compat.is_torch_array(x):
        return x.detach().cpu().numpy()
    if array_api_compat.is_cupy_array(x):
        return x.get()  # CuPy refuses an implicit copy to the host
    return np.asarray(x)


def is_dataarray(value: Any) -> bool:
    """Say whether a value is an xarray DataArray, without importing xarray.

    Example:
        >>> is_dataarray(1.0)
        False
    """
    # a DataArray can only exist once xarray is imported
    xarray = sys.modules.get("xarray")
    return xarray is not None and isinstance(value, xarray.DataArray)


def _report_moves(values: tuple[Any, ...], xp: ModuleType, dtype: Any) -> None:
    """Tell the guard about the host arrays that join ``xp``, and a wider dtype.

    A number or a 0-d array is no move, and neither is a constant of a trace.
    """
    joining = [v for v in values if isinstance(v, (np.ndarray, list, tuple))]
    joining = [v for v in joining if np.ndim(v) > 0]
    others = [v for v in values if not is_host(v)]
    if not joining or not others or any(map(is_traced, others)):
        return
    library = xp.__name__.removeprefix("array_api_compat.").partition(".")[0]
    report_move("floating_inputs", f"NumPy data joined {library}")
    own = [v.dtype for v in others if xp.isdtype(v.dtype, "real floating")]
    if own and xp.result_type(*own) != dtype:
        narrow = xp.result_type(*own)
        report_move("floating_inputs", f"NumPy data widened {narrow} to {dtype}")
