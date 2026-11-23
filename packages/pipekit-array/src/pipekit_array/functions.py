"""The array backend for plain functions: any array library, and labelled arrays.

A function that is no Operator (a formula of a physics library) is written
once, in functions of the array API standard, and runs on NumPy, JAX, torch,
CuPy, dask and any other library that `array_api_compat` knows:

    @elementwise()                                       # labelled arrays
    def density(pressure, temperature):
        xp, (p, t) = floating_inputs(pressure, temperature)  # one dtype, device
        return p / (like(287.0, t, xp) * t)

The host rule (ADR A23): Python numbers, sequences and NumPy arrays live on
the host, and join the library and the device of the other arrays. No other
array is moved, and two libraries other than NumPy in one call raise.
`pipekit_array.array_namespace` and the named layer (the dim-level functions,
`KernelOp`) keep their stricter rule, which refuses NumPy next to JAX.

This module needs ``pipekit-array[numpy,compat]``, so importing
``pipekit_array`` does not import it. It imports no array library but NumPy.
"""

from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any, cast

from pipekit_ops import BackendError, get_adapter

from pipekit_array.kernel import KernelSpec, run_kernel


try:
    import array_api_compat
    import array_api_compat.numpy as numpy_namespace
    import numpy as np
except ImportError as err:
    msg = (
        "pipekit_array.functions needs NumPy and array-api-compat: "
        "install 'pipekit-array[numpy,compat]'"
    )
    raise ImportError(msg) from err


__all__ = [
    "BackendError",
    "elementwise",
    "floating_inputs",
    "is_dataarray",
    "like",
    "namespace_of",
]

# Values that live on the host and join whatever library the others are from.
_HOST_TYPES = (bool, int, float, complex, np.ndarray, np.generic, list, tuple)


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
        if _is_host(value):
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
    devices = (_device_of(value, xp) for value in (*values, *params))
    device = next((found for found in devices if found is not None), None)
    floating = []
    for value in values:
        if _is_number(value):
            continue
        dtype = xp.asarray(value).dtype if _is_host(value) else value.dtype
        if xp.isdtype(dtype, "real floating"):
            floating.append(dtype)
        elif not xp.isdtype(dtype, ("bool", "integral")):
            msg = f"inputs must be real numbers, got dtype {dtype}"
            raise ValueError(msg)
    dtype = xp.result_type(*floating) if floating else xp.asarray(0.0).dtype
    return xp, [_place(value, xp, dtype, device) for value in values]


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
    return _place(value, xp, ref.dtype, _device_of(ref, xp))


def is_dataarray(value: Any) -> bool:
    """Say whether a value is an xarray DataArray, without importing xarray.

    Example:
        >>> is_dataarray(1.0)
        False
    """
    # a DataArray can only exist once xarray is imported
    xarray = sys.modules.get("xarray")
    return xarray is not None and isinstance(value, xarray.DataArray)


def elementwise[F: Callable[..., Any]](*, n_outputs: int = 1) -> Callable[[F], F]:
    """Let an elementwise function take and return labelled arrays.

    Algorithm:
        1. With no labelled array among the arguments, call the function as
           it is.
        2. Hand DataArrays to ``xarray.apply_ufunc``, and other carriers with
           a registered adapter (`DimArray`, geoarray's ``GeoArray``) to the
           path of `KernelOp`. Both align by dimension name and call the
           function on the data. The other arguments are passed through.
        3. Broadcast every result to the dimensions of the inputs together.
        4. Return labelled arrays with the inputs' coordinates. A DataArray
           has no name or attributes, because the quantity and its unit have
           changed; another carrier keeps what its adapter's ``rewrap`` keeps.

    An array without dimension names cannot be aligned with one that has
    them, so it is refused, unless it has no dimensions.

    Args:
        n_outputs: How many arrays the function returns; above 1, a tuple.

    Returns:
        The decorator, for a function without ``*args`` and ``**kwargs``.

    Raises:
        ValueError: (the decorated function) if a labelled array meets an
            array with dimensions but no names.
        BackendError: (the decorated function) if a DataArray meets a carrier
            of another kind, or carriers hold arrays of several libraries.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> @elementwise()
        ... def scaled(x, slope):
        ...     return slope * x
        >>> scaled(np.asarray([1.0, 2.0]), 2.0).tolist()
        [2.0, 4.0]
        >>> scaled(DimArray(np.asarray([1.0, 2.0]), ("x",)), 2.0).dims
        ('x',)
    """

    def decorate(function: F) -> F:
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            values = [v for v in (*args, *kwargs.values()) if not _is_host(v)]
            if any(map(is_dataarray, values)):
                bound = signature.bind(*args, **kwargs)
                return _apply_xarray(function, bound, n_outputs)
            if any(map(_is_carrier, values)):
                bound = signature.bind(*args, **kwargs)
                return _apply_carriers(function, bound, n_outputs)
            return function(*args, **kwargs)

        return cast(Any, wrapper)

    return decorate


# --- the host and the device ---


def _is_host(value: Any) -> bool:
    """Whether a value is ``None``, a Python number, a sequence or NumPy's."""
    return value is None or isinstance(value, _HOST_TYPES)


def _is_number(value: Any) -> bool:
    """A Python real number, which adopts the dtype of the arrays.

    A NumPy scalar carries its own dtype, though ``np.float64`` is a float.
    """
    return isinstance(value, (bool, int, float)) and not isinstance(value, np.generic)


def _device_of(value: Any, xp: ModuleType) -> Any:
    """The device that new arrays join ``value`` on, or ``None`` for the default.

    A JAX tracer has no device, and an uncommitted JAX scalar joins an array
    on any device, so JAX is left to place its own arrays.
    """
    if _is_host(value) or array_api_compat.is_jax_namespace(xp):
        return None
    return array_api_compat.device(value)


def _place(value: Any, xp: ModuleType, dtype: Any, device: Any) -> Any:
    """``value`` as an array of ``dtype``: cast if it is one already, else built.

    ``device`` is where a host value goes; an array stays on its own.
    """
    if not _is_host(value):
        # asarray would detach a torch tensor from its graph (torch 2.5, 2.10)
        return xp.astype(value, dtype, copy=False)
    if device is None:
        return xp.asarray(value, dtype=dtype)
    return xp.asarray(value, dtype=dtype, device=device)


# --- labelled arrays ---


def _is_carrier(value: Any) -> bool:
    """Whether a value has a registered named-array adapter."""
    return not _is_host(value) and get_adapter(value) is not None


def _apply_xarray(
    function: Callable[..., Any], bound: inspect.BoundArguments, n_outputs: int
) -> Any:
    """Call ``function`` through ``xarray.apply_ufunc`` on its DataArrays."""
    names = _labelled_names(bound, is_dataarray)
    result = sys.modules["xarray"].apply_ufunc(
        _on_data(function, bound, names, n_outputs),
        *(bound.arguments[name] for name in names),
        output_core_dims=[()] * n_outputs,
        join="exact",
        dask="allowed",
        # for the coordinates: False would drop their attributes too
        keep_attrs="drop_conflicts",
    )
    if n_outputs == 1:
        return _anonymous(result)
    return tuple(_anonymous(array) for array in result)


def _anonymous(array: Any) -> Any:
    """A DataArray without its name and attributes, which described the input."""
    array = array.rename(None)
    array.attrs = {}
    return array


def _apply_carriers(
    function: Callable[..., Any], bound: inspect.BoundArguments, n_outputs: int
) -> Any:
    """Call ``function`` through `run_kernel` on its carriers."""
    names = _labelled_names(bound, _is_carrier)
    carriers = [bound.arguments[name] for name in names]
    kernel = _on_data(function, bound, names, n_outputs)
    return run_kernel(kernel, _spec(len(names), n_outputs), {}, carriers, {})


def _labelled_names(
    bound: inspect.BoundArguments, is_labelled: Callable[[Any], bool]
) -> list[str]:
    """The names of the labelled arguments, which must be all the arrays there are."""
    names = [name for name, value in bound.arguments.items() if is_labelled(value)]
    first, kind = names[0], type(bound.arguments[names[0]]).__name__
    for name, value in bound.arguments.items():
        if name in names:
            continue
        if _is_carrier(value):
            msg = (
                f"'{name}' is a {type(value).__name__} and '{first}' a {kind}; "
                "convert them to one kind first"
            )
            raise BackendError(msg)
        if _has_dimensions(value):
            msg = (
                f"'{name}' has dimensions but no names, and cannot be aligned with "
                f"the {kind} '{first}'; make it a {kind} too"
            )
            raise ValueError(msg)
    return names


def _on_data(
    function: Callable[..., Any],
    bound: inspect.BoundArguments,
    names: list[str],
    n_outputs: int,
) -> Callable[..., Any]:
    """``function`` of the labelled arguments' data, with its results broadcast."""

    def on_data(*data: Any) -> Any:
        bound.arguments.update(zip(names, data, strict=True))
        values = function(*bound.args, **bound.kwargs)
        shape = np.broadcast_shapes(*(array.shape for array in data))
        if n_outputs == 1:
            return _broadcast(values, shape)
        return tuple(_broadcast(value, shape) for value in values)

    return on_data


def _has_dimensions(value: Any) -> bool:
    """Whether a value is an array or a sequence, rather than a number or an option."""
    if isinstance(value, (list, tuple)):
        return True
    return getattr(value, "ndim", 0) > 0


def _broadcast(array: Any, shape: tuple[int, ...]) -> Any:
    """``array`` in the shape of the inputs together.

    An output may not depend on every input: z of a position on a sphere does
    not depend on the longitude.
    """
    if array.shape == shape:
        return array
    return namespace_of(array).broadcast_to(array, shape)


@functools.lru_cache
def _spec(n_inputs: int, n_outputs: int) -> KernelSpec:
    inputs, outputs = ",".join(["()"] * n_inputs), ",".join(["()"] * n_outputs)
    return KernelSpec(f"{inputs}->{outputs}")
