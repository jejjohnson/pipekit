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
The named layer (the dim-level functions, `KernelOp`) joins NumPy to JAX in
the same way (ADR A24).

This module needs ``pipekit-array[numpy,compat]``, so importing
``pipekit_array`` does not import it. It imports no array library but NumPy.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any, cast

from pipekit_ops import (
    BackendError,
    get_adapter,
    guard_level,
    is_traced,
    report_move,
)

from pipekit_array.kernel import KernelSpec, full_loop, run_kernel


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
    "as_inexact",
    "boolean_like",
    "concrete",
    "constants_like",
    "divide_or_nan",
    "elementwise",
    "floating_inputs",
    "is_dataarray",
    "like",
    "namespace_of",
    "safe_atan2",
    "safe_power",
    "safe_sqrt",
    "squared_magnitude",
    "stand_in",
    "table_like",
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
    if guard_level() != "off":
        _report_moves((*values, *params), xp, dtype)
    return xp, [_place(value, xp, dtype, device) for value in values]


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
    array = xp.asarray(x) if _is_host(x) else x
    if xp.isdtype(array.dtype, ("real floating", "complex floating")):
        return array
    if not xp.isdtype(array.dtype, ("bool", "integral")):
        msg = f"the input must be numeric, got dtype {array.dtype}"
        raise ValueError(msg)
    return xp.astype(array, xp.asarray(0.0).dtype)


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
    device = _device_of(ref, xp)
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
        return _place(table, xp, ref.dtype, _device_of(ref, xp))
    return table_like(np.asarray(table), ref, xp) if _is_host(table) else table


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
    on_host = _is_host(value)
    if on_host:
        value = np.asarray(value)
    # a NumPy dtype is one of every namespace that NumPy arrays join
    namespace = numpy_namespace if on_host else xp
    if not namespace.isdtype(value.dtype, "bool"):
        msg = f"'{name}' must be boolean (True = use), got dtype {value.dtype}"
        raise ValueError(msg)
    return table_like(value, ref, xp) if on_host else value


def divide_or_nan(numerator: Any, denominator: Any, xp: ModuleType) -> Any:
    """The quotient of two arrays, NaN where the denominator is 0.

    It goes through a double ``where``, so a gradient is finite in the other
    elements: the division that is not used divides by 1.

    Args:
        numerator: An array of the namespace ``xp``.
        denominator: An array that broadcasts against it, of its dtype.
        xp: The namespace of the call.

    Returns:
        The quotient.

    Example:
        >>> import numpy as np
        >>> divide_or_nan(np.asarray([1.0, 1.0]), np.asarray([2.0, 0.0]), np)
        array([0.5, nan])
    """
    zero = denominator == 0
    safe = xp.where(zero, xp.ones_like(denominator), denominator)
    quotient = numerator / safe
    return xp.where(zero, xp.full_like(quotient, xp.nan), quotient)


def safe_sqrt(x: Any, xp: ModuleType) -> Any:
    """√x for x > 0, 0 with a zero gradient for x ≤ 0, and NaN for NaN.

    A difference of squares can round below 0, and its root is then 0; a missing
    value stays missing. It goes through a double ``where``, so the gradient is
    finite at 0.

    Args:
        x: A floating array of the namespace ``xp``.
        xp: The namespace of the call.

    Returns:
        The roots.

    Example:
        >>> import numpy as np
        >>> safe_sqrt(np.asarray([4.0, 0.0, -1.0, np.nan]), np)
        array([ 2.,  0.,  0., nan])
    """
    zero = x <= 0  # False for NaN
    # the root of 1 where the result is 0, so that the unused branch is finite
    root = xp.sqrt(xp.where(zero, xp.ones_like(x), x))
    return xp.where(zero, xp.zeros_like(x), root)


def safe_power(x: Any, p: Any, xp: ModuleType) -> Any:
    """xᵖ for x > 0, 0 with a zero gradient for x ≤ 0, and NaN for NaN.

    The rule at x ≤ 0 is that of `safe_sqrt`, and does not depend on p: it is 0
    for a negative p, too. The gradient by p is finite.

    Args:
        x: A floating array of the namespace ``xp``.
        p: The exponent, a number or an array that broadcasts against ``x``.
        xp: The namespace of the call.

    Returns:
        The powers.

    Example:
        >>> import numpy as np
        >>> safe_power(np.asarray([4.0, 0.0, -1.0, np.nan]), 1.5, np)
        array([ 8.,  0.,  0., nan])
    """
    zero = x <= 0
    power = xp.pow(xp.where(zero, xp.ones_like(x), x), constants_like(p, x, xp))
    return xp.where(zero, xp.zeros_like(power), power)


def safe_atan2(y: Any, x: Any, xp: ModuleType) -> Any:
    """atan2(y, x), and 0 with a zero gradient at the origin.

    Args:
        y: A floating array of the namespace ``xp``.
        x: An array that broadcasts against it, of its dtype.
        xp: The namespace of the call.

    Returns:
        The angles [rad], in (-π, π].

    Example:
        >>> import numpy as np
        >>> safe_atan2(np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0]), np).round(6)
        array([0.      , 1.570796])
    """
    origin = (x == 0) & (y == 0)
    # atan2(0, 1) = 0 at the origin, so that the unused branch is finite
    safe_x = xp.where(origin, xp.ones_like(x), x)
    safe_y = xp.where(origin, xp.zeros_like(y), y)
    return xp.atan2(safe_y, safe_x)


def squared_magnitude(z: Any, xp: ModuleType) -> Any:
    """|z|² = Re(z)² + Im(z)², with a gradient at 0, which that of |z| has not.

    Args:
        z: A real or complex array of the namespace ``xp``.
        xp: The namespace of the call.

    Returns:
        The squares, real.

    Example:
        >>> import numpy as np
        >>> squared_magnitude(np.asarray([3.0 + 4.0j, 0.0]), np)
        array([25.,  0.])
    """
    if not xp.isdtype(z.dtype, "complex floating"):
        return z * z
    return xp.real(z) ** 2 + xp.imag(z) ** 2


def stand_in(x: Any, mask: Any, xp: ModuleType, *, reference: Any = None) -> Any:
    """Rows of an array, with a valid one in place of each row to leave out.

    A function that is undefined at a row gives NaN there, and masking its result
    alone leaves 0 times NaN in the reverse pass, which spoils the gradient of every
    row. So it is evaluated somewhere valid instead, and its result masked.

    Args:
        x: The rows, (n, *rest), an array of the namespace ``xp``.
        mask: Which rows to use (True = use), boolean (n,), of the same library.
        xp: The namespace of the call.
        reference: The row, (*rest), to put in place of the others. If not given,
            the first one in use, or the first one of all if none is.

    Returns:
        The rows, in ``x``'s shape and dtype. No gradient reaches a row that is
        not in use, or the reference.

    Example:
        >>> import numpy as np
        >>> x = np.asarray([[np.nan, 0.0], [1.0, 2.0]])
        >>> stand_in(x, np.asarray([False, True]), np)
        array([[1., 2.],
               [1., 2.]])
    """
    if reference is None:
        first = xp.argmax(xp.astype(mask, xp.int32), keepdims=True)
        reference = xp.take(x, first, axis=0)[0, ...]
    reference = _constant(like(reference, x, xp), xp)
    return xp.where(xp.reshape(mask, (-1, *[1] * (x.ndim - 1))), x, reference)


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
           changed; another carrier keeps what its adapter's ``rewrap`` keeps,
           and its missing elements stay missing
           (`pipekit_ops.MissingValueAdapter`).

    An array without dimension names cannot be aligned with one that has
    them, so it is refused, unless it has no dimensions.

    Args:
        n_outputs: How many arrays the function returns; above 1, a tuple.

    Returns:
        The decorator.

    Raises:
        ValueError: (the decorated function) if a labelled array meets an
            array with dimensions but no names.
        BackendError: (the decorated function) if a DataArray meets a carrier
            of another kind, or carriers hold arrays of several libraries
            other than NumPy and JAX.

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
            given = [v for v in (*args, *kwargs.values()) if not _is_host(v)]
            if any(map(is_dataarray, given)):
                apply = _apply_xarray
            elif any(map(_is_carrier, given)):
                apply = _apply_carriers
            else:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            values = {**dict(enumerate(bound.args)), **bound.kwargs}
            return apply(_Call(function, signature, values), n_outputs)

        return cast(Any, wrapper)

    return decorate


# --- the host and the device ---


def _constant(x: Any, xp: ModuleType) -> Any:
    """An array that no gradient reaches."""
    if array_api_compat.is_jax_namespace(xp):
        import jax

        return jax.lax.stop_gradient(x)
    return x.detach() if array_api_compat.is_torch_namespace(xp) else x


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


def _report_moves(values: tuple[Any, ...], xp: ModuleType, dtype: Any) -> None:
    """Tell the guard about the host arrays that join ``xp``, and a wider dtype.

    A number or a 0-d array is no move, and neither is a constant of a trace.
    """
    joining = [v for v in values if isinstance(v, (np.ndarray, list, tuple))]
    joining = [v for v in joining if np.ndim(v) > 0]
    others = [v for v in values if not _is_host(v)]
    if not joining or not others or any(map(is_traced, others)):
        return
    library = xp.__name__.removeprefix("array_api_compat.").partition(".")[0]
    report_move("floating_inputs", f"NumPy data joined {library}")
    own = [v.dtype for v in others if xp.isdtype(v.dtype, "real floating")]
    if own and xp.result_type(*own) != dtype:
        narrow = xp.result_type(*own)
        report_move("floating_inputs", f"NumPy data widened {narrow} to {dtype}")


# --- labelled arrays ---


def _is_carrier(value: Any) -> bool:
    """Whether a value has a registered named-array adapter."""
    return not _is_host(value) and get_adapter(value) is not None


@dataclasses.dataclass(frozen=True, slots=True)
class _Call:
    """One call of ``function``, its arguments by position or keyword.

    A variadic argument is spread out, so each of its values has its own key.
    """

    function: Callable[..., Any]
    signature: inspect.Signature
    values: dict[int | str, Any]

    def name(self, key: int | str) -> str:
        """The parameter at a position, or of a keyword."""
        if isinstance(key, str):
            return key
        parameters = list(self.signature.parameters.values())
        for parameter in parameters[:key]:
            # it takes every position from its own on
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                return parameter.name
        return parameters[key].name

    def labelled(self, is_labelled: Callable[[Any], bool]) -> list[int | str]:
        """The keys of the labelled arguments, which must be all the arrays."""
        keys = [key for key, value in self.values.items() if is_labelled(value)]
        first, kind = self.name(keys[0]), type(self.values[keys[0]]).__name__
        for key, value in self.values.items():
            if key in keys:
                continue
            if _is_carrier(value):
                msg = (
                    f"'{self.name(key)}' is a {type(value).__name__} and '{first}' a "
                    f"{kind}; convert them to one kind first"
                )
                raise BackendError(msg)
            if _has_dimensions(value):
                msg = (
                    f"'{self.name(key)}' has dimensions but no names, and cannot be "
                    f"aligned with the {kind} '{first}'; make it a {kind} too"
                )
                raise ValueError(msg)
        return keys

    def on_data(self, keys: list[int | str]) -> Callable[..., Any]:
        """The function of the data of the arguments ``keys``."""

        def function(*data: Any) -> Any:
            given = self.values | dict(zip(keys, data, strict=True))
            args = [v for key, v in given.items() if isinstance(key, int)]
            kwargs = {key: v for key, v in given.items() if isinstance(key, str)}
            return self.function(*args, **kwargs)

        # the guard names the function the caller knows
        function.__name__ = getattr(self.function, "__name__", "elementwise")
        return function


def _apply_xarray(call: _Call, n_outputs: int) -> Any:
    """The call through ``xarray.apply_ufunc`` on its DataArrays."""
    # xarray is imported by now, so its adapter's module imports nothing new
    from pipekit_array.xarray import anonymous

    keys = call.labelled(is_dataarray)
    result = sys.modules["xarray"].apply_ufunc(
        full_loop(call.on_data(keys), _spec(len(keys), n_outputs)),
        *(call.values[key] for key in keys),
        output_core_dims=[()] * n_outputs,
        join="exact",
        dask="allowed",
        # for the coordinates: False would drop their attributes too
        keep_attrs="drop_conflicts",
    )
    if n_outputs == 1:
        return anonymous(result)
    return tuple(anonymous(array) for array in result)


def _apply_carriers(call: _Call, n_outputs: int) -> Any:
    """The call through `run_kernel` on its carriers."""
    keys = call.labelled(_is_carrier)
    carriers = [call.values[key] for key in keys]
    spec = _spec(len(keys), n_outputs)
    result = run_kernel(call.on_data(keys), spec, {}, carriers, {})
    keep = getattr(get_adapter(carriers[0]), "keep_missing", None)
    if keep is None:
        return result
    if n_outputs == 1:
        return keep(result, *carriers)
    return tuple(keep(out, *carriers) for out in result)


def _has_dimensions(value: Any) -> bool:
    """Whether a value is an array or a sequence, rather than a number or an option."""
    if isinstance(value, (list, tuple)):
        return True
    return getattr(value, "ndim", 0) > 0


@functools.lru_cache
def _spec(n_inputs: int, n_outputs: int) -> KernelSpec:
    inputs, outputs = ",".join(["()"] * n_inputs), ",".join(["()"] * n_outputs)
    return KernelSpec(f"{inputs}->{outputs}")
