"""Let an elementwise function of plain arrays take and return labelled arrays."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import sys
from collections.abc import Callable
from typing import Any, cast

from pipekit_ops import BackendError, get_adapter

from pipekit_array.arrays._backend import is_dataarray
from pipekit_array.arrays._host import is_host
from pipekit_array.kernels._run import anonymous, full_loop, run_kernel
from pipekit_array.kernels._spec import KernelSpec


__all__ = ["elementwise"]


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
            given = [v for v in (*args, *kwargs.values()) if not is_host(v)]
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


def _is_carrier(value: Any) -> bool:
    """Whether a value has a registered named-array adapter."""
    return not is_host(value) and get_adapter(value) is not None


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
