"""The kinds of array a function must accept, and the check that they agree.

The union of the checks of geophysix, radiomax and diagnostix
(``tests/_backends.py``).
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import importlib.util
import unittest
from collections.abc import Callable, Collection
from typing import Any

import numpy as np
from pipekit_ops import BackendError

from pipekit_array.testing._oracles import assert_matches_oracle
from pipekit_array.testing._transforms import require_jax


__all__ = [
    "ArrayKind",
    "array_kinds",
    "assert_backends_agree",
    "assert_refuses_dataarrays",
]


@dataclasses.dataclass(frozen=True, slots=True)
class ArrayKind:
    """One kind of array: how to make it, what comes back, and how to read it.

    Attributes:
        name: What it is called in a message, and in ``kinds=``.
        make: An array of this kind from a NumPy one.
        type: The class, or classes, that a result must have.
        host: The values of a result as NumPy; it checks the device.
    """

    name: str
    make: Callable[[np.ndarray], Any]
    type: Any
    host: Callable[[Any], np.ndarray] = np.asarray

    def read(self, array: Any) -> np.ndarray:
        """The array as NumPy, after checking that it is of this kind.

        Raises:
            AssertionError: If it is of another kind, or on another device.

        Example:
            >>> array_kinds()[0].read(np.ones(2))
            array([1., 1.])
        """
        if not isinstance(array, self.type):
            raise AssertionError(
                f"{self.name} in must give {self.name} out, got {type(array).__name__}"
            )
        return self.host(array)


@functools.cache
def array_kinds() -> tuple[ArrayKind, ...]:
    """The kinds of array whose library is installed.

    The call imports those libraries. In this order: ``numpy``; ``strict`` and
    ``second device``, `array_api_strict` on its default device and on another
    one; ``jax``; ``dask``; ``torch``; ``xarray``.

    Returns:
        The kinds, for ``pytest.mark.parametrize`` and for ``kinds=``.

    Example:
        >>> array_kinds()[0].name
        'numpy'
    """
    kinds = [ArrayKind("numpy", np.asarray, (np.ndarray, np.generic))]
    if _installed("array_api_strict"):
        kinds += _strict_kinds()
    if _installed("jax"):
        import jax

        kinds.append(ArrayKind("jax", jax.numpy.asarray, jax.Array))
    if _installed("dask"):
        import dask.array as da

        kinds.append(ArrayKind("dask", da.from_array, da.Array))
    if _installed("torch"):
        # torch is in no dependency group, so a type checker does not see it
        torch = importlib.import_module("torch")

        def detached(array: Any) -> np.ndarray:
            return array.detach().cpu().numpy()

        kinds.append(ArrayKind("torch", torch.tensor, torch.Tensor, detached))
    if _installed("xarray"):
        import xarray

        kinds.append(ArrayKind("xarray", _labelled, xarray.DataArray))
    return tuple(kinds)


def assert_backends_agree(
    fn: Callable[..., Any],
    *args: Any,
    rtol: float = 1e-12,
    atol: float = 0.0,
    rtol32: float = 1e-5,
    atol32: float = 0.0,
    kinds: Collection[str] | None = None,
    labelled: bool = False,
) -> None:
    """Assert that a function gives NumPy's numbers on every kind of array.

    For each kind: the same kind comes back, on the same device; float64 values
    agree with NumPy's to ``rtol`` and ``atol``; and float32 in gives float32 out,
    which agrees with NumPy's float64 to ``rtol32`` and ``atol32``. JAX is also
    checked under ``jit``. Boolean and integer arguments keep their dtype, and
    boolean and integer results must match exactly.

    Args:
        fn: The function under test; it returns an array or a tuple of them.
        *args: NumPy arrays to call it with; the floating ones are float64.
        rtol: Relative tolerance in float64.
        atol: Absolute tolerance in float64, for results that cross zero.
        rtol32: Relative tolerance of float32 against float64.
        atol32: Absolute tolerance of float32 against float64.
        kinds: The names of the kinds to check; one that is not installed is left
            out. ``None`` checks every kind but ``xarray``.
        labelled: Also check DataArrays, for an elementwise function. Their
            dimensions are named from the right, so they broadcast as NumPy does.

    Raises:
        AssertionError: If a check fails; the message names the kind, the dtype
            and the worst index. Also if JAX is checked without its 64-bit mode.
        ValueError: If ``kinds`` holds a name that is no kind.

    Example:
        >>> from pipekit_array.arrays import floating_inputs
        >>> def root(x):
        ...     xp, (x,) = floating_inputs(x)
        ...     return xp.sqrt(x)
        >>> assert_backends_agree(root, np.arange(3.0), kinds=["numpy", "strict"])
    """
    unknown = set(kinds or ()) - set(_NAMES)
    if unknown:
        msg = f"no such kind of array: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    arrays = [np.asarray(arg) for arg in args]
    expected = _outputs(fn(*arrays))
    checks = ((np.float64, rtol, atol), (np.float32, rtol32, atol32))
    for kind in array_kinds():
        chosen = kind.name != "xarray" if kinds is None else kind.name in kinds
        if not chosen and not (labelled and kind.name == "xarray"):
            continue
        variants = {kind.name: fn}
        if kind.name == "jax":
            variants["jax.jit"] = require_jax().jit(fn)
        for name, variant in variants.items():
            for dtype, relative, absolute in checks:
                label = f"{name} {np.dtype(dtype).name}"
                results = _outputs(
                    variant(*(kind.make(_cast(array, dtype)) for array in arrays))
                )
                if len(results) != len(expected):
                    raise AssertionError(
                        f"{label}: {len(results)} results, {len(expected)} on NumPy"
                    )
                for got, want in zip(results, expected, strict=True):
                    _compare(kind, got, want, dtype, label, relative, absolute)


def assert_refuses_dataarrays(fn: Callable[..., Any], *args: Any) -> None:
    """Assert that a DataArray in any argument raises `BackendError`.

    A function over a grid or an axis takes plain arrays, so that labels are never
    lost silently. Skips without xarray.

    Args:
        fn: The function under test.
        *args: NumPy arrays to call it with; each is labelled in turn.

    Raises:
        AssertionError: If a call is accepted; the message names the argument.

    Example:
        >>> from pipekit_array.arrays import floating_inputs
        >>> def plain(x):
        ...     _, (x,) = floating_inputs(x)
        ...     return x
        >>> assert_refuses_dataarrays(plain, np.ones(2))
    """
    if not _installed("xarray"):
        raise unittest.SkipTest("xarray is not installed")
    for position, arg in enumerate(args):
        inputs = list(args)
        inputs[position] = _labelled(np.asarray(arg))
        try:
            fn(*inputs)
        except BackendError:
            continue
        raise AssertionError(f"argument {position} accepted a DataArray")


# --- the kinds ---

_NAMES = ("numpy", "strict", "second device", "jax", "dask", "torch", "xarray")


def _installed(library: str) -> bool:
    return importlib.util.find_spec(library) is not None


def _strict_kinds() -> list[ArrayKind]:
    """`array_api_strict`, on its default device and on one that is not."""
    import array_api_strict as strict

    strict_type = type(strict.asarray(0.0))
    second, cpu = strict.Device("device1"), strict.Device("CPU_DEVICE")

    def make(array: np.ndarray) -> Any:
        return strict.asarray(array, device=second)

    def host(array: Any) -> np.ndarray:
        if array.device != second:
            raise AssertionError(f"the result is on {array.device}, not on {second}")
        return np.asarray(array.to_device(cpu))

    return [
        ArrayKind("strict", strict.asarray, strict_type),
        ArrayKind("second device", make, strict_type, host),
    ]


def _labelled(array: np.ndarray) -> Any:
    """A DataArray, its dimensions named from the right."""
    import xarray

    dims = tuple(f"dim{array.ndim - axis}" for axis in range(array.ndim))
    return xarray.DataArray(array, dims=dims)


# --- one result ---


def _cast(array: np.ndarray, dtype: Any) -> np.ndarray:
    """A floating array in ``dtype``; a boolean or integer one as it is."""
    return array.astype(dtype) if np.issubdtype(array.dtype, np.floating) else array


def _outputs(result: Any) -> tuple[Any, ...]:
    return result if isinstance(result, tuple) else (result,)


def _compare(
    kind: ArrayKind,
    got: Any,
    want: Any,
    dtype: Any,
    label: str,
    rtol: float,
    atol: float,
) -> None:
    try:
        values = kind.read(got)
        if np.issubdtype(np.asarray(want).dtype, np.floating) and values.dtype != dtype:
            raise AssertionError(f"{values.dtype} came back")
        assert_matches_oracle(values, want, rtol=rtol, atol=atol)
    except AssertionError as err:
        raise AssertionError(f"{label}: {err}") from err
