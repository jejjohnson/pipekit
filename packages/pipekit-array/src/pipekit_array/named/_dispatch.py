"""The adapter, the array namespace and the dims of a call on named carriers.

`array_namespace` returns the Array API namespace its inputs share (numpy,
jax.numpy, torch via array-api-compat, …). It tries the optional
``array-api-compat`` first, because PyTorch tensors have no
``__array_namespace__`` of their own, and otherwise the native lookup
(ADR A1). `namespace_of` adds dask, which has no namespace of its own, and
`join_backends` joins NumPy to JAX (ADR A24). The rest of this module is what the
dim-level functions share: their axes, their masks, their fast paths.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from pipekit_ops import (
    NamedArrayAdapter,
    backend_name,
    get_adapter,
    guard_level,
    is_traced,
    report_move,
)
from pipekit_ops.adapters import BackendError


__all__ = ["array_namespace", "join_backends"]


_NO_BACKEND_MSG = (
    "No input implements the Array API. Install one of: "
    "pipekit-array[numpy], pipekit-array[jax], "
    "pipekit-array[torch], pipekit-array[cupy], "
    "pipekit-array[dask]. PyTorch additionally needs "
    "pipekit-array[compat] for namespace dispatch."
)


def _multi_backend_msg(names: list[str]) -> str:
    return (
        f"Inputs span multiple Array API backends: {names}. "
        "pipekit-array operators require a single backend per call."
    )


_UNSET: Any = object()


_compat: Any = _UNSET


def _compat_namespace() -> Callable[..., Any] | None:
    """``array_api_compat.array_namespace``, or None; looked up once.

    A failed import is not cached by Python, so trying it on every call
    cost about 0.2 ms per operator call (KernelOp's PK-K5 budget is 25 µs).
    """
    global _compat
    if _compat is _UNSET:
        try:
            import array_api_compat

            _compat = array_api_compat.array_namespace
        except ImportError:
            _compat = None
    return _compat


def array_namespace(*xs: Any) -> Any:
    """Return the Array API namespace shared by all input arrays.

    NumPy arrays next to JAX ones join them, so the namespace is JAX's.

    Args:
        *xs: One or more array-like objects. All must share the same
            Array API backend, but for NumPy with JAX.

    Returns:
        The module-like namespace object (e.g. `numpy`, `jax.numpy`,
        `array_api_compat.torch`).

    Raises:
        TypeError: If no input implements the Array API, or if the
            inputs span more than one backend. The error message is
            consistent whether or not `array-api-compat` is installed.
    """
    if len(xs) > 1:
        # ADR A24: NumPy joins JAX, whatever the order
        own = [x for x in xs if not type(x).__module__.startswith("numpy")]
        if own and all(backend_name(x) == "jax" for x in own):
            xs = tuple(own)
    compat_ns = _compat_namespace()

    if compat_ns is not None:
        try:
            return compat_ns(*xs)
        except TypeError as exc:
            # array-api-compat's error messages differ from our
            # contract; re-raise with stable wording so tests and
            # users see a consistent shape.
            msg = str(exc)
            if "Multiple namespaces" in msg:
                names = sorted({type(x).__module__.split(".")[0] for x in xs})
                raise TypeError(_multi_backend_msg(names)) from exc
            raise TypeError(_NO_BACKEND_MSG) from exc

    namespaces: set[Any] = set()
    for x in xs:
        ns = getattr(x, "__array_namespace__", None)
        if ns is None:
            raise TypeError(
                f"Input of type {type(x).__name__!r} does not implement "
                f"the Array API (`__array_namespace__` missing). " + _NO_BACKEND_MSG
            )
        namespaces.add(ns())
    if len(namespaces) == 0:
        raise TypeError(_NO_BACKEND_MSG)
    if len(namespaces) > 1:
        names = sorted({ns.__name__ for ns in namespaces})
        raise TypeError(_multi_backend_msg(names))
    return next(iter(namespaces))


#: a reduction: an array-namespace name ("mean") or a function taking axis=
Reduction = str | Callable[..., Any]


def adapter_of(x: Any, what: str) -> NamedArrayAdapter:
    """The adapter of a named carrier; a `TypeError` for an unnamed value."""
    adapter = get_adapter(x)
    if adapter is None:
        raise TypeError(
            f"{what} works on named carriers, got {type(x).__name__}; wrap a raw "
            "array as pipekit_array.DimArray(data, dims)."
        )
    return adapter


class _DaskNamespace:
    """``dask.array`` under the Array API names the generic paths call.

    dask has ``transpose`` and ``concatenate`` where the Array API has
    ``permute_dims`` and ``concat``; everything else passes through.
    """

    _NAMES: ClassVar[dict[str, str]] = {
        "permute_dims": "transpose",
        "concat": "concatenate",
        "cumulative_sum": "cumsum",
        "cumulative_prod": "cumprod",
    }

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, self._NAMES.get(name, name))

    def take_along_axis(self, data: Any, index: Any, axis: int) -> Any:
        """``np.take_along_axis`` per block, with ``axis`` in one chunk."""
        import numpy as np

        da = self._module
        data = da.asarray(data).rechunk({axis: -1})
        index = da.asarray(index)
        chunks = tuple(
            index.shape[k] if k == axis else data.chunks[k] for k in range(data.ndim)
        )
        index = index.rechunk(chunks)
        return da.map_blocks(
            np.take_along_axis,
            data,
            index,
            axis=axis,
            dtype=data.dtype,
            chunks=index.chunks,
        )


_DASK: list[_DaskNamespace] = []


def namespace_of(data: Any) -> Any:
    """The array namespace of raw data, dask included (which has none of its own)."""
    if type(data).__module__.startswith("dask."):
        if not _DASK:
            import dask.array as da

            _DASK.append(_DaskNamespace(da))
        return _DASK[0]
    return array_namespace(data)


def join_backends(datas: Sequence[Any], what: str) -> list[Any]:
    """The raw arrays on one backend: NumPy arrays next to JAX ones join them.

    The order of the inputs does not matter, and a NumPy array is a constant
    when a JAX one is traced (ADR A24). `pipekit_ops.guard` reports the arrays
    of one dimension or more that join eagerly, and a dtype they widen.

    Args:
        datas: Raw arrays.
        what: The function that joins them, for the guard and the errors.

    Raises:
        BackendError: Any other mix: dask with NumPy or JAX, or another
            library with JAX.

    Example:
        >>> import numpy as np
        >>> first, second = join_backends([np.ones(2), np.ones(3)], "stack")
        >>> first.shape, second.shape
        ((2,), (3,))
    """
    joined = list(datas)
    backends = {backend_name(d) for d in joined}
    if len(backends) < 2:
        return joined
    strangers = [d for d in joined if backend_name(d) == "numpy" and not _is_numpy(d)]
    if backends != {"numpy", "jax"} or strangers:
        names = sorted(backends - {"numpy"} | {_library(d) for d in joined})
        raise BackendError(
            f"{what}: inputs on {names}; put them on one backend first (NumPy joins "
            "JAX, and nothing else moves)."
        )
    import jax.numpy as jnp

    if guard_level() != "off":
        _report_join(joined, what)
    # a Python number stays one: JAX gives it the arrays' dtype
    numbers = (bool, int, float, complex)
    return [
        d if backend_name(d) == "jax" or type(d) in numbers else jnp.asarray(d)
        for d in joined
    ]


def _library(data: Any) -> str:
    """The backend's name, or the library's for a type pipekit has no name for."""
    name = backend_name(data)
    if name != "numpy" or _is_numpy(data):
        return name
    return type(data).__module__.partition(".")[0]


def _is_numpy(data: Any) -> bool:
    """Whether ``data`` is NumPy's, or a Python number or sequence."""
    return type(data).__module__.partition(".")[0] in ("numpy", "builtins")


def _report_join(datas: Sequence[Any], what: str) -> None:
    """Tell the guard about the NumPy arrays that join, and a dtype they widen."""
    import jax.numpy as jnp

    own = [d for d in datas if backend_name(d) == "jax"]
    joining = [d for d in datas if backend_name(d) != "jax" and getattr(d, "ndim", 0)]
    # a NumPy constant of a trace is compiled in, not moved on each call
    if not joining or any(map(is_traced, own)):
        return
    report_move(what, "NumPy data joined jax")
    narrow, wide = jnp.result_type(*own), jnp.result_type(*own, *joining)
    if narrow != wide and jnp.issubdtype(narrow, jnp.inexact):
        report_move(what, f"NumPy data widened {narrow} to {wide}")


def join_mask(
    adapter: NamedArrayAdapter, x: Any, data: Any, mask: Any, what: str
) -> tuple[Any, Any, Any]:
    """``x``, its data and its mask, with NumPy joined to JAX if either is JAX.

    A mask of another backend stays as it is (a NumPy mask on dask data).
    """
    if "jax" not in (backend_name(data), backend_name(mask)):
        return x, data, mask
    own, mask = join_backends([data, mask], what)
    if own is not data:
        x = adapter.rewrap(x, own, adapter.dims(x))
    return x, own, mask


def aligned_mask(where: Any, dims: tuple[str, ...], data: Any, what: str) -> Any:
    """``where`` as a bool array with ``data``'s shape, aligned by dim name.

    The mask stays in its own array namespace (a NumPy mask on dask data,
    a traced one under ``jit``).
    """
    shape = tuple(data.shape)
    mask_adapter = get_adapter(where)
    if mask_adapter is not None:
        mask_dims = tuple(mask_adapter.dims(where))
        extra = [d for d in mask_dims if d not in dims]
        if extra:
            raise ValueError(f"{what}: where= has dim(s) {extra} not in {dims}.")
        mask = mask_adapter.data(where)
        xp = namespace_of(mask)
        sizes = dict(zip(mask_dims, mask.shape, strict=True))
        wrong = {
            d: (sizes[d], shape[i])
            for i, d in enumerate(dims)
            if d in sizes and sizes[d] not in (1, shape[i])
        }
        if wrong:
            detail = ", ".join(f"{d}: {m} vs {n}" for d, (m, n) in wrong.items())
            raise ValueError(f"{what}: where= sizes differ from the data's ({detail}).")
        order = [d for d in dims if d in mask_dims]
        mask = xp.permute_dims(mask, tuple(mask_dims.index(d) for d in order))
        # the mask's own lengths (1 broadcasts), never the data's
        expanded = tuple(sizes.get(d, 1) for d in dims)
    else:
        mask = where
        xp = namespace_of(mask)
        if tuple(mask.shape) != shape[: mask.ndim]:
            raise ValueError(
                f"{what}: a plain where= of shape {tuple(mask.shape)} must match the "
                f"leading dims {dims[: mask.ndim]} of shape {shape[: mask.ndim]}; pass "
                "a named mask to align by dim name."
            )
        expanded = (*mask.shape, *(1,) * (len(shape) - mask.ndim))
    mask = xp.reshape(xp.astype(mask, xp.bool), expanded)
    return xp.broadcast_to(mask, shape)


def fast_path(adapter: NamedArrayAdapter, name: str) -> Callable[..., Any] | None:
    return getattr(adapter, name, None)


def axes_of(dims: tuple[str, ...], names: Sequence[str], what: str) -> tuple[int, ...]:
    missing = [n for n in names if n not in dims]
    if missing:
        raise KeyError(f"{what}: unknown dim(s) {missing}; dims are {dims}.")
    return tuple(dims.index(n) for n in names)


def resolve_function(fn: Reduction, xp: Any, what: str) -> Callable[..., Any]:
    if callable(fn):
        return fn
    resolved = getattr(xp, fn, None)
    if resolved is None:
        raise ValueError(f"{what}: the array namespace has no {fn!r}.")
    return resolved


def check_index_labels(
    adapter: NamedArrayAdapter,
    x: Any,
    index_adapter: NamedArrayAdapter,
    index: Any,
    dims: Sequence[str],
) -> None:
    """`take_along`'s check: on each shared dim the index's labels are x's.

    Raises:
        ValueError: Labels differ in value or order (the index is positional).
    """
    for d in dims:
        mine, theirs = adapter.labels(x, d), index_adapter.labels(index, d)
        if mine is not None and theirs is not None and tuple(mine) != tuple(theirs):
            raise ValueError(
                f"take_along: the index's {d!r} labels {tuple(theirs)} are not x's "
                f"{tuple(mine)}; the index is applied by position, so they must match."
            )
