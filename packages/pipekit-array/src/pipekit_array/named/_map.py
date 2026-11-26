"""A function on each item along a dim.

Each is written once, over the carrier's registered
`pipekit_ops.NamedArrayAdapter`; see `pipekit_array.named` for the rules they share.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pipekit_ops import NamedArrayAdapter, is_traced
from pipekit_ops.carriers import BackendError, backend_name

from pipekit_array.named._dispatch import adapter_of, axes_of, fast_path
from pipekit_array.named._join import _with_labels_of, stack


__all__ = ["map_over"]


def map_over(x: Any, fn: Callable[[Any], Any], dim: str) -> Any:
    """``fn`` on each item along ``dim``; the results stacked along ``dim``, first.

    Each item is ``isel(x, dim=i)``; ``fn`` returns a named carrier, and every
    result must have the same dims.

    On JAX data this is one ``jax.vmap``, for a carrier whose only array is its
    data (an adapter with more has a fast path of its own). A ``fn`` that cannot
    be traced (it branches on a value, or runs on NumPy only) is called on each
    item in a loop instead, unless ``x`` is traced itself.

    Raises:
        KeyError: ``dim`` is not a dim of ``x``.
        TypeError: ``fn`` returns something other than a named carrier.
        ValueError: The results have different dims, or have ``dim`` already.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray, reduce
        >>> cube = DimArray(np.ones((3, 2, 4)), ("time", "band", "x"))
        >>> map_over(cube, lambda item: reduce(item, "sum", "band"), "time").dims
        ('time', 'x')
    """
    adapter = adapter_of(x, "map_over")
    fast = fast_path(adapter, "map_over")
    if fast is not None:
        return fast(x, fn, dim)
    dims = adapter.dims(x)
    (axis,) = axes_of(dims, (dim,), "map_over")
    if backend_name(adapter.data(x)) == "jax" and _only_array(adapter, x):
        mapped = _map_over_vmap(adapter, x, fn, dim, axis)
        if mapped is not None:
            return mapped
    size = adapter.data(x).shape[axis]
    items = [fn(adapter.isel(x, {dim: i})) for i in range(size)]
    return _with_labels_of(adapter, x, stack(items, dim), dim)


def _only_array(adapter: NamedArrayAdapter, x: Any) -> bool:
    """Whether the carrier is a JAX pytree whose one leaf is its data."""
    import jax

    leaves = jax.tree_util.tree_leaves(x)
    return len(leaves) == 1 and leaves[0] is adapter.data(x)


def _map_over_vmap(
    adapter: NamedArrayAdapter, x: Any, fn: Callable[[Any], Any], dim: str, axis: int
) -> Any:
    """`map_over` as one ``jax.vmap``; ``None`` if ``fn`` has to be looped over."""
    import jax

    dims = adapter.dims(x)
    item = adapter.isel(x, {dim: 0})
    item_dims = tuple(d for d in dims if d != dim)
    traced: list[Any] = []

    def one(values: Any) -> Any:
        out = fn(adapter.rewrap(item, values, item_dims))
        traced[:] = [out]
        return adapter_of(out, "map_over").data(out)

    data = jax.numpy.moveaxis(adapter.data(x), axis, 0)
    try:
        mapped = jax.vmap(one)(data)
    except (jax.errors.JAXTypeError, jax.errors.JAXIndexError, BackendError):
        if is_traced(data):
            raise
        return None
    # what one item's result is, but for its data
    (template,) = traced
    out_adapter = adapter_of(template, "map_over")
    if not _only_array(out_adapter, template):
        return None
    out_dims = out_adapter.dims(template)
    if dim in out_dims:
        raise ValueError(f"map_over: fn's result has {dim!r} already: {out_dims}.")
    out = out_adapter.rewrap(template, mapped, (dim, *out_dims))
    return _with_labels_of(adapter, x, out, dim)
