"""Join carriers: on a new dim, along an existing one, or entry by entry.

Each is written once, over the carrier's registered
`pipekit_ops.NamedArrayAdapter`; see `pipekit_array.named` for the rules they share.

`stack_items` is for items that may each carry their own geometry (patches,
per-frame grids); it is `stack` where that is the same thing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pipekit_ops import NamedArrayAdapter

from pipekit_array.named._dispatch import (
    adapter_of,
    axes_of,
    fast_path,
    join_backends,
    namespace_of,
)


__all__ = ["combine", "concat", "stack", "stack_items"]


def _with_labels_of(adapter: NamedArrayAdapter, x: Any, out: Any, dim: str) -> Any:
    """``out`` with the labels ``x`` has on ``dim``, where its adapter can set them.

    Each output position of `map_over` is the input's.
    """
    labels = adapter.labels(x, dim)
    out_adapter = adapter_of(out, "map_over")
    relabel = getattr(out_adapter, "relabel", None)
    if labels is None or relabel is None or out_adapter.labels(out, dim) is not None:
        return out
    return relabel(out, {dim: labels})


def stack(xs: Sequence[Any], dim: str) -> Any:
    """A new leading dim ``dim`` from carriers with the same dims; never aligns."""
    if len(xs) == 0:
        raise ValueError("stack needs at least one carrier.")
    adapter = adapter_of(xs[0], "stack")
    fast = fast_path(adapter, "stack")
    if fast is not None:
        return fast(list(xs), dim)
    checked = adapter.check_compatible(*xs)
    dims = _same_dims(adapter, checked, "stack")
    if dim in dims:
        raise ValueError(f"stack: {dim!r} is already a dim of {dims}; use concat.")
    datas = join_backends([adapter.data(x) for x in checked], "stack")
    xp = namespace_of(datas[0])
    return adapter.rewrap(checked[0], xp.stack(datas), (dim, *dims))


def stack_items(xs: Sequence[Any], dim: str) -> Any:
    """Items along a new leading dim ``dim``, each keeping its own geometry.

    The inverse of the split `map_over` makes: where `stack` requires one
    shared grid (geoarray's `stack` never aligns two), ``stack_items`` lets
    each item carry its own per-item leaves (a patch's transform row, a
    frame's grid) and makes ``dim`` a batch dim. A carrier family with no
    per-item geometry (`DimArray`, ``xr.DataArray``) has nothing to keep
    apart, so this is `stack`; geoarray's adapter provides the
    ``stack_items`` fast path.

    Raises:
        ValueError: ``xs`` is empty, or the items' dims differ.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> items = [DimArray(np.ones((2, 3)), ("y", "x")) for _ in range(4)]
        >>> stack_items(items, "sample").dims
        ('sample', 'y', 'x')
    """
    if len(xs) == 0:
        raise ValueError("stack_items needs at least one carrier.")
    adapter = adapter_of(xs[0], "stack_items")
    fast = fast_path(adapter, "stack_items")
    if fast is not None:
        return fast(list(xs), dim)
    return stack(xs, dim)


def concat(xs: Sequence[Any], dim: str) -> Any:
    """Join carriers along an existing dim; the other dims must agree.

    The generic path drops ``dim``'s labels (the joined dim has a new size);
    an adapter's ``concat`` fast path may join them.
    """
    if len(xs) == 0:
        raise ValueError("concat needs at least one carrier.")
    adapter = adapter_of(xs[0], "concat")
    fast = fast_path(adapter, "concat")
    if fast is not None:
        return fast(list(xs), dim)
    dims = _same_dims(adapter, xs, "concat")
    (axis,) = axes_of(dims, (dim,), "concat")
    # the other dims must combine; dim itself may differ in size
    # the other dims must combine; an empty slice of dim checks them even
    # when an input has no entries along dim
    others = [adapter.isel(x, {dim: slice(0, 0)}) for x in xs]
    adapter.check_compatible(*others)
    datas = join_backends([adapter.data(x) for x in xs], "concat")
    xp = namespace_of(datas[0])
    return adapter.rewrap(xs[0], xp.concat(datas, axis=axis), dims)


def combine(fn: Callable[..., Any], *xs: Any) -> Any:
    """``fn`` of the data of carriers with the same dims; never aligns.

    Inputs are put in the first one's dim order by name before ``fn`` sees
    them, and the result has those dims. NumPy inputs join JAX ones; ``fn``
    then decides what it returns (``np.add`` of JAX arrays is NumPy's).

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> nir = DimArray(np.full((2, 3), 0.4), ("y", "x"))
        >>> red = DimArray(np.full((2, 3), 0.1), ("y", "x"))
        >>> nd = combine(lambda a, b: (a - b) / (a + b), nir, red)
        >>> nd.dims, bool(np.allclose(nd.data, 0.6))
        (('y', 'x'), True)
    """
    if len(xs) == 0:
        raise ValueError("combine needs at least one carrier.")
    adapter = adapter_of(xs[0], "combine")
    fast = fast_path(adapter, "combine")
    if fast is not None:
        return fast(fn, *xs)
    checked = adapter.check_compatible(*xs)
    joined = join_backends([adapter.data(x) for x in checked], "combine")
    dims = adapter.dims(checked[0])
    datas = []
    for x, data in zip(checked, joined, strict=True):
        other = adapter.dims(x)
        if sorted(other) != sorted(dims):
            raise ValueError(f"combine: dims {other} differ from {dims}.")
        if other != dims:
            data = namespace_of(data).permute_dims(
                data, axes_of(other, dims, "combine")
            )
        datas.append(data)
    return adapter.rewrap(checked[0], fn(*datas), dims)


def _same_dims(
    adapter: NamedArrayAdapter, xs: Sequence[Any], what: str
) -> tuple[str, ...]:
    dims = adapter.dims(xs[0])
    for i, x in enumerate(xs[1:], start=1):
        if adapter.dims(x) != dims:
            raise ValueError(
                f"{what}: input {i} has dims {adapter.dims(x)}, not {dims}."
            )
    return dims
