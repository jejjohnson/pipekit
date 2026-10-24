"""Dim-level functions: selection, reduction, stacking and mapping by dim name.

Each is written once, over the carrier's registered
`pipekit.NamedArrayAdapter`, and behaves the same on every registered
carrier: a `DimArray`, an ``xr.DataArray`` (``[xarray]``), a geoarray
carrier. The rule for joining this list (GA-Q3): a function never reads
coordinate values or geometry. ``resample``, ``groupby`` and ``rolling``
stay xarray's; reprojection and sampling at points stay in geoarray.

- `isel`, `sel` — by position, by exact label; an integer or a single
  label drops its dim.
- `transpose` — reorder dims by name.
- `reduce`, `argreduce` — over named dims.
- `take_along` — gather along a dim with an index carrier
  (``argreduce`` then ``take_along`` is a max-NDVI composite).
- `map_over` — a function on each item along a dim.
- `stack`, `concat` — a new dim, or along an existing one.
- `combine` — an element-wise function of carriers with the same dims.

**Fast paths.** An adapter may provide a method of the same name (the
xarray adapter's ``reduce`` stays lazy on dask; geoarray's ``map_over``
is one ``jax.vmap`` over data and transform rows). It is called with the
same arguments instead of the generic path; the test kit
(``pipekit_array.testing``) proves the two agree.

The `Operator` forms live in `pipekit_array.dim_operators`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pipekit import NamedArrayAdapter, get_adapter

from pipekit_array._namespace import array_namespace


__all__ = [
    "argreduce",
    "combine",
    "concat",
    "isel",
    "map_over",
    "reduce",
    "sel",
    "stack",
    "take_along",
    "transpose",
]

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


def namespace_of(data: Any) -> Any:
    """The array namespace of raw data, dask included (which has none of its own)."""
    if type(data).__module__.startswith("dask."):
        import dask.array as da

        return da
    return array_namespace(data)


def _fast_path(adapter: NamedArrayAdapter, name: str) -> Callable[..., Any] | None:
    return getattr(adapter, name, None)


def _axes(dims: tuple[str, ...], names: Sequence[str], what: str) -> tuple[int, ...]:
    missing = [n for n in names if n not in dims]
    if missing:
        raise KeyError(f"{what}: unknown dim(s) {missing}; dims are {dims}.")
    return tuple(dims.index(n) for n in names)


def _resolve(fn: Reduction, xp: Any, what: str) -> Callable[..., Any]:
    if callable(fn):
        return fn
    resolved = getattr(xp, fn, None)
    if resolved is None:
        raise ValueError(f"{what}: the array namespace has no {fn!r}.")
    return resolved


def isel(x: Any, /, **indexers: Any) -> Any:
    """Select by position on named dims; an integer drops its dim.

    ``x`` is positional-only, so a dim may be called ``x``.

    Args:
        x: A named carrier.
        **indexers: ``dim=`` an int, a slice or an integer index array.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.zeros((12, 2, 4)), ("time", "band", "x"))
        >>> isel(cube, time=0).dims
        ('band', 'x')
    """
    adapter = adapter_of(x, "isel")
    _axes(adapter.dims(x), list(indexers), "isel")
    return adapter.isel(x, dict(indexers))


def sel(x: Any, /, **labels: Any) -> Any:
    """Select by exact label on named dims: no ranges, no nearest.

    ``x`` is positional-only, so a dim may be called ``x``.

    A single label drops its dim; a list of labels keeps it.

    Raises:
        KeyError: A dim has no labels, or a label is not among them.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> scene = DimArray(np.zeros((2, 4)), ("band", "x"), {"band": ("B4", "B8")})
        >>> sel(scene, band="B8").dims
        ('x',)
    """
    adapter = adapter_of(x, "sel")
    _axes(adapter.dims(x), list(labels), "sel")
    indexers: dict[str, Any] = {}
    for dim, wanted in labels.items():
        values = adapter.labels(x, dim)
        if values is None:
            raise KeyError(f"sel: dim {dim!r} has no labels; use isel.")
        position = {value: i for i, value in enumerate(values)}
        many = _many_labels(wanted, position)
        if many and not isinstance(wanted, (list, tuple)):
            wanted = list(wanted.tolist())
        for label in wanted if many else (wanted,):
            if label not in position:
                raise KeyError(f"sel: {label!r} is not a label of {dim!r}: {values}.")
        if many:
            indexers[dim] = [position[label] for label in wanted]
        else:
            indexers[dim] = position[wanted]
    return adapter.isel(x, indexers)


def _many_labels(wanted: Any, position: dict[Any, int]) -> bool:
    """Whether ``wanted`` lists several labels (a list, a tuple or a 1-D array).

    A tuple that is itself a label (tuple-valued labels) is one label.
    """
    if isinstance(wanted, list):
        return True
    if isinstance(wanted, tuple):
        try:
            return wanted not in position
        except TypeError:
            return True
    return getattr(wanted, "ndim", 0) == 1


def transpose(x: Any, order: Sequence[str]) -> Any:
    """Reorder dims by name; ``order`` names every dim once."""
    adapter = adapter_of(x, "transpose")
    fast = _fast_path(adapter, "transpose")
    if fast is not None:
        return fast(x, tuple(order))
    dims = adapter.dims(x)
    if sorted(order) != sorted(dims):
        raise ValueError(
            f"transpose: order {tuple(order)} is not a permutation of {dims}."
        )
    data = adapter.data(x)
    xp = namespace_of(data)
    moved = xp.permute_dims(data, _axes(dims, order, "transpose"))
    return adapter.rewrap(x, moved, tuple(order))


def reduce(x: Any, fn: Reduction, dim: str | Sequence[str]) -> Any:
    """Reduce over one or more named dims.

    Args:
        x: A named carrier.
        fn: An array-namespace name (``"mean"``, ``"max"``, ``"nanmean"``) or a
            function taking ``axis=``.
        dim: A dim name, or several.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.ones((12, 2, 4)), ("time", "band", "x"))
        >>> reduce(cube, "mean", "time").dims
        ('band', 'x')
    """
    adapter = adapter_of(x, "reduce")
    names = (dim,) if isinstance(dim, str) else tuple(dim)
    fast = _fast_path(adapter, "reduce")
    if fast is not None:
        return fast(x, fn, names)
    dims = adapter.dims(x)
    axes = _axes(dims, names, "reduce")
    data = adapter.data(x)
    func = _resolve(fn, namespace_of(data), "reduce")
    out = func(data, axis=axes)
    kept = tuple(d for d in dims if d not in names)
    return adapter.rewrap(x, out, kept)


def argreduce(x: Any, fn: Reduction, dim: str) -> Any:
    """The index of a reduction along one dim (``"argmax"``, ``"argmin"``).

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> ndvi = DimArray(np.random.default_rng(0).random((12, 4)), ("time", "x"))
        >>> argreduce(ndvi, "argmax", "time").dims
        ('x',)
    """
    adapter = adapter_of(x, "argreduce")
    fast = _fast_path(adapter, "argreduce")
    if fast is not None:
        return fast(x, fn, dim)
    dims = adapter.dims(x)
    (axis,) = _axes(dims, (dim,), "argreduce")
    data = adapter.data(x)
    func = _resolve(fn, namespace_of(data), "argreduce")
    out = func(data, axis=axis)
    kept = tuple(d for d in dims if d != dim)
    return adapter.rewrap(x, out, kept)


def take_along(x: Any, index: Any, dim: str) -> Any:
    """Gather one entry of ``dim`` per position, with an index carrier.

    ``index`` has ``x``'s dims without ``dim`` (an `argreduce` result, in any
    dim order); the result has those dims too.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.arange(24.0).reshape(3, 2, 4), ("time", "band", "x"))
        >>> best = DimArray(np.array([0, 2, 1, 0]), ("x",))
        >>> take_along(cube, best, "time").dims
        ('band', 'x')
    """
    adapter = adapter_of(x, "take_along")
    fast = _fast_path(adapter, "take_along")
    if fast is not None:
        return fast(x, index, dim)
    index_adapter = adapter_of(index, "take_along")
    dims = adapter.dims(x)
    (axis,) = _axes(dims, (dim,), "take_along")
    rest = tuple(d for d in dims if d != dim)
    index_dims = index_adapter.dims(index)
    if not set(index_dims) <= set(rest):
        raise ValueError(f"take_along: index dims {index_dims} are not among {rest}.")
    data = adapter.data(x)
    xp = namespace_of(data)
    idx = index_adapter.data(index)
    # index → x's dim order with dim as a size-1 axis; missing dims broadcast
    sizes = dict(zip(dims, data.shape, strict=True))
    for d, n in zip(index_dims, idx.shape, strict=True):
        if n != sizes[d]:
            raise ValueError(
                f"take_along: index dim {d!r} has size {n}, but {d!r} has size "
                f"{sizes[d]}; the index is never broadcast."
            )
    present = [d for d in rest if d in index_dims]
    idx = xp.permute_dims(idx, tuple(index_dims.index(d) for d in present))
    full = tuple(1 if d == dim or d not in index_dims else None for d in dims)
    for position, size in enumerate(full):
        if size == 1:
            idx = xp.expand_dims(idx, axis=position)
    shape = tuple(1 if d == dim else n for d, n in zip(dims, data.shape, strict=True))
    idx = xp.broadcast_to(idx, shape)
    gathered = xp.take_along_axis(data, idx, axis=axis)
    out = xp.squeeze(gathered, axis=axis)
    return adapter.rewrap(x, out, rest)


def map_over(x: Any, fn: Callable[[Any], Any], dim: str) -> Any:
    """``fn`` on each item along ``dim``; the results stacked along ``dim``, first.

    Each item is ``isel(x, dim=i)``; ``fn`` returns a named carrier, and every
    result must have the same dims.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.ones((3, 2, 4)), ("time", "band", "x"))
        >>> map_over(cube, lambda item: reduce(item, "sum", "band"), "time").dims
        ('time', 'x')
    """
    adapter = adapter_of(x, "map_over")
    fast = _fast_path(adapter, "map_over")
    if fast is not None:
        return fast(x, fn, dim)
    dims = adapter.dims(x)
    (axis,) = _axes(dims, (dim,), "map_over")
    size = adapter.data(x).shape[axis]
    items = [fn(adapter.isel(x, {dim: i})) for i in range(size)]
    return stack(items, dim)


def stack(xs: Sequence[Any], dim: str) -> Any:
    """A new leading dim ``dim`` from carriers with the same dims; never aligns."""
    if len(xs) == 0:
        raise ValueError("stack needs at least one carrier.")
    adapter = adapter_of(xs[0], "stack")
    fast = _fast_path(adapter, "stack")
    if fast is not None:
        return fast(list(xs), dim)
    checked = adapter.check_compatible(*xs)
    dims = _same_dims(adapter, checked, "stack")
    if dim in dims:
        raise ValueError(f"stack: {dim!r} is already a dim of {dims}; use concat.")
    datas = [adapter.data(x) for x in checked]
    xp = namespace_of(datas[0])
    return adapter.rewrap(checked[0], xp.stack(datas), (dim, *dims))


def concat(xs: Sequence[Any], dim: str) -> Any:
    """Join carriers along an existing dim; the other dims must agree.

    The generic path drops ``dim``'s labels (the joined dim has a new size);
    an adapter's ``concat`` fast path may join them.
    """
    if len(xs) == 0:
        raise ValueError("concat needs at least one carrier.")
    adapter = adapter_of(xs[0], "concat")
    fast = _fast_path(adapter, "concat")
    if fast is not None:
        return fast(list(xs), dim)
    dims = _same_dims(adapter, xs, "concat")
    (axis,) = _axes(dims, (dim,), "concat")
    # the other dims must combine; dim itself may differ in size
    others = [adapter.isel(x, {dim: 0}) for x in xs]
    adapter.check_compatible(*others)
    datas = [adapter.data(x) for x in xs]
    xp = namespace_of(datas[0])
    return adapter.rewrap(xs[0], xp.concat(datas, axis=axis), dims)


def combine(fn: Callable[..., Any], *xs: Any) -> Any:
    """``fn`` of the data of carriers with the same dims; never aligns.

    Inputs are put in the first one's dim order by name before ``fn`` sees
    them, and the result has those dims.

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
    fast = _fast_path(adapter, "combine")
    if fast is not None:
        return fast(fn, *xs)
    checked = adapter.check_compatible(*xs)
    dims = adapter.dims(checked[0])
    datas = []
    for x in checked:
        other = adapter.dims(x)
        if sorted(other) != sorted(dims):
            raise ValueError(f"combine: dims {other} differ from {dims}.")
        data = adapter.data(x)
        if other != dims:
            data = namespace_of(data).permute_dims(data, _axes(other, dims, "combine"))
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
