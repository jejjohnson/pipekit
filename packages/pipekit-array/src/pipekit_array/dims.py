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
- `reduce`, `argreduce` — over named dims; ``reduce(..., where=valid)``
  leaves masked entries out (a padded batch, nodata).
- `take_along` — gather along a dim with an index carrier
  (``argreduce`` then ``take_along`` is a max-NDVI composite).
- `map_over` — a function on each item along a dim.
- `stack`, `concat` — a new dim, or along an existing one.
- `stack_items` — items that may each carry their own geometry (patches,
  per-frame grids) along a new leading dim; `stack` where that is the
  same thing.
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
from typing import Any, ClassVar

from pipekit import NamedArrayAdapter, get_adapter
from pipekit.adapters import BackendError, backend_name

from pipekit_array._namespace import array_namespace
from pipekit_array.dimarray import DimArray


__all__ = [
    "BackendError",
    "argreduce",
    "combine",
    "concat",
    "isel",
    "map_over",
    "reduce",
    "sel",
    "stack",
    "stack_items",
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


class _DaskNamespace:
    """``dask.array`` under the Array API names the generic paths call.

    dask has ``transpose`` and ``concatenate`` where the Array API has
    ``permute_dims`` and ``concat``; everything else passes through.
    """

    _NAMES: ClassVar[dict[str, str]] = {
        "permute_dims": "transpose",
        "concat": "concatenate",
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


def require_one_backend(datas: Sequence[Any], what: str) -> None:
    """Raise `BackendError` when raw arrays sit on more than one backend.

    Inputs never move between backends silently: a NumPy array next to a
    JAX one would otherwise be converted depending on input order.
    """
    backends = {backend_name(d) for d in datas}
    if len(backends) > 1:
        raise BackendError(
            f"{what}: inputs on {sorted(backends)}; call to_jax() on the NumPy "
            "carriers first (inputs never move between backends silently)."
        )


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


def reduce(
    x: Any, fn: Reduction, dim: str | Sequence[str], *, where: Any | None = None
) -> Any:
    """Reduce over one or more named dims.

    Args:
        x: A named carrier.
        fn: An array-namespace name (``"mean"``, ``"max"``, ``"nanmean"``) or a
            function taking ``axis=`` (and ``where=``, when given).
        dim: A dim name, or several.
        where: Optional bool mask of the entries to keep, passed to ``fn``
            as ``where=`` (``"mean"`` then divides by the kept count). Either
            a named carrier, aligned to ``x`` by dim name (its dims must be
            among ``x``'s), or a plain array covering ``x``'s *leading* dims
            (``where.shape == shape[:where.ndim]``): a ``(sample,)`` mask on
            ``(sample, y, x)`` data, as `Padded.valid` is. Never
            right-broadcast. ``"max"`` / ``"min"`` need an ``initial`` for
            a mask: pass a function that supplies one.

    Raises:
        ValueError: A plain ``where`` does not match ``x``'s leading dims,
            or a named one has a dim ``x`` lacks.
        TypeError: ``where`` on data other than NumPy or JAX (the
            reductions of dask, PyTorch and CuPy take no ``where=``).

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.ones((12, 2, 4)), ("time", "band", "x"))
        >>> reduce(cube, "mean", "time").dims
        ('band', 'x')
        >>> batch = DimArray(np.array([[1.0], [3.0], [9.0]]), ("sample", "x"))
        >>> reduce(batch, "mean", "sample", where=np.array([True, True, False])).data
        array([2.])
    """
    adapter = adapter_of(x, "reduce")
    names = (dim,) if isinstance(dim, str) else tuple(dim)
    fast = _fast_path(adapter, "reduce")
    dims = adapter.dims(x)
    data = adapter.data(x)
    if where is not None:
        _require_where_backend(data)
    mask = None if where is None else _aligned_mask(where, dims, data, "reduce")
    if fast is not None:
        return fast(x, fn, names) if mask is None else fast(x, fn, names, where=mask)
    axes = _axes(dims, names, "reduce")
    func = _resolve(fn, namespace_of(data), "reduce")
    out = func(data, axis=axes) if mask is None else func(data, axis=axes, where=mask)
    kept = tuple(d for d in dims if d not in names)
    return adapter.rewrap(x, out, kept)


#: array modules whose reductions take NumPy's ``where=`` (jax: tracers too)
_WHERE_MODULES = ("numpy", "jax", "jaxlib")


def _require_where_backend(data: Any) -> None:
    """Raise unless ``data``'s reductions honour ``where=`` (NumPy, JAX).

    The reductions of dask, CuPy and PyTorch take no ``where=``
    (array-api-compat forwards it to ``torch.mean``, which rejects it;
    ``cupy.mean`` / ``cupy.sum`` have no such parameter), so a mask there would
    fail deep in the backend, or be ignored by a reduction that swallows
    keywords.
    """
    module = type(data).__module__.split(".")[0]
    if module in _WHERE_MODULES or backend_name(data) == "jax":
        return
    raise TypeError(
        f"reduce: where= needs NumPy or JAX data, got {type(data).__name__} "
        f"({module}), whose reductions take no where=. Compute or convert the "
        "carrier first, or mask with NaN and a nan-reduction ('nanmean')."
    )


def _aligned_mask(where: Any, dims: tuple[str, ...], data: Any, what: str) -> Any:
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
    # a JAX index next to NumPy data would be converted (or fail under jit)
    require_one_backend([data, idx], "take_along")
    # index → x's dim order with dim as a size-1 axis; missing dims broadcast
    check_index_labels(adapter, x, index_adapter, index, index_dims)
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
    out = stack(items, dim)
    # each output position is the input's: keep the mapped dim's labels
    labels = adapter.labels(x, dim)
    if labels is not None and isinstance(out, DimArray):
        return DimArray(out.data, out.dims, {**out.labels, dim: labels})
    return out


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
    require_one_backend(datas, "stack")
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
    fast = _fast_path(adapter, "stack_items")
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
    fast = _fast_path(adapter, "concat")
    if fast is not None:
        return fast(list(xs), dim)
    dims = _same_dims(adapter, xs, "concat")
    (axis,) = _axes(dims, (dim,), "concat")
    # the other dims must combine; dim itself may differ in size
    # the other dims must combine; an empty slice of dim checks them even
    # when an input has no entries along dim
    others = [adapter.isel(x, {dim: slice(0, 0)}) for x in xs]
    adapter.check_compatible(*others)
    datas = [adapter.data(x) for x in xs]
    require_one_backend(datas, "concat")
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
    require_one_backend([adapter.data(x) for x in checked], "combine")
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
