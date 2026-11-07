"""Dim-level functions that cut a dim into windows: sliding and non-overlapping.

Phase 3 of the dim-functions plan (``docs/design/api/dim_functions.md``,
ADR A15). Like the functions of `pipekit_array.dims`, each runs on any
carrier with a registered `pipekit_ops.NamedArrayAdapter`, works on
positions only (labels follow the data, their values are never read),
and calls an adapter method of the same name, when there is one, instead
of the generic path.

- `window` — sliding windows along one dim, as a new last dim; full
  windows only, each labelled by its last entry.
- `coarsen` — non-overlapping blocks reduced by a function: `split_dims`
  then `reduce`; the coarsened dim loses its labels.

The `Operator` forms live in `pipekit_array.dim_operators`; the
operators that apply another operator along a dim (`ApplyAlong`) or in
chunks of a dim (`pipekit_array.batching.BatchAlong`) live next to
`MapOver` and `batch`.
"""

from __future__ import annotations

from typing import Any

from pipekit_array import dims as _dims
from pipekit_array.dim_shapes import set_labels, split_dims
from pipekit_array.dims import Reduction, _axes, _fast_path, adapter_of, namespace_of


__all__ = ["BOUNDARIES", "coarsen", "window"]

#: what `coarsen` does when a block size does not divide its dim
BOUNDARIES: tuple[str, ...] = ("exact", "trim")


def window(
    x: Any, /, *, window_dim: str = "window", stride: int = 1, **sizes: int
) -> Any:
    """Sliding windows along one dim; the window becomes a new last dim.

    w_{…, n, …, r} = x_{…, n·s + r, …},  0 ≤ r < S,  0 ≤ n < ⌊(N - S)/s⌋ + 1.

    Full windows only: there is no NaN fill, so float32 stays float32 and
    integers work (A15). Window ``n`` carries the label of its last entry,
    position n·s + S - 1, as xarray's
    ``rolling(...).construct(window_dim).isel(dim=slice(S - 1, None, s))``
    does. For xarray's length-N result, `pad` the dim by S - 1 first.

    Algorithm:
        1. Count the full windows, n = ⌊(N - S)/s⌋ + 1.
        2. Take the S strided slices of the data, x_{r::s}, each n long.
        3. Stack them on a new last axis and rewrap on the slice of end
           positions, so the windowed dim keeps the labels of those.

    The generic path copies the data S times; the xarray adapter's fast
    path is a view.

    Args:
        x: A named carrier.
        window_dim: The new last dim, indexing the entries r of a window.
        stride: The step s ≥ 1 between window starts.
        **sizes: ``dim=S`` for the one dim to window, with 1 ≤ S ≤ N.

    Raises:
        KeyError: The dim is not a dim of ``x``.
        ValueError: ``sizes`` names no dim or several, S or s is not
            positive, S exceeds N, or ``window_dim`` is already a dim.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> ndvi = DimArray(np.arange(12.0), ("time",), {"time": tuple(range(12))})
        >>> windows = window(ndvi, time=3, stride=2)
        >>> windows.dims, windows.shape
        (('time', 'window'), (5, 3))
        >>> windows.labels["time"]
        (2, 4, 6, 8, 10)
        >>> window(DimArray(np.arange(4), ("time",)), time=2).data.tolist()
        [[0, 1], [1, 2], [2, 3]]
    """
    adapter = adapter_of(x, "window")
    if len(sizes) != 1:
        raise ValueError(f"window: give one dim=size, got {sizes}.")
    ((dim, size),) = sizes.items()
    dims = adapter.dims(x)
    (axis,) = _axes(dims, (dim,), "window")
    if window_dim in dims:
        raise ValueError(f"window: {window_dim!r} is already a dim of {dims}.")
    if size < 1 or stride < 1:
        raise ValueError(
            f"window: size and stride must be positive, got {dim}={size}, "
            f"stride={stride}."
        )
    data = adapter.data(x)
    length = data.shape[axis]
    if size > length:
        raise ValueError(
            f"window: a window of {size} does not fit {dim!r} of size {length}."
        )
    fast = _fast_path(adapter, "window")
    if fast is not None:
        return fast(x, dim, size, window_dim=window_dim, stride=stride)
    xp = namespace_of(data)
    # n = ⌊(N - S)/s⌋ + 1
    count = (length - size) // stride + 1
    # the last start is (n - 1)·s
    last_start = (count - 1) * stride
    leading = (slice(None),) * axis
    # x_{r::s}, n entries each, for 0 ≤ r < S
    strided = [
        data[(*leading, slice(r, r + last_start + 1, stride))] for r in range(size)
    ]
    # w_{…, n, …, r} = x_{…, n·s + r, …}
    stacked = xp.stack(strided, axis=-1)
    # window n carries the label of position n·s + S - 1
    ends = adapter.isel(x, {dim: slice(size - 1, size + last_start, stride)})
    out = adapter.rewrap(ends, stacked, (*dims, window_dim))
    return set_labels(adapter, out, {window_dim: None}, "window")


def coarsen(
    x: Any, fn: Reduction, /, *, boundary: str = "exact", **windows: int
) -> Any:
    """Reduce non-overlapping blocks of k entries along each named dim.

    z_{…, i', …} = fn_{0 ≤ r < k} x_{…, k·i' + r, …},  i' < ⌊N/k⌋.

    A composition: `split_dims` each dim into ``(dim, block)`` then
    `reduce` over the blocks, so ``coarsen(x, "mean", time=3)`` is
    ``reduce(split_dims(x, time={"time": 4, "window": 3}), "mean",
    "window")``. A coarsened dim loses its labels: xarray's ``coord_func``
    reads coordinate values, which this layer never does (A15).

    Args:
        x: A named carrier.
        fn: An array-namespace name (``"mean"``, ``"nanmean"``, ``"max"``)
            or a function taking ``axis=``.
        boundary: ``"exact"`` requires k to divide N; ``"trim"`` drops
            the last N mod k entries.
        **windows: ``dim=k`` for each dim to coarsen, with k ≥ 1.

    Raises:
        KeyError: A dim is not a dim of ``x``.
        ValueError: No dim is given, k is not positive, ``boundary`` is
            not one of `BOUNDARIES`, or under ``"exact"`` k does not
            divide N.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> ndvi = DimArray(np.arange(24.0).reshape(2, 12), ("y", "x"))
        >>> coarse = coarsen(ndvi, "mean", x=4)
        >>> coarse.dims, coarse.data.tolist()
        (('y', 'x'), [[1.5, 5.5, 9.5], [13.5, 17.5, 21.5]])
        >>> coarsen(ndvi, "max", boundary="trim", x=5, y=2).data.tolist()
        [[16.0, 21.0]]
        >>> coarsen(ndvi, "mean", x=5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: coarsen: 5 does not divide 'x' of size 12; ...
    """
    adapter = adapter_of(x, "coarsen")
    if not windows:
        raise ValueError("coarsen: give at least one dim=k.")
    if boundary not in BOUNDARIES:
        raise ValueError(
            f"coarsen: boundary must be one of {BOUNDARIES}, got {boundary!r}."
        )
    dims = adapter.dims(x)
    axes = _axes(dims, list(windows), "coarsen")
    data = adapter.data(x)
    lengths = {dim: data.shape[axis] for dim, axis in zip(windows, axes, strict=True)}
    for dim, block in windows.items():
        if block < 1:
            raise ValueError(f"coarsen: {dim}={block} must be positive.")
        if boundary == "exact" and lengths[dim] % block != 0:
            raise ValueError(
                f"coarsen: {block} does not divide {dim!r} of size {lengths[dim]}; "
                "pass boundary='trim' or pad first."
            )
    fast = _fast_path(adapter, "coarsen")
    if fast is not None:
        return fast(x, fn, dict(windows), boundary=boundary)
    # trim: keep the first k·⌊N/k⌋ entries
    tails = {
        dim: slice(0, lengths[dim] - lengths[dim] % block)
        for dim, block in windows.items()
        if lengths[dim] % block != 0
    }
    trimmed = adapter.isel(x, tails) if tails else x
    blocks: dict[str, str] = {}
    for dim in windows:
        blocks[dim] = _block_dim(dim, (*dims, *blocks.values()))
    # x_{…, i', r, …} = x_{…, k·i' + r, …}
    parts = {
        dim: {dim: lengths[dim] // block, blocks[dim]: block}
        for dim, block in windows.items()
    }
    split = split_dims(trimmed, **parts)
    # z_{…, i', …} = fn over r of x_{…, i', r, …}
    return _dims.reduce(split, fn, tuple(blocks.values()))


def _block_dim(dim: str, dims: tuple[str, ...]) -> str:
    """A name for ``dim``'s block entries that none of ``dims`` uses."""
    name = f"{dim}_block"
    while name in dims:
        name = f"_{name}"
    return name
