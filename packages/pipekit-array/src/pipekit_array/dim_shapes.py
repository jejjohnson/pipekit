"""Dim-level functions that change the dims: rename, add, drop, broadcast, merge, split.

Phase 1 of the dim-functions plan (``docs/design/api/dim_functions.md``,
ADRs A11 to A14 and A22). Like the functions of `pipekit_array.dims`, each
runs on any carrier with a registered `pipekit_ops.NamedArrayAdapter`,
works on positions only (labels follow the data, their values are never
read), and calls an adapter method of the same name, when there is one,
instead of the generic path.

- `rename` — new dim names; data, order and labels unchanged.
- `expand_dims`, `squeeze` — add or drop length-1 dims.
- `broadcast_like` — add another carrier's missing dims, in front.
- `merge_dims`, `split_dims` — ``y x`` into ``pixel`` and back; labels
  become tuples and are split back.

**Labels on new or renamed dims** (A22). ``rewrap`` keeps labels by dim
name, so it cannot label a dim whose name or labels are new. `rename`,
`broadcast_like`, `merge_dims` and `split_dims` compute those labels and
set them with the adapter's optional ``relabel``
(`pipekit_ops.LabelledAdapter`). An adapter without it runs them when no
labels would be set, and raises `TypeError` when some would: labels are
never dropped silently.

The `Operator` forms live in `pipekit_array.dim_operators`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from pipekit_ops import NamedArrayAdapter

from pipekit_array.dims import _axes, _fast_path, adapter_of, namespace_of


if TYPE_CHECKING:
    # pipekit-ops>=0.2.0, which pipekit-array requires, may lack it: never
    # import it at runtime
    from pipekit_ops.adapters import LabelledAdapter


__all__ = [
    "broadcast_like",
    "expand_dims",
    "merge_dims",
    "rename",
    "split_dims",
    "squeeze",
]

#: one dim's labels, or None for a dim without them
Labels = tuple[Any, ...] | None


def _labelled(adapter: NamedArrayAdapter) -> LabelledAdapter | None:
    """``adapter`` if it has the optional ``relabel``, detected by name."""
    if _fast_path(adapter, "relabel") is None:
        return None
    return cast("LabelledAdapter", adapter)


def set_labels(
    adapter: NamedArrayAdapter, out: Any, labels: Mapping[str, Labels], what: str
) -> Any:
    """``out`` with the labels of the named dims set through ``relabel``.

    Called after ``rewrap`` with every new or renamed dim, None included:
    ``rewrap`` may have matched a stale label to a reused name.

    Raises:
        TypeError: The adapter has no ``relabel`` and a dim's labels must
            be set or removed.
    """
    if not labels:
        return out
    labelled = _labelled(adapter)
    if labelled is not None:
        return labelled.relabel(out, dict(labels))
    for dim, values in labels.items():
        if values is not None or adapter.labels(out, dim) is not None:
            raise TypeError(
                f"{what}: {type(adapter).__name__} has no relabel, so the labels "
                f"of dim {dim!r} cannot be set; give the adapter relabel "
                f"(pipekit_ops.LabelledAdapter) or a {what} fast path."
            )
    return out


def merged_labels(sources: Sequence[Labels], sizes: Sequence[int]) -> Labels:
    """The labels of a merged dim: tuples of the source labels, in C order.

    An unlabelled source contributes its positions; when no source has
    labels, neither has the merged dim.

    Example:
        >>> merged_labels([("B1", "B2"), None], [2, 3])[:4]
        (('B1', 0), ('B1', 1), ('B1', 2), ('B2', 0))
        >>> merged_labels([None, None], [2, 3]) is None
        True
    """
    if all(values is None for values in sources):
        return None
    parts = [
        tuple(range(size)) if values is None else tuple(values)
        for values, size in zip(sources, sizes, strict=True)
    ]
    return tuple(itertools.product(*parts))


def split_labels(labels: Labels, sizes: Sequence[int]) -> tuple[Labels, ...]:
    """Each part's labels, recovered from a merged dim's tuple labels.

    The inverse of `merged_labels`. A part whose labels are exactly its
    positions ``0 … n-1`` comes back unlabelled, as it was merged. Labels
    that are not tuples of one entry per part, or not a C-order grid,
    give no labels.

    Example:
        >>> split_labels((("B1", 0), ("B1", 1), ("B2", 0), ("B2", 1)), [2, 2])
        (('B1', 'B2'), None)
    """
    unknown: tuple[Labels, ...] = (None,) * len(sizes)
    if labels is None:
        return unknown
    if not all(isinstance(v, tuple) and len(v) == len(sizes) for v in labels):
        return unknown
    parts = []
    for k, size in enumerate(sizes):
        # entry i of part k sits at flat position i·∏_{j>k} n_j
        stride = math.prod(sizes[k + 1 :])
        parts.append(tuple(labels[i * stride][k] for i in range(size)))
    if tuple(itertools.product(*parts)) != tuple(labels):
        return unknown
    return tuple(None if _are_positions(part) else part for part in parts)


def _are_positions(values: tuple[Any, ...]) -> bool:
    return all(type(v) is int and v == i for i, v in enumerate(values))


def rename(x: Any, /, **names: str) -> Any:
    """New dim names; data, order and labels unchanged.

    y = x, with each dim ``d`` called ``names[d]``.

    ``x`` is positional-only, so a dim may be called ``x``.

    Args:
        x: A named carrier.
        **names: ``old=new`` for each dim to rename.

    Raises:
        KeyError: An old name is not a dim of ``x``.
        ValueError: The new names repeat a dim.
        TypeError: The adapter has neither a ``rename`` fast path nor
            ``relabel``, and a renamed dim has labels.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> scene = DimArray(np.zeros((2, 4)), ("band", "x"), {"band": ("B4", "B8")})
        >>> channels = rename(scene, band="channel")
        >>> channels.dims, channels.labels["channel"]
        (('channel', 'x'), ('B4', 'B8'))
        >>> rename(channels, channel="band").labels == scene.labels
        True
    """
    adapter = adapter_of(x, "rename")
    dims = adapter.dims(x)
    _axes(dims, list(names), "rename")
    new_dims = tuple(names.get(d, d) for d in dims)
    if len(set(new_dims)) != len(new_dims):
        raise ValueError(f"rename: the new dims {new_dims} repeat a name.")
    fast = _fast_path(adapter, "rename")
    if fast is not None:
        return fast(x, dict(names))
    out = adapter.rewrap(x, adapter.data(x), new_dims)
    labels = {new: adapter.labels(x, old) for old, new in names.items()}
    return set_labels(adapter, out, labels, "rename")


def expand_dims(x: Any, dim: str | Sequence[str], /, *, axis: int = 0) -> Any:
    """Insert length-1 dims, consecutively, at position ``axis`` of the result.

    y_{…, 0, …} = x_{…}: the new dims have size 1 and no labels.

    A negative ``axis`` counts from the end, so ``axis=-1`` appends.

    Args:
        x: A named carrier.
        dim: A new dim name, or several.
        axis: Where the first new dim goes in the result.

    Raises:
        ValueError: A new name is already a dim, a name repeats, or
            ``axis`` is out of range.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> scene = DimArray(np.zeros((2, 4)), ("band", "x"))
        >>> expand_dims(scene, "member").dims
        ('member', 'band', 'x')
        >>> expand_dims(scene, ("member", "run"), axis=-1).shape
        (2, 4, 1, 1)
    """
    adapter = adapter_of(x, "expand_dims")
    names = (dim,) if isinstance(dim, str) else tuple(dim)
    dims = adapter.dims(x)
    taken = [n for n in names if n in dims]
    if taken:
        raise ValueError(f"expand_dims: {taken} are already dims of {dims}.")
    if len(set(names)) != len(names):
        raise ValueError(f"expand_dims: the new dims {names} repeat a name.")
    # a negative axis counts from the end of the result: -1 appends
    start = axis if axis >= 0 else axis + len(dims) + 1
    if not 0 <= start <= len(dims):
        raise ValueError(f"expand_dims: axis {axis} is out of range for dims {dims}.")
    fast = _fast_path(adapter, "expand_dims")
    if fast is not None:
        return fast(x, names, axis=start)
    data = adapter.data(x)
    xp = namespace_of(data)
    expanded = data
    for offset in range(len(names)):
        expanded = xp.expand_dims(expanded, axis=start + offset)
    new_dims = (*dims[:start], *names, *dims[start:])
    return adapter.rewrap(x, expanded, new_dims)


def squeeze(x: Any, dim: str | Sequence[str], /) -> Any:
    """Drop named length-1 dims.

    y_{…} = x_{…, 0, …}.

    ``dim`` is required: squeezing every length-1 dim would also drop a
    batch of one (A12).

    Args:
        x: A named carrier.
        dim: A dim name, or several; each must have size 1.

    Raises:
        KeyError: A name is not a dim of ``x``.
        ValueError: A named dim's size is not 1.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> squeeze(DimArray(np.zeros((1, 2, 4)), ("time", "band", "x")), "time").dims
        ('band', 'x')
    """
    adapter = adapter_of(x, "squeeze")
    names = (dim,) if isinstance(dim, str) else tuple(dim)
    dims = adapter.dims(x)
    axes = _axes(dims, names, "squeeze")
    shape = adapter.data(x).shape
    wrong = [
        f"{n!r} has size {shape[a]}"
        for n, a in zip(names, axes, strict=True)
        if shape[a] != 1
    ]
    if wrong:
        raise ValueError(f"squeeze: only length-1 dims drop; {', '.join(wrong)}.")
    fast = _fast_path(adapter, "squeeze")
    if fast is not None:
        return fast(x, names)
    return adapter.isel(x, dict.fromkeys(names, 0))


def broadcast_like(x: Any, other: Any, /, *, exclude: Sequence[str] = ()) -> Any:
    """Add ``other``'s missing dims to ``x``, in front, in ``other``'s order.

    y_{a…, d…} = x_{d…}, for every index a of the added dims.

    Shared dims must agree exactly in size and labels (xarray outer-joins
    instead). ``x``'s own dims keep their order, so the result's order is
    xarray's only when the shared dims are in ``other``'s order. The added
    dims carry ``other``'s labels.

    Args:
        x: A named carrier.
        other: A carrier of the same family; only its dims, sizes and
            labels are read.
        exclude: Dims of ``other`` neither added nor checked.

    Raises:
        TypeError: The adapter has no ``relabel`` and an added dim has
            labels.
        ValueError: A shared dim differs in size or labels.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.zeros((12, 2, 4)), ("time", "band", "x"))
        >>> median = DimArray(np.ones((2, 4)), ("band", "x"))
        >>> broadcast_like(median, cube).shape
        (12, 2, 4)
    """
    adapter = adapter_of(x, "broadcast_like")
    other_adapter = adapter_of(other, "broadcast_like")
    excluded = tuple(exclude)
    fast = _fast_path(adapter, "broadcast_like")
    if fast is not None:
        return fast(x, other, exclude=excluded)
    dims = adapter.dims(x)
    other_dims = other_adapter.dims(other)
    shared = [d for d in dims if d in other_dims and d not in excluded]
    added = tuple(d for d in other_dims if d not in dims and d not in excluded)
    # only the shared dims must combine: an empty slice of every other dim
    # checks them without touching the data
    x_part = adapter.isel(x, {d: slice(0, 0) for d in dims if d not in shared})
    other_part = other_adapter.isel(
        other, {d: slice(0, 0) for d in other_dims if d not in shared}
    )
    adapter.check_compatible(x_part, other_part)
    data = adapter.data(x)
    xp = namespace_of(data)
    sizes = dict(zip(other_dims, other_adapter.data(other).shape, strict=True))
    expanded = data
    for _ in added:
        expanded = xp.expand_dims(expanded, axis=0)
    target = (*(sizes[d] for d in added), *data.shape)
    broadcast = xp.broadcast_to(expanded, target)
    out = adapter.rewrap(x, broadcast, (*added, *dims))
    labels = {d: other_adapter.labels(other, d) for d in added}
    return set_labels(adapter, out, labels, "broadcast_like")


def merge_dims(x: Any, /, **dims: Sequence[str]) -> Any:
    """Merge the listed dims into one, placed last, in C order (xarray's ``stack``).

    y_{…, p} = x_{…, i, j}, with p = i·X + j for ``pixel=("y", "x")``.

    The merged dim's labels are tuples of the source labels, with
    positions for an unlabelled source; it has none when no source has
    labels. `split_dims` undoes it.

    Args:
        x: A named carrier.
        **dims: ``new=(source, …)`` for each merged dim; several merges go
            last in the order given.

    Raises:
        KeyError: A source is not a dim of ``x``.
        ValueError: A merge lists no source, a source is merged twice, or
            the result repeats a dim name.
        TypeError: A merge's sources are one string, or the adapter has no
            ``relabel`` and the merged dim has labels.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> bands = {"band": ("B4", "B8")}
        >>> cube = DimArray(np.zeros((2, 3, 4)), ("band", "y", "x"), bands)
        >>> pixels = merge_dims(cube, pixel=("y", "x"))
        >>> pixels.dims, pixels.shape
        (('band', 'pixel'), (2, 12))
        >>> merge_dims(cube, pb=("band", "y")).labels["pb"][:2]
        (('B4', 0), ('B4', 1))
    """
    adapter = adapter_of(x, "merge_dims")
    current = adapter.dims(x)
    for name, group in dims.items():
        if isinstance(group, str):
            raise TypeError(
                f"merge_dims: {name}= takes a sequence of dim names, got {group!r}."
            )
        if len(group) == 0:
            raise ValueError(f"merge_dims: {name}= lists no dim to merge.")
    groups = {name: tuple(group) for name, group in dims.items()}
    sources = [s for group in groups.values() for s in group]
    _axes(current, sources, "merge_dims")
    if len(set(sources)) != len(sources):
        raise ValueError(f"merge_dims: a dim is merged twice in {groups}.")
    kept = tuple(d for d in current if d not in sources)
    new_dims = (*kept, *groups)
    if len(set(new_dims)) != len(new_dims):
        raise ValueError(f"merge_dims: the new dims {new_dims} repeat a name.")
    fast = _fast_path(adapter, "merge_dims")
    if fast is not None:
        return fast(x, groups)
    data = adapter.data(x)
    xp = namespace_of(data)
    sizes = dict(zip(current, data.shape, strict=True))
    moved = xp.permute_dims(data, _axes(current, (*kept, *sources), "merge_dims"))
    # each merged dim has the product of its sources' sizes
    merged_sizes = [math.prod(sizes[s] for s in group) for group in groups.values()]
    merged = xp.reshape(moved, (*(sizes[d] for d in kept), *merged_sizes))
    out = adapter.rewrap(x, merged, new_dims)
    labels = {
        name: merged_labels(
            [adapter.labels(x, s) for s in group], [sizes[s] for s in group]
        )
        for name, group in groups.items()
    }
    return set_labels(adapter, out, labels, "merge_dims")


def split_dims(x: Any, /, **dims: Mapping[str, int]) -> Any:
    """Split one dim into several, in place, in C order (xarray's ``unstack``).

    y_{…, i, j} = x_{…, p}, with p = i·X + j for ``pixel={"y": Y, "x": X}``.

    The sizes are given, not read from the labels, so they are static
    under ``jit`` (A14). Tuple labels from `merge_dims` are split back.

    Args:
        x: A named carrier.
        **dims: ``dim={part: size, …}`` for each dim to split; the sizes
            must multiply to the dim's length. A part may reuse the
            dim's name (``time={"time": 4, "window": 3}``).

    Raises:
        KeyError: A dim to split is not a dim of ``x``.
        ValueError: The sizes do not multiply to the dim's length, a split
            lists no part, or the result repeats a dim name.
        TypeError: The adapter has no ``relabel`` and a part has labels.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> pixels = DimArray(np.zeros((2, 12)), ("band", "pixel"))
        >>> split_dims(pixels, pixel={"y": 3, "x": 4}).dims
        ('band', 'y', 'x')
        >>> cube = DimArray(np.zeros((3, 4)), ("y", "x"), {"x": tuple("abcd")})
        >>> back = split_dims(merge_dims(cube, p=("y", "x")), p={"y": 3, "x": 4})
        >>> back.labels
        {'x': ('a', 'b', 'c', 'd')}
    """
    adapter = adapter_of(x, "split_dims")
    current = adapter.dims(x)
    _axes(current, list(dims), "split_dims")
    parts = {name: dict(part) for name, part in dims.items()}
    data = adapter.data(x)
    sizes = dict(zip(current, data.shape, strict=True))
    for name, part in parts.items():
        if not part:
            raise ValueError(f"split_dims: {name}= lists no part.")
        product = math.prod(part.values())
        if product != sizes[name]:
            raise ValueError(
                f"split_dims: the sizes {part} multiply to {product}, but "
                f"{name!r} has size {sizes[name]}."
            )
    new_dims = tuple(p for d in current for p in parts.get(d, {d: 0}))
    if len(set(new_dims)) != len(new_dims):
        raise ValueError(f"split_dims: the new dims {new_dims} repeat a name.")
    fast = _fast_path(adapter, "split_dims")
    if fast is not None:
        return fast(x, parts)
    xp = namespace_of(data)
    shape = tuple(
        n for d in current for n in (parts[d].values() if d in parts else (sizes[d],))
    )
    out = adapter.rewrap(x, xp.reshape(data, shape), new_dims)
    labels: dict[str, Labels] = {}
    for name, part in parts.items():
        recovered = split_labels(adapter.labels(x, name), tuple(part.values()))
        labels.update(zip(part, recovered, strict=True))
    return set_labels(adapter, out, labels, "split_dims")
