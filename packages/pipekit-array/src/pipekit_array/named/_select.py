"""Select by position, by exact label, or by an index carrier.

Each is written once, over the carrier's registered
`pipekit_ops.NamedArrayAdapter`; see `pipekit_array.named` for the rules they share.
"""

from __future__ import annotations

from typing import Any

from pipekit_array.named._dispatch import (
    adapter_of,
    axes_of,
    check_index_labels,
    fast_path,
    join_backends,
    namespace_of,
)


__all__ = ["isel", "sel", "take_along"]


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
    axes_of(adapter.dims(x), list(indexers), "isel")
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
    axes_of(adapter.dims(x), list(labels), "sel")
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
    fast = fast_path(adapter, "take_along")
    if fast is not None:
        return fast(x, index, dim)
    index_adapter = adapter_of(index, "take_along")
    dims = adapter.dims(x)
    (axis,) = axes_of(dims, (dim,), "take_along")
    rest = tuple(d for d in dims if d != dim)
    index_dims = index_adapter.dims(index)
    if not set(index_dims) <= set(rest):
        raise ValueError(f"take_along: index dims {index_dims} are not among {rest}.")
    # a JAX index next to NumPy data would be converted (or fail under jit)
    datas = [adapter.data(x), index_adapter.data(index)]
    data, idx = join_backends(datas, "take_along")
    xp = namespace_of(data)
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
