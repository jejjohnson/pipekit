"""`DimArray` — a raw array with named dims, and its reference adapter.

What a geo carrier becomes when a kernel consumes one of its spatial dims
(per-column statistics consume ``row`` and keep ``col``), and the plain
named carrier for tests and for libraries with none of their own
(PK-P7, decided 23 Oct 2026).

- **It holds** a raw array (NumPy, JAX or dask), one name per axis, and
  the static labels of any dim (band names, level values).
- **It is** a JAX pytree whose only leaf is the data: dims and labels are
  static, so a new set of values never recompiles. Under a bare
  ``jax.vmap`` (``in_axes=0``) the mapped dim is the first one: it is
  dropped inside and comes back as ``"batch"`` (`vmapped_dims`); to map a
  named dim, use `pipekit_array.map_over`.
- **It does not hold** a georeference, coordinates beyond labels, or
  attrs, and **does not do** alignment, arithmetic or selection by value;
  `DimArray.to_xarray` crosses to xarray when an analysis needs those.

`DimArrayAdapter` is the reference `pipekit.NamedArrayAdapter`: the test
kit (``pipekit_array.testing``) compares every other adapter against it.
"""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Mapping
from typing import Any

from pipekit import Signature


__all__ = ["DimArray", "DimArrayAdapter"]

_XARRAY_HINT = "DimArray.to_xarray needs xarray: pip install 'pipekit-array[xarray]'"


@dataclasses.dataclass(frozen=True, eq=False)
class DimArray:
    """A raw array with one name per axis and static labels on some dims.

    Args:
        data: A NumPy, JAX or dask array.
        dims: One unique name per axis of ``data``.
        labels: Static labels per dim, e.g. ``{"band": ("B11", "B12")}``;
            each has one entry per index of its dim. Dims left out have
            none. Values are stored as tuples.

    Raises:
        ValueError: The dims do not match the data's rank, repeat a name,
            or a label tuple's length differs from its dim's size.

    Example:
        >>> import numpy as np
        >>> mu = DimArray(np.zeros((3, 2)), ("col", "band"), {"band": ["B11", "B12"]})
        >>> mu.dims, mu.shape, mu.labels["band"]
        (('col', 'band'), (3, 2), ('B11', 'B12'))
    """

    data: Any
    dims: tuple[str, ...]
    labels: Mapping[str, tuple[Any, ...]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        dims = tuple(self.dims)
        shape = tuple(self.data.shape)
        if len(dims) != len(shape):
            raise ValueError(f"DimArray dims {dims} for data of shape {shape}.")
        if len(set(dims)) != len(dims):
            raise ValueError(f"DimArray dims must be unique, got {dims}.")
        labels: dict[str, tuple[Any, ...]] = {}
        for dim, values in dict(self.labels).items():
            if dim not in dims:
                raise ValueError(f"labels for {dim!r}, which is not in dims {dims}.")
            entries = tuple(values)
            size = shape[dims.index(dim)]
            if len(entries) != size:
                raise ValueError(
                    f"{len(entries)} labels for dim {dim!r} of size {size}."
                )
            labels[dim] = entries
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "labels", labels)

    @property
    def shape(self) -> tuple[int, ...]:
        """The data's shape."""
        return tuple(self.data.shape)

    @property
    def dtype(self) -> Any:
        """The data's dtype."""
        return self.data.dtype

    def to_xarray(self) -> Any:
        """An ``xr.DataArray`` with the same data and dims; labels become coordinates.

        Raises:
            ImportError: xarray is not installed (``pipekit-array[xarray]``).
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(_XARRAY_HINT) from exc
        coords = {dim: (dim, list(values)) for dim, values in self.labels.items()}
        return xr.DataArray(self.data, dims=self.dims, coords=coords)

    def __repr__(self) -> str:
        labelled = f", labels={sorted(self.labels)}" if self.labels else ""
        return f"DimArray[{' '.join(self.dims)}]{self.shape}{labelled}"


def _build(data: Any, dims: tuple[str, ...], labels: dict[str, Any]) -> DimArray:
    """A DimArray without validation, for pytree unflattening and rewraps."""
    d = object.__new__(DimArray)
    object.__setattr__(d, "data", data)
    object.__setattr__(d, "dims", tuple(dims))
    object.__setattr__(d, "labels", dict(labels))
    return d


def _positions(index: Any) -> Any:
    """An indexer with a boolean mask turned into the positions it selects.

    Raises:
        TypeError: A traced boolean mask (its positions are not static).
    """
    if isinstance(index, numbers.Integral | slice):
        return index
    dtype = getattr(index, "dtype", None)
    is_mask = (dtype is not None and dtype == bool) or (
        isinstance(index, (list, tuple))
        and len(index) > 0
        and all(isinstance(v, bool) for v in index)
    )
    if not is_mask:
        return index
    if "Tracer" in type(index).__name__:
        raise TypeError(
            "isel: a boolean mask under jit has no static size; select with "
            "positions, or mask the data instead."
        )
    import numpy as np

    return np.flatnonzero(np.asarray(index))


def _index_labels(values: tuple[Any, ...], index: Any) -> tuple[Any, ...] | None:
    """Labels after one positional index on their dim; None if the dim drops."""
    if isinstance(index, numbers.Integral):
        return None
    if isinstance(index, slice):
        return values[index]
    # an integer index array: one label per selected position
    return tuple(values[int(i)] for i in index)


class DimArrayAdapter:
    """The reference `pipekit.NamedArrayAdapter`, for `DimArray`."""

    def dims(self, x: DimArray) -> tuple[str, ...]:
        """``x.dims``."""
        return x.dims

    def data(self, x: DimArray) -> Any:
        """``x.data``."""
        return x.data

    def labels(self, x: DimArray, dim: str) -> tuple[Any, ...] | None:
        """The labels of ``dim``, or None."""
        return x.labels.get(dim)

    def isel(self, x: DimArray, indexers: dict[str, Any]) -> DimArray:
        """Select by position; integers drop their dim, labels follow the data.

        Raises:
            KeyError: An indexer names a dim ``x`` does not have.
        """
        unknown = sorted(set(indexers) - set(x.dims))
        if unknown:
            raise KeyError(f"isel: unknown dim(s) {unknown}; dims are {x.dims}.")
        normalized = {dim: _positions(index) for dim, index in indexers.items()}
        # ints and slices at once; each position array on its own axis, so
        # two arrays select independently instead of pairing up (outer indexing)
        basic = tuple(
            index if isinstance(index, numbers.Integral | slice) else slice(None)
            for index in (normalized.get(dim, slice(None)) for dim in x.dims)
        )
        data = x.data[basic]
        kept_dims = [
            dim
            for dim, index in zip(x.dims, basic, strict=True)
            if not isinstance(index, numbers.Integral)
        ]
        for dim, index in normalized.items():
            if isinstance(index, numbers.Integral | slice):
                continue
            axis = kept_dims.index(dim)
            data = data[(slice(None),) * axis + (index,)]
        key = tuple(normalized.get(dim, slice(None)) for dim in x.dims)
        dims: list[str] = []
        labels: dict[str, tuple[Any, ...]] = {}
        for dim, index in zip(x.dims, key, strict=True):
            if isinstance(index, numbers.Integral):
                continue
            dims.append(dim)
            if dim in x.labels:
                kept = _index_labels(x.labels[dim], index)
                if kept is not None:
                    labels[dim] = kept
        return _build(data, tuple(dims), labels)

    def rewrap(self, x: DimArray, data: Any, dims: tuple[str, ...]) -> DimArray:
        """A DimArray around new data; a kept dim keeps its labels if its size did.

        Raises:
            ValueError: ``dims`` does not match ``data``'s rank or repeats.
        """
        labels: dict[str, tuple[Any, ...]] = {}
        shape = tuple(data.shape)
        if len(shape) == len(dims):
            for dim, size in zip(dims, shape, strict=True):
                values = x.labels.get(dim)
                if values is not None and len(values) == size:
                    labels[dim] = values
        return DimArray(data, tuple(dims), labels)

    def check_compatible(self, *xs: DimArray) -> tuple[DimArray, ...]:
        """The arrays back, if every shared dim has one size and one set of labels.

        Raises:
            ValueError: A dim has two sizes, or two different label sets.
        """
        sizes: dict[str, int] = {}
        labels: dict[str, tuple[Any, ...]] = {}
        for i, x in enumerate(xs):
            for dim, size in zip(x.dims, x.shape, strict=True):
                if sizes.setdefault(dim, size) != size:
                    raise ValueError(
                        f"input {i}: dim {dim!r} has size {size}, not {sizes[dim]}."
                    )
                values = x.labels.get(dim)
                if values is None:
                    continue
                if labels.setdefault(dim, values) != values:
                    raise ValueError(
                        f"input {i}: dim {dim!r} has labels {values}, "
                        f"not {labels[dim]}."
                    )
        return xs

    def signature(self, x: DimArray) -> Signature:
        """Dims with their sizes, and the dtype's name."""
        dims = tuple(zip(x.dims, x.shape, strict=True))
        return Signature(dims=dims, dtype=str(x.dtype))


def vmapped_dims(dims: tuple[str, ...], shape: Any) -> tuple[str, ...]:
    """Static dims matched to leaves ``jax.vmap`` added or removed an axis from.

    ``vmap`` maps leaves but not static dims: inside, the data has lost
    its leading axis; outside, the results gained one. With the default
    ``in_axes=0`` / ``out_axes=0`` that axis is the first, so drop the
    first dims, or prepend ``"batch"`` (``"batch_1"``, … if taken). A
    shape that already matches, or no shape (a tracing placeholder),
    leaves ``dims`` alone. Map over any other axis with `map_over`.
    """
    if shape is None or not hasattr(shape, "__len__") or len(shape) == len(dims):
        return dims
    if len(shape) < len(dims):
        return dims[len(dims) - len(shape) :]
    added: list[str] = []
    k = 0
    while len(added) < len(shape) - len(dims):
        name = "batch" if k == 0 else f"batch_{k}"
        k += 1
        if name not in dims:
            added.append(name)
    return (*added, *dims)


def _register_pytree() -> bool:
    """Register `DimArray` with JAX when JAX is installed; never fail without it."""
    try:
        import jax
    except ImportError:
        return False

    def flatten(d: DimArray) -> tuple[tuple[Any], Any]:
        aux = (d.dims, tuple((dim, d.labels[dim]) for dim in d.dims if dim in d.labels))
        return (d.data,), aux

    def flatten_with_keys(d: DimArray) -> tuple[Any, Any]:
        (data,), aux = flatten(d)
        return ((jax.tree_util.GetAttrKey("data"), data),), aux

    def unflatten(aux: Any, leaves: Any) -> DimArray:
        # never validate: JAX unflattens with placeholder leaves while tracing
        dims, labels = aux
        data = leaves[0]
        dims = vmapped_dims(dims, getattr(data, "shape", None))
        return _build(data, dims, {d: v for d, v in labels if d in dims})

    jax.tree_util.register_pytree_with_keys(
        DimArray, flatten_with_keys, unflatten, flatten_func=flatten
    )
    return True


PYTREE_REGISTERED: bool = _register_pytree()
