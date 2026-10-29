"""Sample tables by dim name: `to_rows` and `from_rows`.

A per-sample model (scikit-learn, a pixel classifier) wants a 2-D table:
one row per sample, one column per feature. `to_rows` turns any named
carrier into one, choosing the samples by dim name, and `from_rows` turns
a table of rows back into a carrier like the input, on the data's own
backend (NumPy, JAX under ``jit``, dask):

    x (time, band, y, x) --to_rows(sample_dims=("time", "y", "x"))-->
        rows (time·y·x, band), layout
    rows (n, k) --from_rows(..., feature_dim="component")-->
        (time, component, y, x)

Samples are the C-order product of the sample dims, taken in the
carrier's dim order; features likewise over every other dim. Like the rest
of the dim-level functions, these never read coordinate values (GA-Q3):
labels and geometry come back through the carrier's adapter, which keeps
those of the dims that survive with their size.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

from pipekit_array.dims import adapter_of, namespace_of, transpose


__all__ = ["RowLayout", "from_rows", "to_rows"]


@dataclasses.dataclass(frozen=True)
class RowLayout:
    """How `to_rows` flattened a carrier, so `from_rows` can rebuild one.

    Args:
        dims: The carrier's dims, in its order.
        sizes: The size of each dim, in the same order.
        sample_dims: The dims that became rows, in the carrier's order.
        feature_dims: The dims that became columns, in the carrier's order.
        template: The carrier to rewrap like (left out of equality and repr).
    """

    dims: tuple[str, ...]
    sizes: tuple[int, ...]
    sample_dims: tuple[str, ...]
    feature_dims: tuple[str, ...]
    template: Any = dataclasses.field(default=None, compare=False, repr=False)

    @property
    def n_samples(self) -> int:
        """The number of rows: the product of the sample dims' sizes."""
        return math.prod(self._size(d) for d in self.sample_dims)

    @property
    def n_features(self) -> int:
        """The number of columns: the product of the feature dims' sizes."""
        return math.prod(self._size(d) for d in self.feature_dims)

    def _size(self, dim: str) -> int:
        return self.sizes[self.dims.index(dim)]


def to_rows(x: Any, sample_dims: Sequence[str]) -> tuple[Any, RowLayout]:
    """``x`` as a ``(n_samples, n_features)`` table, and how to undo it.

    Args:
        x: A named carrier.
        sample_dims: The dims whose entries are samples (rows); every other
            dim of ``x`` is a feature (column). At least one.

    Returns:
        The table, on ``x``'s backend, and the `RowLayout` for `from_rows`.
        With no feature dims the table has one column.

    Raises:
        TypeError: ``x`` is not a named carrier.
        ValueError: No sample dims, a repeated one, or one ``x`` lacks.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> scene = DimArray(np.arange(24.0).reshape(2, 3, 4), ("band", "y", "x"))
        >>> rows, layout = to_rows(scene, sample_dims=("y", "x"))
        >>> rows.shape, layout.feature_dims
        ((12, 2), ('band',))
    """
    adapter = adapter_of(x, "to_rows")
    dims = tuple(adapter.dims(x))
    wanted = tuple(sample_dims)
    if not wanted:
        raise ValueError("to_rows needs at least one sample dim.")
    if len(set(wanted)) != len(wanted):
        raise ValueError(f"to_rows: sample dims {wanted} repeat a name.")
    missing = [d for d in wanted if d not in dims]
    if missing:
        raise ValueError(f"to_rows: sample dims {missing} are not in {dims}.")
    samples = tuple(d for d in dims if d in wanted)
    features = tuple(d for d in dims if d not in wanted)
    sizes = tuple(int(n) for n in adapter.data(x).shape)
    layout = RowLayout(dims, sizes, samples, features, template=x)
    data = adapter.data(transpose(x, (*samples, *features)))
    xp = namespace_of(data)
    rows = xp.reshape(data, (layout.n_samples, layout.n_features))
    return rows, layout


def from_rows(rows: Any, layout: RowLayout, *, feature_dim: str | None = None) -> Any:
    """A carrier like ``layout.template`` from a table of rows.

    Args:
        rows: ``(n_samples, n_features)`` gives back the original feature
            dims; ``(n_samples, k)`` with ``feature_dim`` gives one new dim
            of size ``k`` in their place (PCA components, class
            probabilities); ``(n_samples,)`` gives the sample dims only
            (a label or a score per sample).
        layout: From `to_rows`.
        feature_dim: The name of the new feature dim, for a table whose
            columns are not the original features.

    Returns:
        A carrier with ``layout``'s dims in their original order, the new
        feature dim (if any) where the first feature dim was, rewrapped by
        the template's adapter.

    Raises:
        ValueError: The rows do not match the layout, or a new feature dim
            is needed and ``feature_dim`` is missing or already a dim.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> scene = DimArray(np.arange(24.0).reshape(2, 3, 4), ("band", "y", "x"))
        >>> rows, layout = to_rows(scene, sample_dims=("y", "x"))
        >>> from_rows(rows[:, :1] * 0, layout, feature_dim="score").dims
        ('score', 'y', 'x')
    """
    template = layout.template
    if template is None:
        raise ValueError("from_rows needs a layout from to_rows (it has no template).")
    adapter = adapter_of(template, "from_rows")
    shape = tuple(int(n) for n in rows.shape)
    if len(shape) not in (1, 2) or shape[0] != layout.n_samples:
        raise ValueError(
            f"from_rows: rows of shape {shape} do not fit {layout.n_samples} samples "
            f"over {layout.sample_dims}."
        )
    sample_sizes = tuple(layout._size(d) for d in layout.sample_dims)
    if len(shape) == 1:
        block_dims = layout.sample_dims
        block_shape = sample_sizes
        order = layout.sample_dims
    elif feature_dim is None:
        if shape[1] != layout.n_features:
            raise ValueError(
                f"from_rows: {shape[1]} columns are not the {layout.n_features} "
                f"features over {layout.feature_dims}; pass feature_dim= to name "
                "the new columns."
            )
        block_dims = (*layout.sample_dims, *layout.feature_dims)
        block_shape = (*sample_sizes, *(layout._size(d) for d in layout.feature_dims))
        order = layout.dims
    else:
        if feature_dim in layout.dims:
            raise ValueError(
                f"from_rows: feature_dim {feature_dim!r} is already a dim of "
                f"{layout.dims}."
            )
        block_dims = (*layout.sample_dims, feature_dim)
        block_shape = (*sample_sizes, shape[1])
        order = _with_new_feature_dim(layout, feature_dim)
    xp = namespace_of(rows)
    block = xp.reshape(rows, block_shape)
    axes = tuple(block_dims.index(d) for d in order)
    if axes != tuple(range(len(axes))):
        block = xp.permute_dims(block, axes)
    return adapter.rewrap(template, block, tuple(order))


def _with_new_feature_dim(layout: RowLayout, feature_dim: str) -> tuple[str, ...]:
    """The layout's dims with ``feature_dim`` replacing the feature dims."""
    if not layout.feature_dims:
        return (feature_dim, *layout.dims)
    first = layout.dims.index(layout.feature_dims[0])
    kept = [d for d in layout.dims if d not in layout.feature_dims]
    before = [d for d in layout.dims[:first] if d in kept]
    after = [d for d in layout.dims[first:] if d in kept]
    return (*before, feature_dim, *after)
