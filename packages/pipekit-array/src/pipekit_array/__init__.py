"""pipekit-array — Array API operators on top of pipekit.

Operators: namespace dispatch, `MeanScalar`, `StackAlong` and
`ConcatenateAlong` (phase A of `docs/design/boundaries.md` §4), plus
`ApplyAlong` and `BatchAlong` below. The numeric QC asserts, the array
`Histogram` and `ModelOp` of the v0.1 catalogue are not built yet.

Named arrays: `DimArray` (data, dims, labels) and its reference
adapter `DimArrayAdapter`, registered with `pipekit_ops.register_adapter`
on import.

Dim-level functions (`dims`), on any registered carrier: `isel`, `sel`,
`transpose`, `reduce` (``where=`` masks), `argreduce`, `take_along`,
`map_over`, `stack`, `stack_items`, `concat`, `combine`; and (`dim_shapes`)
`rename`, `expand_dims`, `squeeze`, `broadcast_like`, `merge_dims`,
`split_dims`; as operators (`dim_operators`): `ISel`, `Sel`,
`Transpose`, `Reduce`, `ArgReduce`, `TakeAlong`, `MapOver`, `Combine`,
`Rename`, `ExpandDims`, `Squeeze`, `BroadcastLike`, `MergeDims`,
`SplitDims`; phase 2 (`dim_shifts`): `pad`, `roll`, `shift`, `diff`,
`cumreduce`, `where`, as `Pad`, `Roll`, `Shift`, `Diff`, `CumReduce`,
`Where`.
Windows (`dim_windows`): `window`, `coarsen`, with operators `Window`,
`Coarsen`; `ApplyAlong` runs an operator per item along a dim (the
serialisable `MapOver`), and `BatchAlong` (`batching`) runs one on chunks
of a dim.
Statistics and algebra (`dim_algebra`): `quantile`, `dot`, as `Quantile`,
`Dot`.
`StackAlong` and `ConcatenateAlong` take ``dim=`` for named carriers.

Sample tables (`rows`): `to_rows` flattens a carrier to a
``(n_samples, n_features)`` table by dim name and `from_rows` rebuilds
one; `Estimator` fits and applies a scikit-learn-style estimator on any
carrier over them (``[sklearn]``, imported lazily; OP-O3).

einx (``[einx]``): the submodule `pipekit_array.einx` holds `rearrange`,
`reduce`, `einsum`, `repeat` and `Einx`, with dim names as pattern axes on
any carrier (OP-O2); import it as ``from pipekit_array import einx``.

Batches (`batching`): `batch` groups carriers into fixed-size `Padded`
batches (``value``, ``valid``) along a new dim; ``pad="last" | "none" |
"drop"`` decides the short last batch.

Kernels (`kernel`): `KernelSpec` (a generalized-ufunc signature) and
`KernelOp`, which runs one L1 kernel on raw arrays and on every named
carrier; `BackendError` for mixed or unsupported backends.

xarray (``[xarray]``): the ``xr.DataArray`` adapter, `pipekit_array.xarray`,
loads lazily the first time a DataArray reaches `pipekit_ops.get_adapter`, so
importing pipekit-array never imports xarray.
"""

from __future__ import annotations

from pipekit_ops import register_adapter, register_lazy_adapter

from pipekit_array._namespace import array_namespace
from pipekit_array.batching import BatchAlong, Padded, batch
from pipekit_array.combinators import ConcatenateAlong, StackAlong
from pipekit_array.dim_algebra import dot, quantile
from pipekit_array.dim_operators import (
    ApplyAlong,
    ArgReduce,
    BroadcastLike,
    Coarsen,
    Combine,
    CumReduce,
    Diff,
    Dot,
    ExpandDims,
    ISel,
    MapOver,
    MergeDims,
    Pad,
    Quantile,
    Reduce,
    Rename,
    Roll,
    Sel,
    Shift,
    SplitDims,
    Squeeze,
    TakeAlong,
    Transpose,
    Where,
    Window,
)
from pipekit_array.dim_shapes import (
    broadcast_like,
    expand_dims,
    merge_dims,
    rename,
    split_dims,
    squeeze,
)
from pipekit_array.dim_shifts import cumreduce, diff, pad, roll, shift, where
from pipekit_array.dim_windows import coarsen, window
from pipekit_array.dimarray import DimArray, DimArrayAdapter
from pipekit_array.dims import (
    BackendError,
    argreduce,
    combine,
    concat,
    isel,
    map_over,
    reduce,
    sel,
    stack,
    stack_items,
    take_along,
    transpose,
)
from pipekit_array.estimator import Estimator
from pipekit_array.kernel import KernelOp, KernelSpec
from pipekit_array.reductions import MeanScalar
from pipekit_array.rows import RowLayout, from_rows, to_rows


__version__ = "0.1.1"  # x-release-please-version

register_adapter(DimArray, DimArrayAdapter())


def _load_xarray_adapter() -> None:
    from pipekit_array.xarray import register

    register()


register_lazy_adapter("xarray.", _load_xarray_adapter)

__all__ = [
    "ApplyAlong",
    "ArgReduce",
    "BackendError",
    "BatchAlong",
    "BroadcastLike",
    "Coarsen",
    "Combine",
    "ConcatenateAlong",
    "CumReduce",
    "Diff",
    "DimArray",
    "DimArrayAdapter",
    "Dot",
    "Estimator",
    "ExpandDims",
    "ISel",
    "KernelOp",
    "KernelSpec",
    "MapOver",
    "MeanScalar",
    "MergeDims",
    "Pad",
    "Padded",
    "Quantile",
    "Reduce",
    "Rename",
    "Roll",
    "RowLayout",
    "Sel",
    "Shift",
    "SplitDims",
    "Squeeze",
    "StackAlong",
    "TakeAlong",
    "Transpose",
    "Where",
    "Window",
    "argreduce",
    "array_namespace",
    "batch",
    "broadcast_like",
    "coarsen",
    "combine",
    "concat",
    "cumreduce",
    "diff",
    "dot",
    "expand_dims",
    "from_rows",
    "isel",
    "map_over",
    "merge_dims",
    "pad",
    "quantile",
    "reduce",
    "rename",
    "roll",
    "sel",
    "shift",
    "split_dims",
    "squeeze",
    "stack",
    "stack_items",
    "take_along",
    "to_rows",
    "transpose",
    "where",
    "window",
]
