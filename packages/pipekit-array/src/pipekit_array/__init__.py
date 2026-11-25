"""pipekit-array — arrays on any library, with or without dimension names.

Six areas, each with one job. Import a name from its area, never from a file in
it; an area imports only from the ones before it, and `pipekit_array.testing` from
all of them (ADR A30).

- `pipekit_array.arrays` (``[numpy,compat]``): functions of plain arrays. The
  array backend of a library of formulas (`floating_inputs`, `like`), indexed
  updates, loops, checks, safe roots, masked reductions, records.
- `pipekit_array.named`: functions of carriers with dimension names, on any
  registered `pipekit_ops.NamedArrayAdapter`: `isel`, `reduce`, `rename`, `pad`,
  `window`, `stack`, `map_over`, `batch`, `to_rows`, …; and `DimArray`, the plain
  carrier.
- `pipekit_array.kernels`: one array function, written once, on raw arrays and on
  every carrier: `KernelSpec`, `KernelOp`, and the decorator `elementwise`.
- `pipekit_array.operators`: the named functions as `pipekit_ops.Operator`s:
  `ISel`, `Reduce`, `Window`, `ApplyAlong`, …
- `pipekit_array.bridges`: other libraries on named carriers. ``xarray``
  (``[xarray]``), ``einx`` (``[einx]``) and ``sklearn`` (``[sklearn]``).
- `pipekit_array.testing`: checks for the test suites of other libraries.

This package re-exports the named functions, the kernels and the operators.
`pipekit_array.arrays` has names of its own (its `pad` is for plain arrays), and is
imported by name.

The ``xr.DataArray`` adapter loads lazily the first time a DataArray reaches
`pipekit_ops.get_adapter`, so importing pipekit-array never imports xarray.
"""

from __future__ import annotations

from pipekit_ops import BackendError, register_lazy_adapter

from pipekit_array.bridges.sklearn import Estimator
from pipekit_array.kernels import (
    KernelOp,
    KernelSpec,
)
from pipekit_array.named import (
    DimArray,
    DimArrayAdapter,
    Padded,
    RowLayout,
    argreduce,
    array_namespace,
    batch,
    broadcast_like,
    coarsen,
    combine,
    concat,
    cumreduce,
    diff,
    dot,
    expand_dims,
    from_rows,
    isel,
    map_over,
    merge_dims,
    pad,
    quantile,
    reduce,
    rename,
    roll,
    sel,
    shift,
    split_dims,
    squeeze,
    stack,
    stack_items,
    take_along,
    to_rows,
    transpose,
    where,
    window,
)
from pipekit_array.operators import (
    ApplyAlong,
    ArgReduce,
    BatchAlong,
    BroadcastLike,
    Coarsen,
    Combine,
    ConcatenateAlong,
    CumReduce,
    Diff,
    Dot,
    ExpandDims,
    ISel,
    MapOver,
    MeanScalar,
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
    StackAlong,
    TakeAlong,
    Transpose,
    Where,
    Window,
)


__version__ = "0.2.0"  # x-release-please-version


def _load_xarray_adapter() -> None:
    from pipekit_array.bridges.xarray import register

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
