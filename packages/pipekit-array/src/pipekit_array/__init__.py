"""pipekit-array — Array API operators on top of pipekit.

Phase A surface: namespace dispatch + reductions + stack/concat
combinators. The full v0.1 catalogue (12 operators) lands across
phases A-D per `docs/design/boundaries.md` §4.

Named arrays: `DimArray` (data, dims, labels) and its reference
adapter `DimArrayAdapter`, registered with `pipekit.register_adapter`
on import.

Dim-level functions (`dims`), on any registered carrier: `isel`, `sel`,
`transpose`, `reduce` (``where=`` masks), `argreduce`, `take_along`,
`map_over`, `stack`, `stack_items`, `concat`, `combine`; as operators
(`dim_operators`): `ISel`, `Sel`,
`Transpose`, `Reduce`, `ArgReduce`, `TakeAlong`, `MapOver`, `Combine`.
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
loads lazily the first time a DataArray reaches `pipekit.get_adapter`, so
importing pipekit-array never imports xarray.
"""

from __future__ import annotations

from pipekit import register_adapter, register_lazy_adapter

from pipekit_array._namespace import array_namespace
from pipekit_array.batching import Padded, batch
from pipekit_array.combinators import ConcatenateAlong, StackAlong
from pipekit_array.dim_operators import (
    ArgReduce,
    Combine,
    ISel,
    MapOver,
    Reduce,
    Sel,
    TakeAlong,
    Transpose,
)
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


__version__ = "0.0.8"  # x-release-please-version

register_adapter(DimArray, DimArrayAdapter())


def _load_xarray_adapter() -> None:
    from pipekit_array.xarray import register

    register()


register_lazy_adapter("xarray.", _load_xarray_adapter)

__all__ = [
    "ArgReduce",
    "BackendError",
    "Combine",
    "ConcatenateAlong",
    "DimArray",
    "DimArrayAdapter",
    "Estimator",
    "ISel",
    "KernelOp",
    "KernelSpec",
    "MapOver",
    "MeanScalar",
    "Padded",
    "Reduce",
    "RowLayout",
    "Sel",
    "StackAlong",
    "TakeAlong",
    "Transpose",
    "argreduce",
    "array_namespace",
    "batch",
    "combine",
    "concat",
    "from_rows",
    "isel",
    "map_over",
    "reduce",
    "sel",
    "stack",
    "stack_items",
    "take_along",
    "to_rows",
    "transpose",
]
