"""pipekit-array — Array API operators on top of pipekit.

Phase A surface: namespace dispatch + reductions + stack/concat
combinators. The full v0.1 catalogue (12 operators) lands across
phases A-D per `docs/design/boundaries.md` §4.

Named arrays: `DimArray` (data, dims, labels) and its reference
adapter `DimArrayAdapter`, registered with `pipekit.register_adapter`
on import.

Dim-level functions (`dims`), on any registered carrier: `isel`, `sel`,
`transpose`, `reduce`, `argreduce`, `take_along`, `map_over`, `stack`,
`concat`, `combine`; as operators (`dim_operators`): `ISel`, `Sel`,
`Transpose`, `Reduce`, `ArgReduce`, `TakeAlong`, `MapOver`, `Combine`.
`StackAlong` and `ConcatenateAlong` take ``dim=`` for named carriers.
"""

from __future__ import annotations

from pipekit import register_adapter

from pipekit_array._namespace import array_namespace
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
    argreduce,
    combine,
    concat,
    isel,
    map_over,
    reduce,
    sel,
    stack,
    take_along,
    transpose,
)
from pipekit_array.reductions import MeanScalar


__version__ = "0.0.3"  # x-release-please-version

register_adapter(DimArray, DimArrayAdapter())

__all__ = [
    "ArgReduce",
    "Combine",
    "ConcatenateAlong",
    "DimArray",
    "DimArrayAdapter",
    "ISel",
    "MapOver",
    "MeanScalar",
    "Reduce",
    "Sel",
    "StackAlong",
    "TakeAlong",
    "Transpose",
    "argreduce",
    "array_namespace",
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
