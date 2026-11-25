"""The functions of `pipekit_array.named` as `Operator`s, for pipelines and config.

Each wraps one function and computes its output `Signature` without running.
Operators that take only names and numbers round-trip through ``dumps`` /
``loads``; those that hold a user function (`MapOver`, `Combine`) set
``forbid_in_yaml``, like `pipekit_ops.Lambda`.

`Reduce` and `ArgReduce` take the reduction by its array-namespace name
(``"mean"``, ``"argmax"``) so they serialise; the function forms also take a
callable. `MeanScalar`, `StackAlong` and `ConcatenateAlong` run on raw arrays.

A saved pipeline records an operator under ``pipekit_array``, wherever its file
is (ADR A30).
"""

from __future__ import annotations

from pipekit_array.operators._join import (
    Combine,
    ConcatenateAlong,
    StackAlong,
)
from pipekit_array.operators._map import (
    ApplyAlong,
    BatchAlong,
    MapOver,
)
from pipekit_array.operators._reduce import (
    ArgReduce,
    CumReduce,
    Dot,
    MeanScalar,
    Quantile,
    Reduce,
)
from pipekit_array.operators._select import (
    ISel,
    Sel,
    TakeAlong,
)
from pipekit_array.operators._shape import (
    BroadcastLike,
    ExpandDims,
    MergeDims,
    Rename,
    SplitDims,
    Squeeze,
    Transpose,
)
from pipekit_array.operators._shift import (
    Diff,
    Pad,
    Roll,
    Shift,
    Where,
)
from pipekit_array.operators._window import (
    Coarsen,
    Window,
)


__all__ = [
    "ApplyAlong",
    "ArgReduce",
    "BatchAlong",
    "BroadcastLike",
    "Coarsen",
    "Combine",
    "ConcatenateAlong",
    "CumReduce",
    "Diff",
    "Dot",
    "ExpandDims",
    "ISel",
    "MapOver",
    "MeanScalar",
    "MergeDims",
    "Pad",
    "Quantile",
    "Reduce",
    "Rename",
    "Roll",
    "Sel",
    "Shift",
    "SplitDims",
    "Squeeze",
    "StackAlong",
    "TakeAlong",
    "Transpose",
    "Where",
    "Window",
]
