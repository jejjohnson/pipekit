"""Functions of named carriers: arrays with dimension names.

A carrier is a `DimArray`, an ``xr.DataArray`` (``[xarray]``), a geoarray
carrier: anything with a registered `pipekit_ops.NamedArrayAdapter`. Each
function is written once, over that adapter, and behaves the same on every
carrier. The rule for joining this list (GA-Q3): a function never reads
coordinate values or geometry. ``resample``, ``groupby`` and ``rolling``
stay xarray's; reprojection and sampling at points stay in geoarray.

- **Select**: `isel`, `sel`, `take_along`.
- **Shape**: `transpose`, `rename`, `expand_dims`, `squeeze`, `broadcast_like`,
  `merge_dims`, `split_dims`.
- **Reduce**: `reduce`, `argreduce`, `cumreduce`, `quantile`, `dot`.
- **Shift**: `pad`, `roll`, `shift`, `diff`, `where`.
- **Window**: `window`, `coarsen`.
- **Join**: `stack`, `stack_items`, `concat`, `combine`.
- **Map**: `map_over`.
- **Batches and tables**: `batch`, `Padded`, `to_rows`, `from_rows`, `RowLayout`.
- **The plain carrier**: `DimArray`, `DimArrayAdapter`; `core_dims` and
  `vmapped_dims` for a carrier that is a pytree.
- **Dispatch**: `array_namespace`, `join_backends`.

**Fast paths.** An adapter may provide a method of the same name (the
xarray adapter's ``reduce`` stays lazy on dask; geoarray's ``map_over``
is one ``jax.vmap`` over data and transform rows). It is called with the
same arguments instead of the generic path; the test kit
(``pipekit_array.testing``) proves the two agree.

The `Operator` forms live in `pipekit_array.operators`, in files of the same
names. This area needs no extra.
"""

from __future__ import annotations

from pipekit_array.named._batches import (
    PAD_POLICIES,
    Padded,
    PadPolicy,
    batch,
)
from pipekit_array.named._carrier import (
    DimArray,
    DimArrayAdapter,
    core_dims,
    vmapped_dims,
)
from pipekit_array.named._dispatch import (
    array_namespace,
    join_backends,
)
from pipekit_array.named._join import (
    combine,
    concat,
    stack,
    stack_items,
)
from pipekit_array.named._map import map_over
from pipekit_array.named._reduce import (
    METHODS,
    argreduce,
    cumreduce,
    dot,
    quantile,
    reduce,
)
from pipekit_array.named._rows import (
    RowLayout,
    from_rows,
    to_rows,
)
from pipekit_array.named._select import (
    isel,
    sel,
    take_along,
)
from pipekit_array.named._shape import (
    broadcast_like,
    expand_dims,
    merge_dims,
    rename,
    split_dims,
    squeeze,
    transpose,
)
from pipekit_array.named._shift import (
    PAD_MODES,
    diff,
    pad,
    roll,
    shift,
    where,
)
from pipekit_array.named._window import (
    BOUNDARIES,
    coarsen,
    window,
)


__all__ = [
    "BOUNDARIES",
    "METHODS",
    "PAD_MODES",
    "PAD_POLICIES",
    "DimArray",
    "DimArrayAdapter",
    "PadPolicy",
    "Padded",
    "RowLayout",
    "argreduce",
    "array_namespace",
    "batch",
    "broadcast_like",
    "coarsen",
    "combine",
    "concat",
    "core_dims",
    "cumreduce",
    "diff",
    "dot",
    "expand_dims",
    "from_rows",
    "isel",
    "join_backends",
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
    "vmapped_dims",
    "where",
    "window",
]
