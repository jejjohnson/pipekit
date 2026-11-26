"""The ready-made operators.

Blocks (`Identity`, `Const`, `Lambda`, `Sink`), dict plumbing (`Get` … `ValMap`), iterables (`Filter` … `SortBy`), `Fanout`, control flow (`Branch`, `Switch`, `Try`, `Coalesce`, `Retry`), flow limits (`Timeout`, `RateLimit`, `Until`), caching (`Cache`, `Memoize`) and quality control (`Quarantine` and the `Assert*` checks).
"""

from __future__ import annotations

from pipekit_ops.operators._blocks import (
    Const,
    Identity,
    Lambda,
    Sink,
)
from pipekit_ops.operators._cache import (
    Cache,
    Memoize,
)
from pipekit_ops.operators._combine import (
    Fanout,
)
from pipekit_ops.operators._control import (
    Branch,
    Coalesce,
    Retry,
    Switch,
    Try,
)
from pipekit_ops.operators._dicts import (
    Assoc,
    Dissoc,
    Get,
    GetIn,
    Merge,
    Pick,
    Rename,
    UpdateIn,
    ValMap,
)
from pipekit_ops.operators._flow import (
    RateLimit,
    Timeout,
    Until,
)
from pipekit_ops.operators._qc import (
    AssertCallable,
    AssertDType,
    AssertHasAttribute,
    AssertShape,
    AssertSignature,
    AssertThat,
    QCError,
    Quarantine,
)
from pipekit_ops.operators._seq import (
    Concat,
    Count,
    CountBy,
    Filter,
    GroupBy,
    Not,
    PartitionAll,
    Reduce,
    Remove,
    SortBy,
    Take,
    Unique,
    Zip,
)


__all__ = [
    "AssertCallable",
    "AssertDType",
    "AssertHasAttribute",
    "AssertShape",
    "AssertSignature",
    "AssertThat",
    "Assoc",
    "Branch",
    "Cache",
    "Coalesce",
    "Concat",
    "Const",
    "Count",
    "CountBy",
    "Dissoc",
    "Fanout",
    "Filter",
    "Get",
    "GetIn",
    "GroupBy",
    "Identity",
    "Lambda",
    "Memoize",
    "Merge",
    "Not",
    "PartitionAll",
    "Pick",
    "QCError",
    "Quarantine",
    "RateLimit",
    "Reduce",
    "Remove",
    "Rename",
    "Retry",
    "Sink",
    "SortBy",
    "Switch",
    "Take",
    "Timeout",
    "Try",
    "Unique",
    "Until",
    "UpdateIn",
    "ValMap",
    "Zip",
]
