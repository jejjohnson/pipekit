"""pipekit — carrier-agnostic operator-graph framework.

The v0.1 public surface, grouped by module:

- **Group A — foundations** (`_base/`): `Operator`, `ConfigMixin`,
  `Sequential`, `Graph` + `Input` + `Node`.
- **Group B — composition convenience** (`compose`):
  `pipe`, `compose`, `compose_left`, `complement`, `juxt`.
- **Group C — building blocks** (`blocks`):
  `Identity`, `Const`, `Lambda`, `Sink`.
- **Group D — control flow** (`control`):
  `Branch`, `Switch`, `Try`, `Coalesce`, `Retry`.
- **Flow limits** (`flow`): `Timeout`, `RateLimit`, `Until`.
- **Group E — observers** (`observe`):
  `Tap`, `Snapshot`, `ShapeTrace`, `Profile`, `Histogram`.
- **Group F — combination** (`combine`): `Fanout`.
- **Dict plumbing** (`dicts`): `Get`, `GetIn`, `Pick`, `Dissoc`, `Rename`,
  `Merge`, `Assoc`, `UpdateIn`, `ValMap`.
- **Iterables** (`seq`): `Not`, `Filter`, `Remove`, `Reduce`,
  `PartitionAll`, `Concat`, `Zip`, `GroupBy`, `CountBy`, `Count`, `Unique`,
  `Take`, `SortBy`.
- **Group G — caching** (`cache`): `Cache`, `Memoize`.
- **Group H — quality control** (`qc`): `Quarantine`, `AssertShape`,
  `AssertDType`, `AssertHasAttribute`, `AssertCallable`, `QCError`.
- **Group I — shape inference** (`signature`): `Signature`.
- **Group J — parallelism** (`parallel`): `ThreadMap`, `ProcessMap`,
  `AsyncMap`, `BatchedMap`, `check_pickleable`.
- **Execution** (`execution`): `Map`, `executor_scope`, `MapError`,
  `Failed`, `Prefetch`, `Fold` — one engine over serial, thread, process,
  async and any borrowed `concurrent.futures.Executor`.
- **Group K — stores** (`store`): `Store`, `MemoryStore`,
  `DirectoryStore`, `store_scope` — where ``Cache(op, store=name)`` keeps
  its entries, bound around the call; a `DirectoryStore` survives a
  restart and is shared by processes on one file system.
- **Group L — serialisation** (`serial`): `dumps`, `loads`,
  `loads_sandboxed`, `from_state_sandboxed`, `register`, `unregister`,
  `allowed`.
- **Group M — state primitives** (`state`): `StatefulOperator`,
  `CarryState`.
- **Group N — structural protocols** (`protocols`): `Predictor`,
  `FittableTransformer`.
- **Named-array adapters** (`adapters`): `NamedArrayAdapter`,
  `LabelledAdapter` (the optional ``relabel``),
  `register_adapter`, `register_lazy_adapter`, `unregister_adapter`,
  `get_adapter`,
  `signature_of`, `MissingAdapterError` — how carrier-agnostic code
  reaches a carrier with named dims; `backend_of`, `backend_name` and
  `BackendError` — the array backend an input sits on, which
  ``Operator.backends`` checks.
- **Import hooks** (`imports`): `on_import(name, callback)` runs code once
  a module is imported without importing it (JAX pytree registration).

Hashing: `pipeline_hash(op)` is the content hash of a whole pipeline,
the one the pipekit-experiment registries store under. `pipekit_ops.hashing`
also holds the stable-hash primitives (`stable_json`, `stable_repr`,
`sha256_hex`) reused by `Cache` and the pipekit-train datasets.

Testing: `pipekit_ops.testing.assert_round_trip(op, *inputs)` checks the
config round-trip contract (same `pipeline_hash`, same output after
``loads(dumps(op))``) from a library's own test suite.

See the master plan (Report 2) for the full v0.1 inventory.
"""

from pipekit_ops._base.graph import Graph, Input, Node
from pipekit_ops._base.operator import Carrier, ConfigMixin, Operator
from pipekit_ops._base.sequential import Sequential
from pipekit_ops.adapters import (
    BackendError,
    LabelledAdapter,
    MissingAdapterError,
    NamedArrayAdapter,
    backend_name,
    backend_of,
    get_adapter,
    register_adapter,
    register_lazy_adapter,
    signature_of,
    unregister_adapter,
)
from pipekit_ops.blocks import Const, Identity, Lambda, Sink
from pipekit_ops.cache import Cache, Memoize
from pipekit_ops.combine import Fanout
from pipekit_ops.compose import complement, compose, compose_left, juxt, pipe
from pipekit_ops.control import Branch, Coalesce, Retry, Switch, Try
from pipekit_ops.dicts import (
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
from pipekit_ops.execution import (
    Failed,
    Fold,
    Map,
    MapError,
    Prefetch,
    executor_scope,
)
from pipekit_ops.flow import RateLimit, Timeout, Until
from pipekit_ops.hashing import pipeline_hash
from pipekit_ops.imports import on_import
from pipekit_ops.observe import Histogram, Profile, ShapeTrace, Snapshot, Tap
from pipekit_ops.parallel import (
    AsyncMap,
    BatchedMap,
    ProcessMap,
    ThreadMap,
    check_pickleable,
)
from pipekit_ops.protocols import FittableTransformer, Predictor
from pipekit_ops.qc import (
    AssertCallable,
    AssertDType,
    AssertHasAttribute,
    AssertShape,
    QCError,
    Quarantine,
)
from pipekit_ops.seq import (
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
from pipekit_ops.serial import (
    allowed,
    dumps,
    from_state_sandboxed,
    loads,
    loads_sandboxed,
    register,
    unregister,
    upgrade_state,
)
from pipekit_ops.signature import Signature, compute_output_signature
from pipekit_ops.state import CarryState, StatefulOperator
from pipekit_ops.store import DirectoryStore, MemoryStore, Store, store_scope


__all__ = [
    "AssertCallable",
    "AssertDType",
    "AssertHasAttribute",
    "AssertShape",
    "Assoc",
    "AsyncMap",
    "BackendError",
    "BatchedMap",
    "Branch",
    "Cache",
    "Carrier",
    "CarryState",
    "Coalesce",
    "Concat",
    "ConfigMixin",
    "Const",
    "Count",
    "CountBy",
    "DirectoryStore",
    "Dissoc",
    "Failed",
    "Fanout",
    "Filter",
    "FittableTransformer",
    "Fold",
    "Get",
    "GetIn",
    "Graph",
    "GroupBy",
    "Histogram",
    "Identity",
    "Input",
    "LabelledAdapter",
    "Lambda",
    "Map",
    "MapError",
    "Memoize",
    "MemoryStore",
    "Merge",
    "MissingAdapterError",
    "NamedArrayAdapter",
    "Node",
    "Not",
    "Operator",
    "PartitionAll",
    "Pick",
    "Predictor",
    "Prefetch",
    "ProcessMap",
    "Profile",
    "QCError",
    "Quarantine",
    "RateLimit",
    "Reduce",
    "Remove",
    "Rename",
    "Retry",
    "Sequential",
    "ShapeTrace",
    "Signature",
    "Sink",
    "Snapshot",
    "SortBy",
    "StatefulOperator",
    "Store",
    "Switch",
    "Take",
    "Tap",
    "ThreadMap",
    "Timeout",
    "Try",
    "Unique",
    "Until",
    "UpdateIn",
    "ValMap",
    "Zip",
    "allowed",
    "backend_name",
    "backend_of",
    "check_pickleable",
    "complement",
    "compose",
    "compose_left",
    "compute_output_signature",
    "dumps",
    "executor_scope",
    "from_state_sandboxed",
    "get_adapter",
    "juxt",
    "loads",
    "loads_sandboxed",
    "on_import",
    "pipe",
    "pipeline_hash",
    "register",
    "register_adapter",
    "register_lazy_adapter",
    "signature_of",
    "store_scope",
    "unregister",
    "unregister_adapter",
    "upgrade_state",
]

__version__ = "0.2.0"  # x-release-please-version
