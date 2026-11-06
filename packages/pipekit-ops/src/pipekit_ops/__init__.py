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
- **Group E — observers** (`observe`):
  `Tap`, `Snapshot`, `ShapeTrace`, `Profile`, `Histogram`.
- **Group F — combination** (`combine`): `Fanout`.
- **Group G — caching** (`cache`): `Cache`, `Memoize`.
- **Group H — quality control** (`qc`): `Quarantine`, `AssertShape`,
  `AssertDType`, `AssertHasAttribute`, `AssertCallable`, `QCError`.
- **Group I — shape inference** (`signature`): `Signature`.
- **Group J — parallelism** (`parallel`): `ThreadMap`, `ProcessMap`,
  `AsyncMap`, `BatchedMap`, `check_pickleable`.
- *(Group K is reserved in the master plan for a disk-cache backend;
  nothing ships under it yet.)*
- **Group L — serialisation** (`serial`): `dumps`, `loads`,
  `loads_sandboxed`, `from_state_sandboxed`, `register`, `unregister`,
  `allowed`.
- **Group M — state primitives** (`state`): `StatefulOperator`,
  `CarryState`.
- **Group N — structural protocols** (`protocols`): `Predictor`,
  `FittableTransformer`.
- **Named-array adapters** (`adapters`): `NamedArrayAdapter`,
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


__all__ = [
    "AssertCallable",
    "AssertDType",
    "AssertHasAttribute",
    "AssertShape",
    "AsyncMap",
    "BackendError",
    "BatchedMap",
    "Branch",
    "Cache",
    "Carrier",
    "CarryState",
    "Coalesce",
    "ConfigMixin",
    "Const",
    "Fanout",
    "FittableTransformer",
    "Graph",
    "Histogram",
    "Identity",
    "Input",
    "Lambda",
    "Memoize",
    "MissingAdapterError",
    "NamedArrayAdapter",
    "Node",
    "Operator",
    "Predictor",
    "ProcessMap",
    "Profile",
    "QCError",
    "Quarantine",
    "Retry",
    "Sequential",
    "ShapeTrace",
    "Signature",
    "Sink",
    "Snapshot",
    "StatefulOperator",
    "Switch",
    "Tap",
    "ThreadMap",
    "Try",
    "allowed",
    "backend_name",
    "backend_of",
    "check_pickleable",
    "complement",
    "compose",
    "compose_left",
    "compute_output_signature",
    "dumps",
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
    "unregister",
    "unregister_adapter",
    "upgrade_state",
]

__version__ = "0.1.5"  # x-release-please-version
