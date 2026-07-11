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
  `loads_sandboxed`, `register`, `unregister`, `allowed`.
- **Group M — state primitives** (`state`): `StatefulOperator`,
  `CarryState`.
- **Group N — structural protocols** (`protocols`): `Predictor`,
  `FittableTransformer`.

Shared internals: `pipekit.hashing` holds the stable-hash primitives
(`stable_json`, `stable_repr`, `sha256_hex`) reused by `Cache`, the
pipekit-train datasets, and the pipekit-experiment registries.

See the master plan (Report 2) for the full v0.1 inventory.
"""

from pipekit._base.graph import Graph, Input, Node
from pipekit._base.operator import Carrier, ConfigMixin, Operator
from pipekit._base.sequential import Sequential
from pipekit.blocks import Const, Identity, Lambda, Sink
from pipekit.cache import Cache, Memoize
from pipekit.combine import Fanout
from pipekit.compose import complement, compose, compose_left, juxt, pipe
from pipekit.control import Branch, Coalesce, Retry, Switch, Try
from pipekit.observe import Histogram, Profile, ShapeTrace, Snapshot, Tap
from pipekit.parallel import (
    AsyncMap,
    BatchedMap,
    ProcessMap,
    ThreadMap,
    check_pickleable,
)
from pipekit.protocols import FittableTransformer, Predictor
from pipekit.qc import (
    AssertCallable,
    AssertDType,
    AssertHasAttribute,
    AssertShape,
    QCError,
    Quarantine,
)
from pipekit.serial import allowed, dumps, loads, loads_sandboxed, register, unregister
from pipekit.signature import Signature, compute_output_signature
from pipekit.state import CarryState, StatefulOperator


__all__ = [
    "AssertCallable",
    "AssertDType",
    "AssertHasAttribute",
    "AssertShape",
    "AsyncMap",
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
    "check_pickleable",
    "complement",
    "compose",
    "compose_left",
    "compute_output_signature",
    "dumps",
    "juxt",
    "loads",
    "loads_sandboxed",
    "pipe",
    "register",
    "unregister",
]

__version__ = "0.0.1"  # x-release-please-version
