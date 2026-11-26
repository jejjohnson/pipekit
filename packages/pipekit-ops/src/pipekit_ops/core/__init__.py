"""The operator and the ways to put operators together.

`Operator` and `ConfigMixin`; `Sequential` and `Graph` (with `Input` and `Node`);
`StatefulOperator` and `CarryState` for operators that carry state from call to call;
the structural seams `Predictor` and `FittableTransformer`; and `pipe`, `compose`,
`compose_left`, `juxt` and `complement`.

For libraries that write operators: `nested_config` is the config of an operator held by
another, `require_operator` checks that a value is one, `resolve_loaded` finds a loaded
operator class by name, `record_under` records classes under the module a library
imports them from, and `record_moves` lets states saved under a former module load.
"""

from __future__ import annotations

from pipekit_ops.core._compose import (
    complement,
    compose,
    compose_left,
    juxt,
    pipe,
)
from pipekit_ops.core._graph import (
    Graph,
    Input,
    Node,
)
from pipekit_ops.core._operator import (
    Carrier,
    ConfigMixin,
    Operator,
    nested_config,
    record_moves,
    record_under,
    require_operator,
    resolve_loaded,
)
from pipekit_ops.core._protocols import (
    FittableTransformer,
    Predictor,
)
from pipekit_ops.core._sequential import (
    Sequential,
)
from pipekit_ops.core._state import (
    CarryState,
    StatefulOperator,
)


__all__ = [
    "Carrier",
    "CarryState",
    "ConfigMixin",
    "FittableTransformer",
    "Graph",
    "Input",
    "Node",
    "Operator",
    "Predictor",
    "Sequential",
    "StatefulOperator",
    "complement",
    "compose",
    "compose_left",
    "juxt",
    "nested_config",
    "pipe",
    "record_moves",
    "record_under",
    "require_operator",
    "resolve_loaded",
]
