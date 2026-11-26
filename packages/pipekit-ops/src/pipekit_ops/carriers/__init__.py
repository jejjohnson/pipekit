"""How code reaches a value: its carrier, its array backend and its shape.

`NamedArrayAdapter` and its registry say how to work with a carrier that has dimension
names; `backend_of` and `to_backend` say which array library a value sits on and move
it; `Signature` describes a value's shape; `guard` reports data moving between NumPy and
JAX; `is_traced` says whether a value holds a JAX tracer; `on_import` and
`import_optional` run or load the code of an optional library. Nothing here knows what
an operator is.
"""

from __future__ import annotations

from pipekit_ops.carriers._adapters import (
    BackendError,
    ConvertingAdapter,
    LabelledAdapter,
    MissingAdapterError,
    MissingValueAdapter,
    NamedArrayAdapter,
    backend_name,
    backend_of,
    get_adapter,
    register_adapter,
    register_lazy_adapter,
    signature_of,
    to_backend,
    unregister_adapter,
)
from pipekit_ops.carriers._guards import (
    BackendWarning,
    guard,
    guard_level,
    report_move,
)
from pipekit_ops.carriers._imports import (
    import_optional,
    on_import,
)
from pipekit_ops.carriers._signature import (
    Signature,
    compute_output_signature,
)
from pipekit_ops.carriers._tracing import (
    is_concrete,
    is_traced,
)


__all__ = [
    "BackendError",
    "BackendWarning",
    "ConvertingAdapter",
    "LabelledAdapter",
    "MissingAdapterError",
    "MissingValueAdapter",
    "NamedArrayAdapter",
    "Signature",
    "backend_name",
    "backend_of",
    "compute_output_signature",
    "get_adapter",
    "guard",
    "guard_level",
    "import_optional",
    "is_concrete",
    "is_traced",
    "on_import",
    "register_adapter",
    "register_lazy_adapter",
    "report_move",
    "signature_of",
    "to_backend",
    "unregister_adapter",
]
