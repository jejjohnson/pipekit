"""Functions of plain arrays: any array library, no dimension names.

A function that is no Operator (a formula of a physics library) is written
once, in functions of the array API standard, and runs on NumPy, JAX, torch,
CuPy, dask and any other library that `array_api_compat` knows:

    @elementwise()                                       # labelled arrays
    def density(pressure, temperature):
        xp, (p, t) = floating_inputs(pressure, temperature)  # one dtype, device
        return p / (like(287.0, t, xp) * t)

- **The call**: `namespace_of`, `floating_inputs`, `as_inexact`, `concrete`,
  `is_dataarray`.
- **Values of the host**: `like`, `constants_like`, `table_like`, `boolean_like`.
- **Numerics**: `divide_or_nan`, `safe_sqrt`, `safe_power`, `safe_atan2`,
  `squared_magnitude`, `stand_in`.
- **Copies with changes**: `set_at`, `add_at`, `multiply_at`, `pad`.
- **Transformations**: `vmap`, `scan`, `stop_gradient`, `optimization_barrier`.
- **Checks**: `check_values`, `require_floating`.
- **Masked, weighted reductions**: `weighted_sum`, `weighted_mean`, `Groups`,
  `reduction_inputs`, `used_weights`.
- **Records**: `register_record`.

`pipekit_array.kernels.elementwise` lets such a function take labelled arrays.
This area needs ``pipekit-array[numpy,compat]``, so importing ``pipekit_array``
does not import it. It imports no other area, and no array library but NumPy.
"""

from __future__ import annotations

from pipekit_ops import BackendError


try:
    import array_api_compat  # noqa: F401
    import numpy  # noqa: F401
except ImportError as err:
    msg = (
        "pipekit_array.arrays needs NumPy and array-api-compat: "
        "install 'pipekit-array[numpy,compat]'"
    )
    raise ImportError(msg) from err

from pipekit_array.arrays._backend import (
    as_inexact,
    concrete,
    floating_inputs,
    is_dataarray,
    namespace_of,
)
from pipekit_array.arrays._checks import (
    check_values,
    require_floating,
)
from pipekit_array.arrays._host import (
    boolean_like,
    constants_like,
    like,
    table_like,
)
from pipekit_array.arrays._indexing import (
    add_at,
    multiply_at,
    pad,
    set_at,
)
from pipekit_array.arrays._numerics import (
    divide_or_nan,
    safe_atan2,
    safe_power,
    safe_sqrt,
    squared_magnitude,
    stand_in,
)
from pipekit_array.arrays._records import register_record
from pipekit_array.arrays._reductions import (
    NO_GROUP,
    Groups,
    reduction_inputs,
    used_weights,
    weighted_mean,
    weighted_sum,
)
from pipekit_array.arrays._transforms import (
    optimization_barrier,
    scan,
    stop_gradient,
    vmap,
)


__all__ = [
    "NO_GROUP",
    "BackendError",
    "Groups",
    "add_at",
    "as_inexact",
    "boolean_like",
    "check_values",
    "concrete",
    "constants_like",
    "divide_or_nan",
    "floating_inputs",
    "is_dataarray",
    "like",
    "multiply_at",
    "namespace_of",
    "optimization_barrier",
    "pad",
    "reduction_inputs",
    "register_record",
    "require_floating",
    "safe_atan2",
    "safe_power",
    "safe_sqrt",
    "scan",
    "set_at",
    "squared_magnitude",
    "stand_in",
    "stop_gradient",
    "table_like",
    "used_weights",
    "vmap",
    "weighted_mean",
    "weighted_sum",
]
