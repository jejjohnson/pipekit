"""One array function on every carrier: `KernelSpec`, `KernelOp`, `elementwise`.

A shapes-only kernel declares the axes it consumes and produces with a
generalized-ufunc signature (``"(c)->()"``, ``"(r,c),(c),(c,d)->(r)"``),
the convention NumPy (``np.vectorize``), JAX (``jnp.vectorize``) and
xarray (``apply_ufunc`` core dims) already share. `KernelOp` wraps the
kernel once and runs it:

    raw arrays        → the kernel, called as is
    named carriers    → core dims moved last by name, the kernel on the
                        data, the result rewrapped by the adapter
                        (NumPy eagerly, JAX traced under jit)
    an adapter with   → that adapter's own path (the xarray adapter:
    ``apply_kernel``    one ``apply_ufunc``, lazy on dask)

Kernel conventions:

1. **Consumed axes last.** Core dims arrive as the trailing axes.
2. **Broadcast over leading axes,** or set ``vectorize=True`` and pay for
   ``np.vectorize`` / ``jnp.vectorize``.
3. **Non-carrier inputs are keyword parameters** (a background mean, a
   threshold), passed whole and never mapped. Geometry derived from the
   carrier (``geometry=``) is passed the same way; geometry that varies
   per pixel has the carrier's dims and is a positional input instead.
4. **Write against the array namespace** (``xp.linalg.solve``) to run on
   every backend. A kernel that needs a JAX-only library (lineax)
   declares ``backends=("jax",)`` (spike finding F3).

Rules from the spike (geotoolz#516):

- **Outputs keep the first input's dim order** (F2): a gufunc puts core
  dims last, so each output is transposed back, with dims new to the
  output last.
- **Every output has the loop dims of all the inputs.** An output that
  ignores an input is broadcast (#246).
- **A DataArray's name and attributes are dropped,** because the kernel
  changes the quantity and its unit. Coordinates keep theirs, and other
  carriers keep what their adapter's ``rewrap`` keeps.
- **``geometry=`` needs one geometry per call** (F4): on a carrier whose
  adapter reports ``batch_dims`` (per-item transforms), `KernelOp` raises
  and names ``map_over``.
- **An output-only core dim declares its size** (F5): ``KernelSpec.sizes``
  maps it to an input core dim, so a lazy backend can plan the output.

Carriers of different kinds combine by dim name and size (a geo carrier
``"band row col"`` with a `DimArray` ``"col band"``); grid checks
(``check_compatible``) run between carriers of the same kind.

`elementwise` is the decorator for a function of plain arrays that maps each
element: it aligns labelled arrays by name and calls the function on their data.
It needs ``pipekit-array[numpy,compat]`` and is imported on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pipekit_array.kernels._op import KernelOp
from pipekit_array.kernels._run import (
    derive_geometry,
    run_kernel,
)
from pipekit_array.kernels._spec import (
    KernelSpec,
    kernel_output_dims,
    kernel_output_signature,
)


if TYPE_CHECKING:
    from pipekit_array.kernels._elementwise import elementwise


__all__ = [
    "KernelOp",
    "KernelSpec",
    "derive_geometry",
    "elementwise",
    "kernel_output_dims",
    "kernel_output_signature",
    "run_kernel",
]


def __getattr__(name: str) -> Any:
    # it needs pipekit-array[numpy,compat], which `KernelOp` does not
    if name == "elementwise":
        from pipekit_array.kernels._elementwise import elementwise

        return elementwise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
