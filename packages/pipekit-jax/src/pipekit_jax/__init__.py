"""pipekit-jax — JAX / Equinox carrier integration for pipekit.

Public surface:

- `JaxOperator` — base class for operators that are both a
  ``pipekit_ops.Operator`` and an ``eqx.Module`` (JSON config, pytree
  leaves, weight round trip). Contract checks live in
  `pipekit_jax.testing.check_jax_operator`.
- `ScanCycle` / `EnsembleScanCycle` / `Rollout` — ``jax.lax.scan``-based
  time stepping: the jit-native counterpart of ``pipekit_cycle.Cycle``
  (history returned by ``ScanCycle.run``), its ``vmap``-ed ensemble form,
  and a carrier-only trajectory rollout.
- `JaxModelOp` — wraps an ``eqx.Module`` as a ``pipekit_ops.Operator``,
  with ``serialize_weights`` / ``with_weights`` / ``from_registry``
  methods so trained weights round-trip byte-identically through any
  ``pipekit_experiment.ModelRegistry`` via the registry's ``weights``
  blob.
- `to_jax` / `JaxKernelOp` — a ``pipekit_array.KernelOp`` as a
  `JaxOperator` whose array parameters are leaves (behind the
  ``[array]`` extra; imported lazily).

Importing this package also registers ``pipekit_ops.Sequential``,
``pipekit_ops.Graph`` and `JaxModelOp` as JAX pytrees (`pipekit_jax.pytree`),
so ``jax.grad`` / ``eqx.filter_grad`` reach the leaves of operators
inside a pipeline.

See master plan Report 11 (companion of pipekit-train) and
`pipekit-train/docs/design/boundaries.md §13.1` for the v0.1
caveat this package closes.
"""

from typing import Any

from pipekit_jax import pytree as pytree  # registers Sequential / Graph
from pipekit_jax.cycle import EnsembleScanCycle, Rollout, ScanCycle
from pipekit_jax.model_op import JaxModelOp
from pipekit_jax.operator import JaxOperator


def __getattr__(name: str) -> Any:
    # to_jax / JaxKernelOp need the [array] extra (pipekit-array)
    if name in ("to_jax", "JaxKernelOp"):
        from pipekit_jax import kernel

        return getattr(kernel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EnsembleScanCycle",
    "JaxKernelOp",
    "JaxModelOp",
    "JaxOperator",
    "Rollout",
    "ScanCycle",
    "to_jax",
]

__version__ = "0.3.0"  # x-release-please-version
