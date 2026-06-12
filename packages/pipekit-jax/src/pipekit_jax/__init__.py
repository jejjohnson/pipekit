"""pipekit-jax — JAX / Equinox carrier integration for pipekit.

Ships one operator:

- `JaxModelOp` — wraps an ``eqx.Module`` as a ``pipekit.Operator``,
  with ``serialize_weights`` / ``with_weights`` methods so trained
  weights round-trip byte-identically through any
  ``pipekit_experiment.ModelRegistry`` via the registry's ``weights``
  blob.

See master plan Report 11 (companion of pipekit-train) and
`pipekit-train/docs/design/boundaries.md §13.1` for the v0.1
caveat this package closes.
"""

from typing import Any

from pipekit_jax.adjoints import to_diffrax_adjoint, truncated_scan
from pipekit_jax.model_op import JaxModelOp


def __getattr__(name: str) -> Any:
    # DiffraxForwardModel needs the [diffrax] extra; import lazily so the
    # base package keeps working without it.
    if name == "DiffraxForwardModel":
        from pipekit_jax.diffrax_model import DiffraxForwardModel

        return DiffraxForwardModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiffraxForwardModel",
    "JaxModelOp",
    "to_diffrax_adjoint",
    "truncated_scan",
]

__version__ = "0.0.1"  # x-release-please-version
