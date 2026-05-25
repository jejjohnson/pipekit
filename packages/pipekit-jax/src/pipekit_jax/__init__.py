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

from pipekit_jax.model_op import JaxModelOp


__all__ = ["JaxModelOp"]

__version__ = "0.0.1"  # x-release-please-version
