# pipekit-jax

JAX / Equinox carrier integration for `pipekit`. Ships `JaxModelOp`,
the public successor to the in-package
`pipekit_train.adapters.equinox.EquinoxModelOp` stand-in that v0.1
of `pipekit-train` shipped with. Adds byte-identical weight
round-trip through `pipekit_experiment.ModelRegistry` via the
registry's `weights` blob — closing the v0.1 reproducibility caveat
documented in `pipekit-train/docs/design/boundaries.md §13.1`.

## JaxModelOp

::: pipekit_jax.model_op
