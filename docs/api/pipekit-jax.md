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

## Adjoint interpreters

JAX realisations of the `pipekit_cycle.adjoints` specs: `truncated_scan`
implements `TruncatedAdjoint` at the cycle layer (a `jax.lax.scan` whose
carry is `stop_gradient`-ed outside the trailing-`k` window), and
`to_diffrax_adjoint` maps specs onto `diffrax.AbstractAdjoint` instances
for the dynamics layer.

::: pipekit_jax.adjoints

## DiffraxForwardModel

A `pipekit_cycle.ForwardModel` backed by `diffrax.diffeqsolve` — ODE or
SDE — that also satisfies `SupportsAdjoint`, so the gradient strategy of
the forecast step is configuration. Requires the `pipekit-jax[diffrax]`
extra.

::: pipekit_jax.diffrax_model
