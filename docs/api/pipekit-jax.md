# pipekit-jax

JAX / Equinox carrier integration for `pipekit`. Ships `JaxModelOp`
(the public counterpart of the in-package
`pipekit_train.adapters.equinox.EquinoxModelOp`, adding byte-identical
weight round-trip through `pipekit_experiment.ModelRegistry` via the
registry's `weights` blob), `DiffraxForwardModel` (a
`pipekit_cycle.ForwardModel` behind the `[diffrax]` extra), and the
adjoint interpreters `to_diffrax_adjoint` / `truncated_scan`.

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
