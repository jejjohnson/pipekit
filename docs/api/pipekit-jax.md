# pipekit-jax

JAX / Equinox carrier integration for `pipekit`. Ships `JaxModelOp`
(the public counterpart of the in-package
`pipekit_train.adapters.equinox.EquinoxModelOp`, adding byte-identical
weight round-trip through `pipekit_experiment.ModelRegistry` via the
registry's `weights` blob), `DiffraxForwardModel` (a
`pipekit_cycle.ForwardModel` behind the `[diffrax]` extra), and the
adjoint interpreters `to_diffrax_adjoint` / `truncated_scan`.

## Pytree containers

Importing `pipekit_jax` registers `pipekit.Sequential`,
`pipekit.Graph` and `JaxModelOp` as JAX pytrees, so `jax.grad` / `eqx.filter_grad` /
`jax.vmap` see through a pipeline to the leaves of the operators it
holds.

::: pipekit_jax.pytree

## JaxOperator

Base class for operators that are both a `pipekit.Operator` and an
`eqx.Module`: JSON-safe config, pytree leaves, and the weight-blob
round trip. Use `check_jax_operator` in a library's test suite to
assert the contract (eager ≡ jit, batching, visible leaves, a config
round trip that keeps learnable fields learnable).

::: pipekit_jax.operator

::: pipekit_jax.testing

::: pipekit_jax.weights

## ScanCycle and Rollout

`jax.lax.scan`-based time stepping. `ScanCycle` is the jit-native
counterpart of `pipekit_cycle.Cycle` — same `(carrier, state)` contract
and history convention, one scan regardless of `n_steps`, history
returned by `ScanCycle.run`. `Rollout` is the carrier-only form,
returning a trajectory sampled every `save_every` steps.
`checkpoint=True` rematerialises in chunks so `grad` stores O(√n)
carries instead of O(n).

Measured on a 2-vCPU CPU VM (5-point diffusion step on a 128² grid):

| steps | compile `Cycle` (unrolled) | compile `ScanCycle` | grad compile `Cycle` | grad compile `ScanCycle` |
|---:|---:|---:|---:|---:|
| 10 | 0.19 s | 0.12 s | 2.9 s | 0.26 s |
| 50 | — | — | 356 s | 0.28 s |
| 100 | 1.04 s | 0.14 s | — | — |
| 1000 | 19.0 s | 0.13 s | did not finish in 10 min | — |

| steps | grad temp memory, `checkpoint=False` | `checkpoint=True` |
|---:|---:|---:|
| 100 | 6.6 MiB | 1.7 MiB |
| 1000 | 62.9 MiB | 4.4 MiB |
| 4000 | 250.4 MiB | 8.3 MiB |

::: pipekit_jax.cycle

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
