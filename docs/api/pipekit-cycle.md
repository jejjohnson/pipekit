# pipekit-cycle

Time-stepping, data assimilation, observation operators. Built on
`pipekit.state` (the `StatefulOperator` + `CarryState` primitives in
pipekit core). Master plan reference: Report 10.

## Cycle wrappers

::: pipekit_cycle.cycle

## Protocols

::: pipekit_cycle.protocols

## Adjoint strategy specs

A shared, declarative vocabulary for how gradients flow through the
three nested rollouts of a differentiable DA system — the dynamics
integration, the inner variational solve, and the assimilation cycle.
Each execution layer interprets a spec into its native mechanism
(diffrax adjoints via `pipekit_jax.to_diffrax_adjoint`, optimistix
adjoints in vardax, scan transforms in filterax):

| Spec | dynamics (diffrax) | inner solve (optimistix) | cycle (scan) |
|---|---|---|---|
| `DirectAdjoint` | store-everything | unrolled | plain scan — exact, O(T) memory |
| `RecursiveCheckpointAdjoint` | treeverse ✅ default | checkpointed | checkpointed scan — exact, O(√T) |
| `TruncatedAdjoint(k)` | — | k-step / one-step | truncated scan — biased, O(1) in T, chaos-tolerant |
| `ImplicitAdjoint` | steady-state only | IFT at the fixed point | — |
| `BacksolveAdjoint` | continuous adjoint ⚠ unstable for chaotic/stiff dynamics | — | — |

::: pipekit_cycle.adjoints

## Data-assimilation cycles

::: pipekit_cycle.da

## Observation operators

::: pipekit_cycle.obs

## Forward-model adapters

::: pipekit_cycle.forward

## Carry-state subclasses

::: pipekit_cycle.state
