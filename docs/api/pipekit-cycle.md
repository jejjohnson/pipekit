# pipekit-cycle

Time-stepping, data assimilation, observation operators.
Built on `pipekit_ops.core` (the `StatefulOperator` and `CarryState` primitives of pipekit core).
Master plan reference: Report 10.

The package is four areas, each with one job, and an area imports only from the ones before it ([C1](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-cycle/docs/design/decisions.md)).

## Protocols

The seams that a library satisfies without importing pipekit-cycle.

::: pipekit_cycle.protocols

## Cycles

`pipekit_cycle.cycles` steps a model through time, with the states the cycles carry and the adjoint specs.

The adjoint specs are a shared, declarative vocabulary for how gradients flow through the
three nested rollouts of a differentiable DA system — the dynamics
integration, the inner variational solve, and the assimilation cycle.
Each execution layer interprets a spec into its native mechanism
(diffrax adjoints via `integrix.adjoint.to_diffrax_adjoint`, optimistix
adjoints in vardax, scan transforms in filterax):

| Spec | dynamics (diffrax) | inner solve (optimistix) | cycle (scan) |
|---|---|---|---|
| `DirectAdjoint` | store-everything | unrolled | plain scan — exact, O(T) memory |
| `RecursiveCheckpointAdjoint` | treeverse ✅ default | checkpointed | checkpointed scan — exact, O(√T) |
| `TruncatedAdjoint(k)` | — | k-step / one-step | truncated scan — biased, O(1) in T, chaos-tolerant |
| `ImplicitAdjoint` | steady-state only | IFT at the fixed point | — |
| `BacksolveAdjoint` | continuous adjoint ⚠ unstable for chaotic/stiff dynamics | — | — |

::: pipekit_cycle.cycles

## Components

`pipekit_cycle.components` has the observation operators and the forward-model adapters.

::: pipekit_cycle.components

## Data assimilation

::: pipekit_cycle.assimilation
