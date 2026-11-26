# pipekit-ops

The pipekit core, imported as `pipekit_ops`: carrier-agnostic operators and operator graphs.
Pure Python, no third-party dependencies.
Until 0.2.0 this package was `pipekit`.
Master plan reference: Report 2.

The package is seven areas, each with one job.
Import a name from the package or from its area, never from a file in it.
An area imports only from the ones before it, and `testing` from all of them ([O17](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-ops/docs/design/decisions.md)).

## Carriers

`pipekit_ops.carriers` is how code reaches a value: the named-array adapters and their registry, the array backend a value sits on, its `Signature`, the NumPy–JAX `guard`, tracer detection and optional imports.

::: pipekit_ops.carriers

## Core

`pipekit_ops.core` has `Operator`, `Sequential`, `Graph`, the state primitives, the structural protocols and the composition helpers.
It also has the helpers for libraries that write operators: `nested_config`, `require_operator`, `resolve_loaded`, `record_under` and `record_moves`.

::: pipekit_ops.core

## Saving

`pipekit_ops.saving` writes and reads a pipeline's state, hashes it, and holds the stores that `Cache` keeps its entries in.

::: pipekit_ops.saving

## Operators

`pipekit_ops.operators` has the ready-made operators: blocks, dict and iterable plumbing, `Fanout`, control flow, flow limits, caching and quality control.

::: pipekit_ops.operators

## Execution

`pipekit_ops.execution` runs an operator over many inputs: `Map`, its presets, `Prefetch` and `Fold`.

::: pipekit_ops.execution

## Observe

`pipekit_ops.observe` looks inside a running pipeline: the observers and `jit_report`.

::: pipekit_ops.observe

## Testing

::: pipekit_ops.testing
