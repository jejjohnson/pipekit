# pipekit-array

Array-API operators on top of `pipekit`. One operator runs on numpy,
JAX, CuPy, PyTorch, or dask via `array_namespace(x)` dispatch — see
the [design doc](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/README.md)
for the full architecture, ADRs, and the v0.1 operator catalogue.

Phase A of the v0.1 implementation has landed: namespace dispatch +
reductions + stack/concat combinators. Phases B–D add geometry, QC,
and the trained-model wrapper `ModelOp`.

## `array_namespace`

::: pipekit_array._namespace

## Reductions

::: pipekit_array.reduce

## Combinators

::: pipekit_array.combinators
