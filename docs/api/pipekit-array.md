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

::: pipekit_array.reductions

## Combinators

::: pipekit_array.combinators

## Named arrays

::: pipekit_array.dimarray

## Dim-level functions

::: pipekit_array.dims

::: pipekit_array.dim_operators

## Sample tables

`to_rows(x, sample_dims)` turns any named carrier into a `(n_samples, n_features)` table, rows over the sample dims and columns over the rest, and `from_rows(rows, layout)` turns a table back into a carrier like the input: with the original features, a new feature dim (`feature_dim="component"`) or, for one value per sample, the sample dims only. Both run on the data's own backend (geostack OP-O3).

::: pipekit_array.rows

## einx

`pipekit_array.einx` writes tensor algebra as einx patterns whose axis names are the carrier's dim names: `rearrange`, `reduce`, `einsum`, `repeat` and the `Einx` operator, on NumPy or JAX, rewrapped through the first input's adapter so surviving dims keep their labels, coordinates and grid. A dask input raises `BackendError` naming `.compute()`: einx has no dask backend. geostack OP-O2.

::: pipekit_array.einx

## Batches

`batch` groups carriers into fixed-size `Padded` batches (`value`,
`valid`) along a new dim, padding a short last batch with copies of its
last item (`pad="last"`), yielding it short (`"none"`) or dropping it
(`"drop"`). `stack_items` is the stacking it uses; leave the padding out
of a statistic with `reduce(..., where=padded.valid)`. ADR A10.

::: pipekit_array.batching

## Kernels

::: pipekit_array.kernel

## xarray

::: pipekit_array.xarray

## Test kit

Shipped in the wheel, for carrier libraries' own test suites (gates PK-K1 and
PK-K4; pipekit's own tests run PK-K2, PK-K3 and, under `-m slow`, PK-K5).

::: pipekit_array.testing
