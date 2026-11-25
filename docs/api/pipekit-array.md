# pipekit-array

Array-API operators on top of `pipekit-ops`.
One operator runs on NumPy, JAX, CuPy, PyTorch or dask because it dispatches on `array_namespace(x)`.
The [design doc](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/README.md) has the architecture, the ADRs and the v0.1 operator catalogue.

What ships today: namespace dispatch, `MeanScalar`, `StackAlong` and `ConcatenateAlong`, the named carrier `DimArray`, all the dim-level functions with their operator twins (`ApplyAlong` included), padded batches and `BatchAlong`, sample tables and `Estimator`, the einx bridge, `KernelOp`, the xarray adapter and the test kit.
What is still to build from the v0.1 catalogue: the numeric QC family (`AssertClose`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction`), the array-aware `Histogram` and the trained-model wrapper `ModelOp`.

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

::: pipekit_array.dim_shapes

::: pipekit_array.dim_shifts

::: pipekit_array.dim_windows

::: pipekit_array.dim_algebra

::: pipekit_array.dim_operators

## Sample tables

`to_rows(x, sample_dims)` turns any named carrier into a `(n_samples, n_features)` table, with rows over the sample dims and columns over the rest.
`from_rows(rows, layout)` turns a table back into a carrier like the input: with the original features, with a new feature dim (`feature_dim="component"`) or, for one value per sample, with the sample dims only.
Both run on the data's own backend (geostack OP-O3).

::: pipekit_array.rows

## scikit-learn estimators

`Estimator` fits and applies a scikit-learn-style estimator on any named carrier, through `to_rows` and `from_rows` (`[sklearn]`, imported lazily; geostack OP-O3).

::: pipekit_array.estimator

## einx

`pipekit_array.einx` writes tensor algebra as einx patterns whose axis names are the carrier's dim names: `rearrange`, `reduce`, `einsum`, `repeat` and the `Einx` operator.
It runs on NumPy or JAX and rewraps through the first input's adapter, so the dims that survive keep their labels, coordinates and grid.
A dask input raises `BackendError` naming `.compute()`, because einx has no dask backend (geostack OP-O2).

::: pipekit_array.einx

## Batches

`batch` groups carriers into fixed-size `Padded` batches (`value`, `valid`) along a new dim.
A short last batch is padded with copies of its last item (`pad="last"`), yielded short (`"none"`) or dropped (`"drop"`).
`stack_items` is the stacking it uses; leave the padding out of a statistic with `reduce(..., where=padded.valid)` (ADR A10).
`BatchAlong(inner, size, dim, pad)` is the operator form for one carrier: it cuts `dim` into chunks, runs `inner` on each and joins the results (ADR A19).

::: pipekit_array.batching

## Kernels

::: pipekit_array.kernel

## Plain functions

`pipekit_array.functions` is the array backend of functions that are no operators, such as the formulas of geophysix and radiomax (`[numpy,compat]`).
Python numbers, sequences and NumPy arrays join the library and the device of the other arrays (ADR A23); the named layer joins NumPy to JAX, and refuses other mixes (ADR A24).

::: pipekit_array.functions

### Hooks

`pipekit_array.hooks` holds what the array API standard leaves out: indexed updates (`set_at`, `add_at`, `multiply_at`), loops (`vmap`, `scan`), `pad`, `stop_gradient` and `optimization_barrier`.
Each is JAX's own call on JAX data, and a plain equivalent on every other library (ADR A27).

::: pipekit_array.hooks

### Checks

`pipekit_array.checks` raises at once on a concrete array, and when the compiled code runs on a traced one (through equinox, imported under a trace only).

::: pipekit_array.checks

### Masked and weighted reductions

`pipekit_array.weighted` is written on that backend: sums and means with a mask and weights, of an array or of static groups of samples (an average over months, rings or labelled regions).
The array API standard has no `nansum` and no `bincount`, and a mask says which samples count, never a NaN (ADR A25).

::: pipekit_array.weighted

### Records

`register_record` makes a frozen dataclass a JAX pytree once JAX loads, for libraries whose import never loads JAX.

::: pipekit_array.records

## xarray

::: pipekit_array.xarray

## Test kit

The test kit ships in the wheel, for carrier libraries' own test suites (gates PK-K1 and PK-K4).
pipekit's own tests run PK-K2, PK-K3 and, under `-m slow`, PK-K5.

::: pipekit_array.testing
