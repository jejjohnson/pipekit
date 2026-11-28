# pipekit-array

Array-API operators on top of `pipekit-ops`.
One operator runs on NumPy, JAX, CuPy, PyTorch or dask because it dispatches on `array_namespace(x)`.
The [design pages](https://jejjohnson.github.io/pipekit/readme-2/) have the architecture, the ADRs and the operator catalogue the package was planned from.

What ships today: namespace dispatch, `MeanScalar`, `StackAlong` and `ConcatenateAlong`, the named carrier `DimArray`, all the dim-level functions with their operator twins (`ApplyAlong` included), padded batches and `BatchAlong`, sample tables and `Estimator`, the einx bridge, `KernelOp`, the xarray adapter and the test kit.
What is still to build from the v0.1 catalogue: the numeric QC family (`AssertClose`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction`), the array-aware `Histogram` and the trained-model wrapper `ModelOp`.

The package is six areas, each with one job.
Import a name from its area, never from a file in it.
An area imports only from the ones before it, and `testing` from all of them ([ADR A30](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/decisions.md)).

## Plain arrays

`pipekit_array.arrays` is for functions of plain arrays, such as the formulas of geophysix and radiomax (`[numpy,compat]`).
Python numbers, sequences and NumPy arrays join the library and the device of the other arrays (ADR A23).
What the array API standard leaves out (indexed updates, loops, padding) is JAX's own call on JAX data, and a plain equivalent on every other library (ADR A27).

::: pipekit_array.arrays

## Named carriers

`pipekit_array.named` is for carriers with dimension names: a `DimArray`, an `xr.DataArray`, or any carrier with a registered adapter.
It joins NumPy to JAX, and refuses other mixes (ADR A24).
`to_rows` and `from_rows` turn a carrier into a `(n_samples, n_features)` table and back (geostack OP-O3), and `batch` groups carriers into fixed-size `Padded` batches (ADR A10).

::: pipekit_array.named

## Kernels

`pipekit_array.kernels` runs one array function on raw arrays and on every carrier.

::: pipekit_array.kernels

::: pipekit_array.kernels._elementwise

## Operators

`pipekit_array.operators` has the operator twin of each named function, and `MeanScalar`, `StackAlong` and `ConcatenateAlong` for raw arrays.

::: pipekit_array.operators

## Bridges

`pipekit_array.bridges` is for other libraries on named carriers.

### xarray

The adapter of `xr.DataArray` loads the first time one arrives (`[xarray]`).

::: pipekit_array.bridges.xarray

### einx

einx patterns whose axis names are the carrier's dim names (`[einx]`, geostack OP-O2).
A dask input raises `BackendError` naming `.compute()`, because einx has no dask backend.

::: pipekit_array.bridges.einx

### scikit-learn

`Estimator` fits and applies a scikit-learn-style estimator on any named carrier, through `to_rows` and `from_rows` (`[sklearn]`, imported lazily; geostack OP-O3).

::: pipekit_array.bridges.sklearn

## Test kit

The test kit ships in the wheel, for other libraries' own test suites.
`check_adapter` and `check_kernel_op` are for carrier libraries (gates PK-K1 and PK-K4); pipekit's own tests run PK-K2, PK-K3 and, under `-m slow`, PK-K5.
The others are for functions on any array library (ADR A29).

::: pipekit_array.testing
