<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-array/logo-dark.svg">
    <img alt="pipekit-array" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-array/logo-light.svg" width="386">
  </picture>
</p>

# pipekit-array

> **Status: v0.1.1, partly built.**
> The dim-level functions, the named carriers and the data-flow operators ship today; the numeric QC family, the array `Histogram` and `ModelOp` are designed but not built yet.
> The [design doc][design] has the full plan.

Duck-typed array operators on top of `pipekit-ops`, written against the [Python Array API standard](https://data-apis.org/array-api/).
One operator runs on every backend (NumPy, JAX, CuPy, PyTorch, dask) because it asks the input for its namespace with `array_namespace(x)`.
On named carriers (a `DimArray`, an `xr.DataArray`, a geoarray carrier) the same functions work by dim name instead of axis.

## What ships

| Module | Public names |
| --- | --- |
| `_namespace` | `array_namespace` |
| `reductions` | `MeanScalar` |
| `combinators` | `StackAlong`, `ConcatenateAlong` (`axis=` for raw arrays, `dim=` for named carriers) |
| `dimarray` | `DimArray`, `DimArrayAdapter` |
| `dims` | `isel`, `sel`, `transpose`, `reduce`, `argreduce`, `take_along`, `map_over`, `stack`, `stack_items`, `concat`, `combine`, `BackendError` |
| `dim_shapes` | `rename`, `expand_dims`, `squeeze`, `broadcast_like`, `merge_dims`, `split_dims` |
| `dim_shifts` | `pad`, `roll`, `shift`, `diff`, `cumreduce`, `where` |
| `dim_windows` | `window`, `coarsen` |
| `dim_algebra` | `quantile`, `dot` |
| `dim_operators` | the operator twin of each dim function (`ISel`, `Reduce`, `Rename`, `Diff`, `Window`, `Quantile`, `Dot`, …) and `ApplyAlong` |
| `batching` | `batch`, `Padded`, `BatchAlong` |
| `rows` | `to_rows`, `from_rows`, `RowLayout` |
| `estimator` | `Estimator` (`[sklearn]`) |
| `kernel` | `KernelSpec`, `KernelOp` |
| `einx` | `rearrange`, `reduce`, `einsum`, `repeat`, `Einx` (`[einx]`; import as `from pipekit_array import einx`) |
| `xarray` | the `xr.DataArray` adapter (`[xarray]`), loaded the first time a DataArray reaches `pipekit_ops.get_adapter` |
| `testing` | `check_adapter`, `check_kernel_op` (the test kit for carrier libraries) |

Not built yet, from the v0.1 catalogue in [`docs/design/api/`][design-api]:

- **QC:** `AssertClose`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction`;
- **Observation:** the array-aware `Histogram` controller;
- **Inference:** `ModelOp`, the framework-agnostic trained-model wrapper (see [ADR A8][adr-a8]).

Two planned operators were retired instead of built: `Subsample` is `isel` with a stepped slice, and `ApplyToBands` became `ApplyAlong` ([ADR A15][adr-a11]).
The planned array-shaped `BatchedMap` became `BatchAlong`, so it no longer shares a name with `pipekit_ops.BatchedMap` ([ADR A19][adr-a11], which supersedes A5).

## A first example

```python
import numpy as np
from pipekit_array import DimArray, reduce, window

# Shapes: T = 12 scenes (time); Y = X = 4 pixels (y, x); S = 3 (window)

data = np.arange(12 * 4 * 4, dtype=np.float32).reshape(12, 4, 4)  # → (T, Y, X)
# today (pipekit-array)
cube = DimArray(data, ("time", "y", "x"))       # (T, Y, X) → (T, Y, X)

# today (pipekit-array)
# w_{nijr} = x_{n+r,ij}, 0 ≤ r < S
windows = window(cube, time=3)                  # (T, Y, X) → (T−2, Y, X, S)

# today (pipekit-array)
# m_{nij} = (1/S) ∑ᵣ w_{nijr}
smooth = reduce(windows, "mean", "window")      # (T−2, Y, X, S) → (T−2, Y, X)
```

## Design doc

- [`docs/design/`][design]: the index, vision, architecture, ADRs, boundaries and per-operator specs.
- [Report 3, "Sister libraries"][report-3] is the upstream master-plan source.

[design]: docs/design/README.md
[design-api]: docs/design/api/README.md
[adr-a8]: docs/design/decisions.md#a8-modelop-split-pipekit-array-vs-pipekit-jax
[adr-a11]: docs/design/decisions.md#a11-to-a22-the-missing-dim-level-functions
[report-3]: https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_3_pipekit_array.md
