---
status: draft
version: 0.1.0
---

# pipekit-array Design Doc

**A thin layer of duck-typed array operators on top of `pipekit-ops`, written against the [Python Array API standard][array-api], so a single operator runs on NumPy, JAX, CuPy, PyTorch or dask without a rewrite.**

[array-api]: https://data-apis.org/array-api/

## Status

pipekit-array 0.1.1 ships part of this design.

| Part | Status |
| --- | --- |
| Namespace dispatch (`array_namespace`), `MeanScalar`, `StackAlong`, `ConcatenateAlong` | today (phase A) |
| Named carriers (`DimArray`) and every dim-level function of [api/dim_functions.md](api/dim_functions.md), with operator twins | today (#204 to #207) |
| `ApplyAlong` (replaces `ApplyToBands`) and `BatchAlong` (replaces the array-shaped `BatchedMap`) | today (#206) |
| `batch` / `Padded`, `to_rows` / `from_rows`, `Estimator`, the einx bridge, `KernelOp`, the xarray adapter, the test kit | today |
| `Subsample` | retired: it is `isel` with a stepped slice (A15) |
| `Histogram` (array-aware), `AssertClose`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction`, `ModelOp` | not built |

## Structure

```
docs/design/
├── README.md             # This file: index and reading order.
├── vision.md             # Motivation, scope, the Array API bet.
├── architecture.md       # The dispatch pattern and how operators
│                         # delegate to array_namespace(x).
├── decisions.md          # ADRs A1 to A22 (dispatch, scope vs core,
│                         # backend extras, named carriers, the
│                         # dim-level functions).
├── boundaries.md         # Non-goals, open questions, v0.1 DoD.
└── api/
    ├── README.md         # Operator catalogue: at-a-glance table.
    ├── operators.md      # Layer 1: data-flow ops (ApplyAlong,
    │                     # StackAlong, ConcatenateAlong, Subsample,
    │                     # MeanScalar, BatchAlong, Histogram).
    ├── qc.md             # Layer 2: numeric QC (AssertClose,
    │                     # AssertValueRange, AssertNoNaN,
    │                     # AssertValidFraction).
    ├── inference.md      # Layer 3: ModelOp (framework-agnostic
    │                     # inference; numpy / JAX / torch / Keras).
    └── dim_functions.md  # The dim-level functions (rename,
                          # merge_dims, shift, window, quantile, dot, …)
                          # and their xarray parity table.
```

## Reading order

1. **[vision.md](vision.md)**: why a dedicated array layer at all, and what the Array API gives us for free.
2. **[architecture.md](architecture.md)**: the `array_namespace` dispatch pattern, the four shape rules every operator follows, and how `pipekit-array` slots between `pipekit-ops` and the domain libraries (`geotoolz`, `xr_toolz`).
3. **[decisions.md](decisions.md)**: the ADRs that lock the shape in.
4. **[api/README.md](api/README.md)**, then **[operators](api/operators.md)**, **[qc](api/qc.md)**, **[inference](api/inference.md)** and **[dim_functions](api/dim_functions.md)**.
5. **[boundaries.md](boundaries.md)**: what is deferred and why.

## Sources

The design brings together two upstream sources:

- The master plan, [`research_journal_v2/notes/geotoolz/master_plan/toolz_3_pipekit_array.md`][report-3] (Report 3, "Sister libraries on top of pipekit").
  It sets the three-library carrier story (`pipekit-array` for arrays, `geotoolz` for `GeoTensor`, `xr_toolz` for xarray) and the operator catalogue (about 12 operators, about 400 LOC) that pipekit-array v0.1 ships.
- The `pipekit-ops` core surface, in particular:
  - [`pipekit_ops.qc`][pipekit-qc], whose module docstring defers numeric checks to `pipekit-array`;
  - [`pipekit_ops.observe.Histogram`][pipekit-observe], the controller pattern this design extends with a carrier-aware `to_array=...` default;
  - [`pipekit_ops.parallel.BatchedMap`][pipekit-parallel], the iterable-shaped variant; pipekit-array's array-shaped operator is named `BatchAlong` so it does *not* shadow it (ADR A19).

[report-3]: https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_3_pipekit_array.md
[pipekit-qc]: ../../../pipekit-ops/src/pipekit_ops/qc.py
[pipekit-observe]: ../../../pipekit-ops/src/pipekit_ops/observe.py
[pipekit-parallel]: ../../../pipekit-ops/src/pipekit_ops/parallel.py

Where the two sources diverge, this design resolves the tension explicitly in [decisions.md](decisions.md).
