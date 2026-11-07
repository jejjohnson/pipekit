---
status: draft
version: 0.1.0
---

# pipekit-array Design Doc

**A thin layer of duck-typed array operators on top of `pipekit`,
implemented against the [Python Array API standard][array-api] so a
single operator runs on numpy, JAX, CuPy, PyTorch, or dask without
rewrite.**

[array-api]: https://data-apis.org/array-api/

## Structure

```
docs/design/
├── README.md             # This file — index and reading order.
├── vision.md             # Motivation, scope, the Array API bet.
├── architecture.md       # The dispatch pattern + how operators
│                         # delegate to array_namespace(x).
├── decisions.md          # ADRs A1–A21 (dispatch, scope vs core,
│                         # backend extras, named carriers, the
│                         # missing dim-level functions).
├── boundaries.md         # Non-goals, open questions, v0.1 DoD.
└── api/
    ├── README.md         # Operator catalog — at-a-glance table.
    ├── operators.md      # Layer 1 — data-flow ops (ApplyAlong,
    │                     # StackAlong, ConcatenateAlong, Subsample,
    │                     # MeanScalar, BatchAlong, Histogram).
    ├── qc.md             # Layer 2 — numeric QC (AssertClose,
    │                     # AssertValueRange, AssertNoNaN,
    │                     # AssertValidFraction).
    ├── inference.md      # Layer 3 — ModelOp (framework-agnostic
    │                     # inference; numpy / JAX / torch / Keras).
    └── dim_functions.md  # The missing dim-level functions (rename,
                          # merge_dims, shift, window, quantile, dot, …)
                          # and their xarray parity table.
```

## Reading Order

1. **[vision.md](vision.md)** — why a dedicated array layer at all,
   and what the Array API gives us for free.
2. **[architecture.md](architecture.md)** — the `array_namespace`
   dispatch pattern, the four ADR-level shape rules every operator
   follows, and how `pipekit-array` slots between `pipekit` and the
   domain libraries (`geotoolz`, `xr_toolz`).
3. **[decisions.md](decisions.md)** — the ADRs that lock the shape in.
4. **[api/README.md](api/README.md)** → drill into
   **[operators](api/operators.md)** → **[qc](api/qc.md)** →
   **[inference](api/inference.md)** →
   **[dim_functions](api/dim_functions.md)**.
5. **[boundaries.md](boundaries.md)** — what's deferred and why.

## Sources

The design synthesises two upstream sources:

- The master plan, [`research_journal_v2/notes/geotoolz/master_plan/toolz_3_pipekit_array.md`][report-3]
  (Report 3, "Sister libraries on top of pipekit"). Establishes the
  three-library carrier story (`pipekit-array` for arrays, `geotoolz`
  for `GeoTensor`, `xr_toolz` for xarray) and the operator catalogue
  (~12 operators, ~400 LOC) that pipekit-array v0.1 ships.
- The existing `pipekit-ops` core surface — particularly
  [`pipekit_ops.qc`][pipekit-qc] (which explicitly defers numeric checks
  to `pipekit-array`, see its module docstring),
  [`pipekit_ops.observe.Histogram`][pipekit-observe] (the controller
  pattern this design extends with a carrier-aware `to_array=...`
  default), and [`pipekit_ops.parallel.BatchedMap`][pipekit-parallel]
  (the iterable-shaped variant; pipekit-array's array-shaped operator
  is named `BatchAlong` so it does *not* shadow it — see ADR A19).

[report-3]: https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_3_pipekit_array.md
[pipekit-qc]: ../../../pipekit/src/pipekit/qc.py
[pipekit-observe]: ../../../pipekit/src/pipekit/observe.py
[pipekit-parallel]: ../../../pipekit/src/pipekit/parallel.py

Where the two sources diverge, this design resolves the tension
explicitly — see [decisions.md](decisions.md).
