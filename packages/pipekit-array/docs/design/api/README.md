---
status: draft
version: 0.1.0
---

# API — Operator Catalog

The v0.1 surface is 12 operators across 7 modules. The grouping
mirrors `pipekit` core where it makes sense
(`combinators` ↔ `pipekit.combine`, `qc` ↔ `pipekit.qc`,
`observe` ↔ `pipekit.observe`, `parallel` ↔ `pipekit.parallel`).

## At a glance

| Module             | Operator                | One-liner                                                     |
| ------------------ | ----------------------- | ------------------------------------------------------------- |
| `combinators`      | `ApplyToBands`          | Split along an axis, apply `inner` per slice, stack back.     |
| `combinators`      | `StackAlong`            | `xp.stack` list-of-arrays along a new axis.                   |
| `combinators`      | `ConcatenateAlong`      | `xp.concatenate` along an existing axis.                      |
| `geom`             | `Subsample`             | Stride-decimate along given axes.                             |
| `observe`          | `Histogram`             | Controller; `.at(key)` captures per-tap distributions.        |
| `reductions`       | `MeanScalar`            | `xp.mean` reduction; optional `axis`.                         |
| `parallel`         | `BatchedMap`            | Split along axis 0 into chunks, apply `inner`, concatenate.   |
| `qc`               | `Diff`                  | Compare to a stored reference; raise on numeric drift.        |
| `qc`               | `AssertValueRange`      | Pass-through; raise/warn on out-of-range.                     |
| `qc`               | `AssertNoNaN`           | Pass-through; raise on any NaN.                               |
| `qc`               | `AssertValidFraction`   | Pass-through; raise if `< min_valid` non-NaN fraction.        |
| `inference`        | `ModelOp`               | Framework-agnostic inference wrapper.                          |

## Named carriers: dim-level functions

On any carrier with a registered `pipekit.NamedArrayAdapter` (a `DimArray`,
an `xr.DataArray`, a geoarray carrier), by dim name rather than axis.
Each is a function in `dims` and, where it serialises, an operator in
`dim_operators` (GA-Q3; geostack `docs/design/pipekit_kernels.md`).

| Function     | Operator    | Does                                                    |
| ------------ | ----------- | ------------------------------------------------------- |
| `isel`       | `ISel`      | position lookup; an int drops the dim                   |
| `sel`        | `Sel`       | exact label lookup; no ranges, no nearest               |
| `transpose`  | `Transpose` | reorder dims by name                                    |
| `reduce`     | `Reduce`    | reduce over named dims (the operator takes a name)      |
| `argreduce`  | `ArgReduce` | index of a reduction along a dim                        |
| `take_along` | `TakeAlong` | gather along a dim with an index carrier                |
| `map_over`   | `MapOver`   | a function per item along a dim (`forbid_in_yaml`)      |
| `stack`      | `StackAlong(dim=)` | a new leading dim                                |
| `concat`     | `ConcatenateAlong(dim=)` | join along an existing dim                 |
| `combine`    | `Combine`   | element-wise function of same-dim carriers (`forbid_in_yaml`) |

An adapter may override any of them with a method of the same name
(xarray's `reduce` stays lazy on dask; geoarray's `map_over` is one
`jax.vmap`); the test kit proves the override equals the generic path.

## Reading order

1. **[operators.md](operators.md)** — the data-flow operators (the
   combinators, the reducer, the array-shaped `BatchedMap`, the
   `Histogram` controller, `Subsample`). The "what does the
   pipeline do" surface.
2. **[qc.md](qc.md)** — the numeric QC family. The "what does the
   pipeline guard against" surface.
3. **[inference.md](inference.md)** — `ModelOp`. The trained-model
   wrapper that closes the train→serve loop for pipekit-train's
   Lightning and Keras adapters.

Each operator spec covers: constructor signature, `_apply` contract,
`get_config()` shape, `compute_output_signature` behaviour (if
non-trivial), per-backend caveats, and a one-or-two-line example.
