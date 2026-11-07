---
status: draft
version: 0.1.0
---

# API: operator catalogue

The v0.1 plan is 12 operators across 7 modules.
The grouping mirrors `pipekit-ops` core where it makes sense (`combinators` with `pipekit_ops.combine`, `qc` with `pipekit_ops.qc`, `observe` with `pipekit_ops.observe`, `batching` with `pipekit_ops.parallel`).
Six of the 12 are built or replaced today; the QC family, `Histogram` and `ModelOp` are not built yet.

## At a glance

| Module | Operator | One-liner | Status |
| --- | --- | --- | --- |
| `dim_operators` | `ApplyAlong` | `inner` per item along a dim, stacked back; serialises (A15). | today |
| `combinators` | `StackAlong` | `xp.stack` of a list along a new axis (`axis=`), or `stack` of named carriers (`dim=`). | today |
| `combinators` | `ConcatenateAlong` | `xp.concat` along an existing axis (`axis=`), or `concat` of named carriers (`dim=`). | today |
| `geom` | `Subsample` | Stride-decimate along given axes. | retired: `isel` with a stepped slice (A15) |
| `observe` | `Histogram` | Controller; `.at(key)` captures per-tap distributions. | not built |
| `reductions` | `MeanScalar` | `xp.mean` reduction; optional `axis`, `keepdims`. | today |
| `batching` | `BatchAlong` | Split along a dim into chunks, apply `inner`, concatenate (A19). | today |
| `qc` | `AssertClose` | Compare to a stored reference; raise on numeric drift (A21). | not built |
| `qc` | `AssertValueRange` | Pass-through; raise or warn on out-of-range values. | not built |
| `qc` | `AssertNoNaN` | Pass-through; raise on any NaN. | not built |
| `qc` | `AssertValidFraction` | Pass-through; raise if the non-NaN fraction is `< min_valid`. | not built |
| `inference` | `ModelOp` | Framework-agnostic inference wrapper. | not built |

Beyond the catalogue, the package ships `batch` and `Padded` (A10), `to_rows` / `from_rows` and `Estimator` (OP-O3), the einx bridge `pipekit_array.einx` (OP-O2), `KernelSpec` / `KernelOp`, the xarray adapter and the test kit `pipekit_array.testing`.

## Named carriers: dim-level functions

These run on any carrier with a registered `pipekit_ops.NamedArrayAdapter` (a `DimArray`, an `xr.DataArray`, a geoarray carrier), by dim name rather than axis.
Each is a function in `dims` or one of the phase modules below, and, where it serialises, an operator in `dim_operators` (GA-Q3; geostack `docs/design/pipekit_kernels.md`).
Every row ships today.

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
| `rename`     | `Rename`    | new dim names; labels follow (`dim_shapes`)             |
| `expand_dims`| `ExpandDims`| insert length-1 dims                                    |
| `squeeze`    | `Squeeze`   | drop named length-1 dims                                |
| `broadcast_like` | `BroadcastLike` | add another carrier's missing dims, in front    |
| `merge_dims` | `MergeDims` | `y x` into `pixel`, placed last; tuple labels          |
| `split_dims` | `SplitDims` | `pixel` back into `y x`; tuple labels split back        |
| `pad`        | `Pad`       | widen named dims; four modes (`dim_shifts`)             |
| `roll`       | `Roll`      | cyclic shift; labels stay                               |
| `shift`      | `Shift`     | shift with a fill; labels stay                          |
| `diff`       | `Diff`      | n-th forward difference along a dim                     |
| `cumreduce`  | `CumReduce` | cumulative function along a dim (the operator takes a name) |
| `where`      | `Where`     | keep where a mask is True, aligned by dim name          |
| `window`     | `Window`    | sliding full windows as a new last dim (`dim_windows`)  |
| `coarsen`    | `Coarsen`   | reduce non-overlapping blocks; the dim loses its labels |
| `map_over`   | `ApplyAlong`| an operator per item along a dim; serialises            |
| `quantile`   | `Quantile`  | quantiles over named dims; a list `q` adds a `quantile` dim (`dim_algebra`) |
| `dot`        | `Dot`       | sum of a product over named dims; shared dims not summed are batch dims |

All sixteen functions of **[dim_functions.md](dim_functions.md)** ship today: phase 1 in `dim_shapes` (#204), phase 2 in `dim_shifts` (#205), phase 3 in `dim_windows` (#206), phase 4 in `dim_algebra` (#207).
The design page has the xarray parity table and the as-built notes.

An adapter may override any of them with a method of the same name (xarray's `reduce` stays lazy on dask; geoarray's `map_over` is one `jax.vmap`).
The test kit proves the override equals the generic path.

## Reading order

1. **[operators.md](operators.md)**: the data-flow operators (the combinators, the reducer, the carrier-shaped `BatchAlong`, the `Histogram` controller, `Subsample`).
   This is the "what does the pipeline do" surface.
2. **[qc.md](qc.md)**: the numeric QC family.
   This is the "what does the pipeline guard against" surface.
3. **[inference.md](inference.md)**: `ModelOp`, the trained-model wrapper that closes the train-to-serve loop for pipekit-train's Lightning and Keras adapters.
4. **[dim_functions.md](dim_functions.md)**: the dim-level functions, their xarray parity and their testing gates.

Each operator spec covers the constructor signature, the `_apply` contract, the `get_config()` shape, the `compute_output_signature` behaviour (if non-trivial), per-backend caveats, and a one- or two-line example.
