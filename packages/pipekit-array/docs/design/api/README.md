---
status: draft
version: 0.1.0
---

# API — Operator Catalog

The v0.1 surface is 12 operators across 7 modules. The grouping
mirrors `pipekit-ops` core where it makes sense
(`combinators` ↔ `pipekit_ops.combine`, `qc` ↔ `pipekit_ops.qc`,
`observe` ↔ `pipekit_ops.observe`, `parallel` ↔ `pipekit_ops.parallel`).

## At a glance

| Module             | Operator                | One-liner                                                     |
| ------------------ | ----------------------- | ------------------------------------------------------------- |
| `dim_operators`    | `ApplyAlong`            | `inner` per item along a dim, stacked back; serialises (A15). |
| `combinators`      | `StackAlong`            | `xp.stack` list-of-arrays along a new axis.                   |
| `combinators`      | `ConcatenateAlong`      | `xp.concatenate` along an existing axis.                      |
| `geom`             | `Subsample`             | Stride-decimate along given axes.                             |
| `observe`          | `Histogram`             | Controller; `.at(key)` captures per-tap distributions.        |
| `reductions`       | `MeanScalar`            | `xp.mean` reduction; optional `axis`.                         |
| `batching`         | `BatchAlong`            | Split along a dim into chunks, apply `inner`, concatenate (A19). |
| `qc`               | `AssertClose`           | Compare to a stored reference; raise on numeric drift (A21).  |
| `qc`               | `AssertValueRange`      | Pass-through; raise/warn on out-of-range.                     |
| `qc`               | `AssertNoNaN`           | Pass-through; raise on any NaN.                               |
| `qc`               | `AssertValidFraction`   | Pass-through; raise if `< min_valid` non-NaN fraction.        |
| `inference`        | `ModelOp`               | Framework-agnostic inference wrapper.                          |

## Named carriers: dim-level functions

On any carrier with a registered `pipekit_ops.NamedArrayAdapter` (a `DimArray`,
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

The first fourteen of **[dim_functions.md](dim_functions.md)** ship today (phase 1 in `dim_shapes`, phase 2 in `dim_shifts`, phase 3 in `dim_windows`); the rest (`quantile`, `dot`) are proposed there, with an xarray parity table.

An adapter may override any of them with a method of the same name
(xarray's `reduce` stays lazy on dask; geoarray's `map_over` is one
`jax.vmap`); the test kit proves the override equals the generic path.

## Reading order

1. **[operators.md](operators.md)** — the data-flow operators (the
   combinators, the reducer, the carrier-shaped `BatchAlong`, the
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
