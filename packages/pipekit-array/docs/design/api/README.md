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
| `reduce`           | `MeanScalar`            | `xp.mean` reduction; optional `axis`.                         |
| `parallel`         | `BatchedMap`            | Split along axis 0 into chunks, apply `inner`, concatenate.   |
| `qc`               | `Diff`                  | Compare to a stored reference; raise on numeric drift.        |
| `qc`               | `AssertValueRange`      | Pass-through; raise/warn on out-of-range.                     |
| `qc`               | `AssertNoNaN`           | Pass-through; raise on any NaN.                               |
| `qc`               | `AssertValidFraction`   | Pass-through; raise if `< min_valid` non-NaN fraction.        |
| `inference`        | `ModelOp`               | Framework-agnostic inference wrapper.                          |

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
