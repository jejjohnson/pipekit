# pipekit-array

> **Status — v0.0 / design locked, implementation pending.** The
> source directory contains no operators yet; the [v0.1 design
> doc][design] is the load-bearing artifact for this package.

Duck-typed array operators on top of `pipekit`, implemented against
the [Python Array API standard](https://data-apis.org/array-api/).
One operator, all backends — numpy, JAX, CuPy, PyTorch, dask — via
`array_namespace(x)` dispatch.

The v0.1 catalogue (12 operators, 7 modules; specs in
[`docs/design/api/`][design-api]):

- **Combinators** (`ApplyToBands`, `StackAlong`, `ConcatenateAlong`)
- **Geometry** (`Subsample`)
- **Observation** (`Histogram` — carrier-aware controller)
- **Reduction** (`MeanScalar`)
- **Parallelism** (`BatchedMap` — array-shaped; see
  [ADR A5][adr-a5])
- **QC** (`Diff`, `AssertValueRange`, `AssertNoNaN`,
  `AssertValidFraction`)
- **Inference** (`ModelOp` — framework-agnostic trained-model
  wrapper; see [ADR A8][adr-a8])

## Design doc

- [`docs/design/`][design] — index, vision, architecture, ADRs,
  boundaries, per-operator specs.
- [Report 3 — Sister libraries][report-3] is the upstream master-plan
  source.

[design]: docs/design/README.md
[design-api]: docs/design/api/README.md
[adr-a5]: docs/design/decisions.md#a5-pipekit-arraybatchedmap-is-array-shaped-not-a-shadow
[adr-a8]: docs/design/decisions.md#a8-modelop-split-pipekit-array-vs-pipekit-jax
[report-3]: https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_3_pipekit_array.md
