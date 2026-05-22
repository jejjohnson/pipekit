# pipekit-array

> **Status — v0.0 / planning only.** This package is scaffolded for
> future work; the source directory currently contains no operators.

Planned: duck-typed array operators on top of `pipekit`, implemented
against the [Python Array API standard](https://data-apis.org/array-api/).
Multi-backend (numpy, JAX, CuPy, PyTorch) via `array_namespace`
dispatch.

Planned operator catalogue (from the master plan):
`ApplyToBands`, `Subsample`, `Diff`, `AssertValueRange`, `AssertNoNaN`,
`AssertValidFraction`, `ModelOp`, `BatchedMap`, `MeanScalar`,
`StackAlong`, `ConcatenateAlong`, `Histogram` (carrier-aware version).

See [Report 3 — Sister libraries](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_3_pipekit_array.md).
