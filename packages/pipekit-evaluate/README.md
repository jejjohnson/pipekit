# pipekit-evaluate

> **Status — v0.0 / early.** The `benchmark` subpackage (the
> orchestration layer of the benchmark ladder) has landed; the
> `Unit × Lens × Stage` scorer taxonomy is still planned.

## `benchmark` — the orchestration layer (implemented)

The domain-independent framework from
[`docs/design/benchmark_ladder.md`](docs/design/benchmark_ladder.md): the
`level × rung × complexity` cube, carrier-agnostic protocols, and the
"simpler thing is ground truth" reference rule. It *selects and invokes*
scorers — it does not replace the taxonomy below.

```python
from pipekit_evaluate.benchmark import Benchmark, Axis, ReferenceRule

report = Benchmark(task=task, estimators=estimators, scorers=scorers).run()
report.to_dict()  # {cell label: {metric: value}}
```

- **Protocols** (`Task`, `Estimator`, `Estimate`, `Oracle`, `Prior`,
  `Scorer`) — runtime-checkable, satisfied structurally by domain packages
  (`plumax`, `somax`, `xtremax`) without importing `pipekit-evaluate`. The
  `Task` reuses `pipekit-cycle`'s `ForwardModel` / `ObservationOperator`
  (install the `cycle` extra for the concrete types).
- **Cube** (`Cell`, `RUNG_NAMES`, `LEVEL_NAMES`) — the
  `level × rung × complexity` coordinate.
- **Reference rule** (`ReferenceRule`, `ReferenceSource`, `Axis`) — the
  §4.3 invariant as data, resolved along both the staged and the
  per-rung complexity axes.

## `Unit × Lens × Stage` scorers (planned)

A multidimensional scorer taxonomy with three orthogonal axes:

- **Unit** (what's scored): Field / Statistic / Trajectory / Event / Budget
- **Lens** (kind of critique): Point-wise / Probabilistic / Spectral /
  Structural / Detection / Physical-constraint
- **Stage** (lifecycle): Training / Validation / Final / Monitoring

The scorers depend on `pipekit-array` (for the array reductions)
and on `xr_toolz.lagrangian` / `xr_toolz.events` (for the
trajectory- and event-flavoured units).

See [Report 11 — pipekit-evaluate](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_11_pipekit_evaluate.md).
