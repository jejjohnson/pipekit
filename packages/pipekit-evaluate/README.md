# pipekit-evaluate

> **Status — v0.0 / early.** The `benchmark` subpackage (the orchestration layer of the benchmark ladder) and the `Unit × Lens × Stage` scorer taxonomy have landed.

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

## `Unit × Lens × Stage` scorers (implemented)

A scorer places itself on three orthogonal axes:

- **Unit** (what's scored): field, statistic, trajectory, event, budget.
- **Lens** (what kind of failure it can see): point-wise, probabilistic, spectral, structural, detection, physical-constraint.
- **Stage** (where it may run): training, validation, final, monitoring.

The tags live on the scorer, not on the metric: metrics stay in the carrier libraries (`xrtoolz.metrics`, `geotoolz`), which never import `pipekit-evaluate`. `MetricScorer` wraps any `(prediction, reference) -> result` metric operator, tags it, and flattens its result (a Dataset, a mapping or a scalar) into named floats. `tags_of` reads any scorer's tags; an untagged scorer is a point-wise field score for validation and final evaluation.

```python
import pipekit_evaluate as pev
from xrtoolz.metrics import RMSE, SSIM

scorers = [
    pev.MetricScorer(RMSE("ch4", dims="x")),
    pev.MetricScorer(SSIM("ch4", ["y", "x"]), lens="structural", stages={"final"}),
]
pev.tags_of(scorers[1]).lens  # Lens.STRUCTURAL
```

The design is [geostack Report 14](https://github.com/jejjohnson/geostack/blob/main/docs/reports/14_pipekit_evaluate.md), workstream A.
