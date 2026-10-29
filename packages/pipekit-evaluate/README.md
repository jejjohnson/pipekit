# pipekit-evaluate

> **Status — v0.0 / early.** The `benchmark` subpackage (the orchestration layer of the benchmark ladder), the `Unit × Lens × Stage` scorer taxonomy and the evaluation report have landed.

## `benchmark` — the orchestration layer (implemented)

The domain-independent framework from
[`docs/design/benchmark_ladder.md`](docs/design/benchmark_ladder.md): the
`level × rung × complexity` cube, carrier-agnostic protocols, and the
"simpler thing is ground truth" reference rule. It *selects and invokes*
scorers — it does not replace the taxonomy below.

```python
from pipekit_evaluate.benchmark import Benchmark, Axis, ReferenceRule

report = Benchmark(task=task, estimators=estimators, scorers=scorers).run()
report.to_dict()["cells"]["L4/r5/c0"]["scores"]
# {"oracle_posterior": {...}, "known_theta": {...}}
```

- **Protocols** (`Task`, `Estimator`, `Estimate`, `CovarianceEstimate`,
  `Oracle`, `Prior`, `Scorer`) — runtime-checkable, satisfied structurally by domain packages
  (`plumax`, `somax`, `xtremax`) without importing `pipekit-evaluate`. The
  `Task` reuses `pipekit-cycle`'s `ForwardModel` / `ObservationOperator`
  (install the `cycle` extra for the concrete types).
- **Cube** (`Cell`, `RUNG_NAMES`, `LEVEL_NAMES`) — the
  `level × rung × complexity` coordinate.
- **Reference rule** (`ReferenceRule`, `ReferenceSource`, `Axis`) — the
  §4.3 invariant as data, resolved along both the staged and the
  per-rung complexity axes.
- **Several references per cell** — `ReferenceRule.also` names further references a rung is scored against beside its primary one; by default the amortized rung (5) has the oracle's posterior and the known truth. A scorer is run against the references it names in its optional `references` attribute (`MetricScorer(..., references={"oracle_posterior", "known_theta"})`), and against the primary one only when it names none, so a scorer written for a posterior is never handed the raw truth. `CellResult.by_reference` holds the scores against each.

## `Unit × Lens × Stage` scorers (implemented)

A scorer places itself on three orthogonal axes:

- **Unit** (what's scored): field, statistic, trajectory, event, budget.
- **Lens** (what kind of failure it can see): point-wise, probabilistic, spectral, structural, detection, physical-constraint.
- **Stage** (where it may run): training, validation, final, monitoring.

The tags live on the scorer, not on the metric: metrics stay in the carrier libraries (`xrops.metrics`, `geotoolz`), which never import `pipekit-evaluate`. `MetricScorer` wraps any `(prediction, reference) -> result` metric operator, tags it, and flattens its result (a Dataset, a mapping or a scalar) into named floats. `tags_of` reads any scorer's tags; an untagged scorer is a point-wise field score for validation and final evaluation.

```python
import pipekit_evaluate as pev
from xrops.metrics import RMSE, SSIM

scorers = [
    pev.MetricScorer(RMSE("ch4", dims="x")),
    pev.MetricScorer(SSIM("ch4", ["y", "x"]), lens="structural", stages={"final"}),
]
pev.tags_of(scorers[1]).lens  # Lens.STRUCTURAL
```

## `EvaluationReport` (implemented)

`evaluate` runs every scorer on one prediction and one reference and returns an `EvaluationReport`: one tagged entry per value, keyed `"<scorer>.<metric>"`. The report filters by axis, diffs against an earlier report, and has a content hash over its entries, references and config that ignores entry order and the timestamp.

```python
report = pev.evaluate(prediction, truth, scorers, model_ref=model_hash)
report.filter(lens="structural").to_pandas()
report.diff(last_week)   # {key: (last week's value, this value)} for changed keys
report.content_hash      # equal for two runs that score the same values
```

A benchmark-ladder run becomes the same kind of report: `Benchmark(...).run().to_report()` gives one entry per scored value, keyed `"<cell>:<reference>:<scorer>.<metric>"`, so `report.filter(reference="known_theta")` cuts across the cube.

The design is [geostack Report 14](https://github.com/jejjohnson/geostack/blob/main/docs/reports/14_pipekit_evaluate.md), workstream A.
