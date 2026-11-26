"""pipekit-evaluate — multidimensional evaluation framework.

Three areas, each with one job. An area imports only from the ones before it.

- `pipekit_evaluate.scoring`: the ``Unit x Lens x Stage`` tags, `MetricScorer`,
  the `Estimate` and `Scorer` protocols, and `ReferenceSource`.
- `pipekit_evaluate.reports`: `evaluate` and the content-hashed
  `EvaluationReport`.
- `pipekit_evaluate.benchmark`: the orchestration layer of the benchmark ladder,
  the ``level x rung x complexity`` cube, the `Task`, `Estimator` and `Oracle`
  protocols, the reference rule and `Benchmark`.

This package re-exports the names it always has. Import a name from the
package or from its area, never from a file in it.
"""

from __future__ import annotations

from pipekit_evaluate import benchmark
from pipekit_evaluate.benchmark import (
    Axis,
    Benchmark,
    BenchmarkReport,
    Cell,
    CellResult,
    Estimator,
    Oracle,
    Prior,
    ReferenceRule,
    Task,
)
from pipekit_evaluate.reports import EvaluationReport, ReportEntry, evaluate
from pipekit_evaluate.scoring import (
    DEFAULT_STAGES,
    CovarianceEstimate,
    Estimate,
    Lens,
    MetricScorer,
    ReferenceSource,
    Scorer,
    ScorerTags,
    Stage,
    Unit,
    tags_of,
)


__version__ = "0.2.0"  # x-release-please-version

__all__ = [
    "DEFAULT_STAGES",
    "Axis",
    "Benchmark",
    "BenchmarkReport",
    "Cell",
    "CellResult",
    "CovarianceEstimate",
    "Estimate",
    "Estimator",
    "EvaluationReport",
    "Lens",
    "MetricScorer",
    "Oracle",
    "Prior",
    "ReferenceRule",
    "ReferenceSource",
    "ReportEntry",
    "Scorer",
    "ScorerTags",
    "Stage",
    "Task",
    "Unit",
    "benchmark",
    "evaluate",
    "tags_of",
]
