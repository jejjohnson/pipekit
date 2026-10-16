"""pipekit-evaluate — multidimensional evaluation framework.

The ``benchmark`` subpackage ships the orchestration layer of the
benchmark ladder (the ``level x rung x complexity`` cube, the
carrier-agnostic protocols, and the reference rule). The ``taxonomy``
module carries the ``Unit x Lens x Stage`` tags, ``scorers`` turns any
metric operator into a tagged scorer, and ``report`` collects a run's
tagged scores into a content-hashed :class:`EvaluationReport`.
"""

from __future__ import annotations

from pipekit_evaluate import benchmark
from pipekit_evaluate.benchmark import (
    Axis,
    Benchmark,
    BenchmarkReport,
    Cell,
    CellResult,
    CovarianceEstimate,
    Estimate,
    Estimator,
    Oracle,
    Prior,
    ReferenceRule,
    ReferenceSource,
    Scorer,
    Task,
)
from pipekit_evaluate.report import EvaluationReport, ReportEntry, evaluate
from pipekit_evaluate.scorers import MetricScorer
from pipekit_evaluate.taxonomy import (
    DEFAULT_STAGES,
    Lens,
    ScorerTags,
    Stage,
    Unit,
    tags_of,
)


__version__ = "0.0.2"  # x-release-please-version

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
