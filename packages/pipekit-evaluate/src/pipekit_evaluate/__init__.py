"""pipekit-evaluate — multidimensional evaluation framework.

The ``benchmark`` subpackage ships the orchestration layer of the
benchmark ladder (the ``level x rung x complexity`` cube, the
carrier-agnostic protocols, and the reference rule). The
``Unit x Lens x Stage`` scorer taxonomy is still planned.
"""

from __future__ import annotations

from pipekit_evaluate import benchmark
from pipekit_evaluate.benchmark import (
    Axis,
    Benchmark,
    BenchmarkReport,
    Cell,
    CellResult,
    Estimate,
    Estimator,
    Oracle,
    Prior,
    ReferenceRule,
    ReferenceSource,
    Scorer,
    Task,
)


__version__ = "0.0.1"  # x-release-please-version

__all__ = [
    "Axis",
    "Benchmark",
    "BenchmarkReport",
    "Cell",
    "CellResult",
    "Estimate",
    "Estimator",
    "Oracle",
    "Prior",
    "ReferenceRule",
    "ReferenceSource",
    "Scorer",
    "Task",
    "benchmark",
]
