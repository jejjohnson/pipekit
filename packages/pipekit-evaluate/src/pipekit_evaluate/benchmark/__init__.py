"""The benchmark ladder — orchestration layer of ``pipekit-evaluate``.

This subpackage implements the domain-independent framework described in
``packages/pipekit-evaluate/docs/design/benchmark_ladder.md``: the
``level x rung x complexity`` cube, the carrier-agnostic protocols, and the
"simpler thing is ground truth" reference rule. It sits *on top of* the
``Unit x Lens x Stage`` scorer taxonomy (:mod:`pipekit_evaluate.taxonomy`)
— it selects and invokes scorers, it does not replace them.
"""

from __future__ import annotations

from pipekit_evaluate.benchmark.cube import (
    LEVEL_NAMES,
    RUNG_NAMES,
    Cell,
)
from pipekit_evaluate.benchmark.protocols import (
    CovarianceEstimate,
    Estimate,
    Estimator,
    Oracle,
    Prior,
    Scorer,
    Task,
)
from pipekit_evaluate.benchmark.reference import (
    DEFAULT_ALSO,
    DEFAULT_COMPLEXITY,
    DEFAULT_STAGED,
    Axis,
    ReferenceRule,
    ReferenceSource,
)
from pipekit_evaluate.benchmark.run import (
    Benchmark,
    BenchmarkReport,
    CellResult,
)


__all__ = [
    "DEFAULT_ALSO",
    "DEFAULT_COMPLEXITY",
    "DEFAULT_STAGED",
    "LEVEL_NAMES",
    "RUNG_NAMES",
    "Axis",
    "Benchmark",
    "BenchmarkReport",
    "Cell",
    "CellResult",
    "CovarianceEstimate",
    "Estimate",
    "Estimator",
    "Oracle",
    "Prior",
    "ReferenceRule",
    "ReferenceSource",
    "Scorer",
    "Task",
]
