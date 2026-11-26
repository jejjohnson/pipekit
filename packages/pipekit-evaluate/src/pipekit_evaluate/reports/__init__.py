"""A run's tagged scores, collected into one report.

`evaluate` scores a run with a set of scorers, and `EvaluationReport` holds the
`ReportEntry` of each score under a content hash.
"""

from __future__ import annotations

from pipekit_evaluate.reports._report import EvaluationReport, ReportEntry, evaluate


__all__ = [
    "EvaluationReport",
    "ReportEntry",
    "evaluate",
]
