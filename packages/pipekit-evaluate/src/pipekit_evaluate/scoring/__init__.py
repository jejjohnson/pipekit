"""Scores and what they are tagged with.

`Unit`, `Lens` and `Stage` tag a scorer by what it scores, how it looks at it, and when
in a run it applies, as `ScorerTags` (`tags_of` reads them). `MetricScorer` turns any
metric operator into a tagged scorer. `Estimate`, `CovarianceEstimate` and `Scorer` are
the protocols of what a method returns and what grades it, and `ReferenceSource` (with
`as_sources`) says where a score's ground truth comes from.
"""

from __future__ import annotations

from pipekit_evaluate.scoring._metric import MetricScorer
from pipekit_evaluate.scoring._protocols import CovarianceEstimate, Estimate, Scorer
from pipekit_evaluate.scoring._sources import ReferenceSource, as_sources
from pipekit_evaluate.scoring._taxonomy import (
    DEFAULT_STAGES,
    Lens,
    ScorerTags,
    Stage,
    Unit,
    tags_of,
)


__all__ = [
    "DEFAULT_STAGES",
    "CovarianceEstimate",
    "Estimate",
    "Lens",
    "MetricScorer",
    "ReferenceSource",
    "Scorer",
    "ScorerTags",
    "Stage",
    "Unit",
    "as_sources",
    "tags_of",
]
