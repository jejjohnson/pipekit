"""The ``Unit x Lens x Stage`` taxonomy, carried by scorers as data.

An evaluation picks one value on each of three axes:

* **Unit** — what is scored: a field, a statistic, a trajectory, an event
  or a budget.
* **Lens** — what kind of failure the score can see: point-wise,
  probabilistic, spectral, structural, detection or physical-constraint.
* **Stage** — where in a model's life the score may run (training,
  validation, final evaluation, monitoring), which bounds how cheap and
  differentiable it must be.

The tags live on the scorer, not on the metric. The metrics stay in the
carrier libraries (``xrops.metrics``), which never import
``pipekit-evaluate``; a scorer declares its tags as optional attributes
(``unit``, ``lens``, ``stages``, ``differentiable``) and :func:`tags_of`
reads them, filling in the defaults. A scorer with only a ``name`` is
therefore a point-wise field score for validation and final evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Unit(StrEnum):
    """What a score is computed on."""

    FIELD = "field"
    STATISTIC = "statistic"
    TRAJECTORY = "trajectory"
    EVENT = "event"
    BUDGET = "budget"


class Lens(StrEnum):
    """What kind of failure a score can see."""

    POINTWISE = "pointwise"
    PROBABILISTIC = "probabilistic"
    SPECTRAL = "spectral"
    STRUCTURAL = "structural"
    DETECTION = "detection"
    PHYSICAL = "physical"


class Stage(StrEnum):
    """Where in a model's life a score may run."""

    TRAINING = "training"
    VALIDATION = "validation"
    FINAL = "final"
    MONITORING = "monitoring"


#: The stages an untagged scorer runs at: neither cheap nor differentiable
#: enough to be assumed fit for training or for monitoring.
DEFAULT_STAGES: frozenset[Stage] = frozenset({Stage.VALIDATION, Stage.FINAL})


def _as_stages(stages: Stage | str | Iterable[Stage | str]) -> frozenset[Stage]:
    # A bare string is one stage, not an iterable of characters.
    if isinstance(stages, str):
        stages = (stages,)
    return frozenset(Stage(s) for s in stages)


@dataclass(frozen=True, init=False)
class ScorerTags:
    """A scorer's position in the taxonomy.

    Strings are accepted wherever an enum member is, and one stage may be
    given on its own, so ``ScorerTags(lens="structural", stages="final")``
    is valid; the fields always hold enum members after construction.

    Attributes:
        unit: What the scorer scores.
        lens: What kind of failure it can see.
        stages: The stages it may run at; never empty.
        differentiable: Whether its value can be differentiated with
            respect to the prediction (a requirement for use as a loss).
    """

    unit: Unit
    lens: Lens
    stages: frozenset[Stage]
    differentiable: bool

    # A hand-written __init__ so the accepted inputs (strings, one stage)
    # are in the signature, while the fields keep their normalised types.
    def __init__(
        self,
        unit: Unit | str = Unit.FIELD,
        lens: Lens | str = Lens.POINTWISE,
        stages: Stage | str | Iterable[Stage | str] = DEFAULT_STAGES,
        differentiable: bool = False,
    ) -> None:
        normalised = _as_stages(stages)
        if not normalised:
            raise ValueError("a scorer must run at one stage or more")
        object.__setattr__(self, "unit", Unit(unit))
        object.__setattr__(self, "lens", Lens(lens))
        object.__setattr__(self, "stages", normalised)
        object.__setattr__(self, "differentiable", bool(differentiable))

    def runs_at(self, stage: Stage | str) -> bool:
        """Whether the scorer may run at ``stage``."""
        return Stage(stage) in self.stages


def tags_of(scorer: Any) -> ScorerTags:
    """Read a scorer's taxonomy tags, defaulting the ones it does not set.

    Args:
        scorer: Any object; the optional attributes ``unit``, ``lens``,
            ``stages`` and ``differentiable`` are read when present.

    Returns:
        The scorer's :class:`ScorerTags`. A scorer with none of the
        attributes is a point-wise field score for validation and final
        evaluation, and not differentiable.

    Raises:
        ValueError: If an attribute holds a value outside its axis.
    """
    defaults = ScorerTags()
    return ScorerTags(
        unit=getattr(scorer, "unit", defaults.unit),
        lens=getattr(scorer, "lens", defaults.lens),
        stages=getattr(scorer, "stages", defaults.stages),
        differentiable=bool(getattr(scorer, "differentiable", defaults.differentiable)),
    )
