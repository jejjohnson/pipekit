"""Turn any metric operator into a tagged :class:`Scorer`.

The metrics live with their carriers (``xrtoolz.metrics`` for xarray,
``geotoolz`` for GeoTensors); :class:`MetricScorer` wraps one, records its
place in the ``Unit x Lens x Stage`` taxonomy, and returns the named scalar
values every scorer returns. Nothing here imports a carrier library: the
metric's output is read structurally.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pipekit_evaluate.benchmark.protocols import Estimate
from pipekit_evaluate.taxonomy import DEFAULT_STAGES, Lens, ScorerTags, Stage, Unit


#: The key of a metric value that carries no name of its own.
DEFAULT_KEY = "value"


@dataclass(frozen=True)
class MetricScorer:
    """A metric operator as a tagged :class:`Scorer`.

    The metric is any callable ``(prediction, reference) -> result``, such
    as an ``xrtoolz.metrics`` operator. An :class:`Estimate` passed as the
    prediction or the reference is handed to the metric as
    ``from_estimate(estimate)``, by default its point estimate, so the same
    scorer serves :func:`evaluate`-style calls on raw carriers and the
    benchmark ladder's estimates. A probabilistic scorer has no default: a
    point would silently drop the uncertainty it is meant to score, so it
    must say what to take from an estimate (``lambda e: e.sample(100)``).

    The result becomes ``{name: float}``:

    * a Dataset-like result (one with ``data_vars``) gives one entry per
      data variable;
    * a mapping gives one entry per key;
    * anything else is one value, keyed by its ``name`` when it has a
      string one (a named ``DataArray``) and by ``"value"`` otherwise.

    Every value must be a scalar with no dimensions, even singleton ones; a
    metric that keeps a dimension (a score per lead time or per region)
    must be reduced first. Because the values become Python floats, a
    :class:`MetricScorer` is never differentiable; a scorer meant for use
    as a loss implements :class:`Scorer` itself.

    Attributes:
        metric: The metric operator.
        name: The scorer's name; defaults to the metric's ``__name__`` or,
            failing that, its class name.
        unit: What the metric scores.
        lens: What kind of failure it can see.
        stages: The stages it may run at.
        from_estimate: What to hand the metric for an :class:`Estimate`;
            ``None`` means its point estimate, which a probabilistic scorer
            refuses.
    """

    metric: Callable[[Any, Any], Any]
    name: str = ""
    unit: Unit | str = Unit.FIELD
    lens: Lens | str = Lens.POINTWISE
    stages: Stage | str | Iterable[Stage | str] = DEFAULT_STAGES
    from_estimate: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            name = getattr(self.metric, "__name__", type(self.metric).__name__)
            object.__setattr__(self, "name", name)
        # Validate and normalise the tags through the one constructor that
        # owns the rules, so strings become enum members here too.
        tags = ScorerTags(self.unit, self.lens, self.stages)
        object.__setattr__(self, "unit", tags.unit)
        object.__setattr__(self, "lens", tags.lens)
        object.__setattr__(self, "stages", tags.stages)

    def __call__(self, prediction: Any, reference: Any) -> dict[str, float]:
        """Score ``prediction`` against ``reference``.

        Args:
            prediction: A carrier, or an :class:`Estimate`.
            reference: A carrier, or an :class:`Estimate` (an oracle's
                posterior).

        Returns:
            The metric's values by name.

        Raises:
            ValueError: If an estimate cannot be handed to the metric (a
                probabilistic scorer without ``from_estimate``, or an
                estimate with no point), or a metric value is not a scalar.
        """
        result = self.metric(self._carrier(prediction), self._carrier(reference))
        return _as_scores(result, scorer=self.name)

    def _carrier(self, value: Any) -> Any:
        if not isinstance(value, Estimate):
            return value
        if self.from_estimate is not None:
            return self.from_estimate(value)
        if self.lens is Lens.PROBABILISTIC:
            raise ValueError(
                f"scorer {self.name!r} is probabilistic, so its point estimate "
                "would drop the uncertainty it scores; pass from_estimate= to "
                "say what to take from an Estimate (e.g. lambda e: e.sample(100))"
            )
        point = value.point()
        if point is None:
            raise ValueError(
                f"MetricScorer needs a point estimate, but "
                f"{type(value).__name__}.point() returned None"
            )
        return point


def _as_scores(result: Any, *, scorer: str) -> dict[str, float]:
    items: list[tuple[Any, Any]]
    data_vars = getattr(result, "data_vars", None)
    if data_vars is not None:
        items = list(data_vars.items())
    elif isinstance(result, Mapping):
        items = list(result.items())
    else:
        name = getattr(result, "name", None)
        items = [(name if isinstance(name, str) and name else DEFAULT_KEY, result)]
    scores: dict[str, float] = {}
    for key, value in items:
        # float() accepts a size-1 array or a DataArray with a singleton
        # dimension, which would silently drop that dimension.
        if getattr(value, "ndim", 0) != 0:
            raise _not_scalar(scorer, key)
        try:
            scores[str(key)] = float(value)
        except (TypeError, ValueError) as err:
            raise _not_scalar(scorer, key) from err
    return scores


def _not_scalar(scorer: str, key: Any) -> ValueError:
    return ValueError(
        f"scorer {scorer!r}: metric value {key!r} is not a scalar; "
        "reduce it over every dimension before scoring"
    )
