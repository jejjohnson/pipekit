"""The estimate a method returns and the scorer that grades it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Mapping


@runtime_checkable
@runtime_checkable
class Estimate(Protocol):
    """A point and/or a posterior produced by an estimator.

    Scorers branch on what is present: a point-only estimate returns
    ``None`` from :meth:`sample`. Per ``benchmark_ladder.md`` §8 decision
    1, :meth:`sample` is the required posterior surface and
    :meth:`log_prob` is optional.

    Members:
        point(): Best point estimate (mean / MAP / MLE), or ``None``.
        sample(n, key): ``n`` posterior draws, or ``None`` if point-only.
        log_prob(theta): Optional posterior log-density at ``theta``.
    """

    def point(self) -> Any: ...

    def sample(self, n: int, *, key: Any = None) -> Any: ...

    def log_prob(self, theta: Any) -> Any: ...


@runtime_checkable
class CovarianceEstimate(Estimate, Protocol):
    """An :class:`Estimate` that also reports its posterior covariance.

    A separate protocol rather than a member of :class:`Estimate`, so every
    existing estimate keeps satisfying :class:`Estimate`. A scorer that needs
    second moments (Gaussian coverage, a Mahalanobis distance, a spread-skill
    ratio) checks ``isinstance(estimate, CovarianceEstimate)`` instead of
    probing attributes or drawing samples.

    Members:
        covariance(): The posterior covariance of :meth:`point`'s quantity,
            as a ``(d, d)`` matrix for a ``(d,)`` point.
    """

    def covariance(self) -> Any: ...


@runtime_checkable
class Scorer(Protocol):
    """Score one prediction against one reference.

    This is the seam onto the ``Unit x Lens x Stage`` taxonomy: a concrete
    scorer *is* a ``(Unit, Lens)`` pair. The benchmark layer only requires
    a name and a callable that returns a mapping of named metric values.
    A scorer places itself in the taxonomy through the optional attributes
    ``unit``, ``lens``, ``stages`` and ``differentiable``, which are not
    protocol members; :func:`pipekit_evaluate.taxonomy.tags_of` reads them
    with their defaults, and :class:`pipekit_evaluate.scoring.MetricScorer`
    sets them on any metric operator. A further optional attribute,
    ``references``, names the reference sources a scorer can be run
    against in the benchmark ladder; without it a scorer is run against
    each cell's primary reference only
    (:func:`pipekit_evaluate.benchmark.run.scores_against`).

    Members:
        name: Short scorer identifier, used to namespace its metrics.
        __call__(prediction, reference): Named metric values.
    """

    name: str

    def __call__(self, prediction: Estimate, reference: Any) -> Mapping[str, float]: ...
