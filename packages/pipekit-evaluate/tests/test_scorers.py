"""Tests for the scorer taxonomy and the metric adapter.

The metric outputs of carrier libraries (an xarray ``Dataset`` or a named
``DataArray``) are faked structurally, so these tests need no carrier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from pipekit_evaluate import (
    DEFAULT_STAGES,
    Lens,
    MetricScorer,
    Scorer,
    ScorerTags,
    Stage,
    Unit,
    tags_of,
)


def rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - reference) ** 2)))


@dataclass
class PointEstimate:
    value: Any

    def point(self) -> Any:
        return self.value

    def sample(self, n: int, *, key: Any = None) -> Any:
        return None

    def log_prob(self, theta: Any) -> Any:
        raise NotImplementedError


@dataclass
class NamedScalar:
    """A 0-d result with a ``name``, like a reduced ``xr.DataArray``."""

    name: str | None
    value: float

    def __float__(self) -> float:
        return self.value


@dataclass
class DatasetLike:
    """A result with ``data_vars``, like an ``xr.Dataset``."""

    data_vars: dict[str, Any]


# --------------------------------------------------------------------------- #
# Gate A1: tags
# --------------------------------------------------------------------------- #
def test_untagged_scorer_is_a_pointwise_field_score() -> None:
    @dataclass
    class Bare:
        name: str = "bare"

        def __call__(self, prediction: Any, reference: Any) -> dict[str, float]:
            return {}

    tags = tags_of(Bare())
    assert tags == ScorerTags(Unit.FIELD, Lens.POINTWISE, DEFAULT_STAGES, False)
    assert tags.stages == {Stage.VALIDATION, Stage.FINAL}


def test_tags_of_reads_the_scorers_own_tags() -> None:
    @dataclass
    class Spectral:
        name: str = "spectral"
        lens: str = "spectral"
        stages: tuple[str, ...] = ("final",)

    tags = tags_of(Spectral())
    assert (tags.unit, tags.lens, tags.stages) == (
        Unit.FIELD,
        Lens.SPECTRAL,
        {Stage.FINAL},
    )


def test_tags_accept_strings_and_a_single_stage() -> None:
    tags = ScorerTags(unit="event", lens="detection", stages="monitoring")
    assert tags.unit is Unit.EVENT
    assert tags.lens is Lens.DETECTION
    # one stage, not the characters of "monitoring"
    assert tags.stages == {Stage.MONITORING}
    assert tags.runs_at("monitoring")
    assert not tags.runs_at(Stage.FINAL)


@pytest.mark.parametrize(
    "kwargs",
    [{"lens": "vibes"}, {"unit": "pixel"}, {"stages": ["later"]}, {"stages": ()}],
)
def test_tags_reject_values_off_their_axis(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ScorerTags(**kwargs)


def test_metric_scorer_reports_its_tags() -> None:
    scorer = MetricScorer(
        rmse, lens="structural", stages={"validation", "final"}, unit="statistic"
    )
    assert isinstance(scorer, Scorer)
    assert tags_of(scorer) == ScorerTags(
        Unit.STATISTIC, Lens.STRUCTURAL, frozenset({Stage.VALIDATION, Stage.FINAL})
    )
    assert scorer.lens is Lens.STRUCTURAL


def test_metric_scorer_name_defaults_to_the_metric() -> None:
    class RMSE:
        def __call__(self, prediction: Any, reference: Any) -> float:
            return rmse(prediction, reference)

    assert MetricScorer(rmse).name == "rmse"
    assert MetricScorer(RMSE()).name == "RMSE"
    assert MetricScorer(rmse, name="err").name == "err"


# --------------------------------------------------------------------------- #
# Gate A2: adapter parity on Report 14's stage 1
# --------------------------------------------------------------------------- #
@pytest.fixture
def plume() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A Gaussian plume, the plume shifted by 6 cells, and the flat field.

    Peak 50, width 3 cells, on a periodic transect of 40 cells.
    """
    n, sigma, shift = 40, 3.0, 6
    cells = np.arange(n)
    distance = np.minimum(abs(cells - n // 2), n - abs(cells - n // 2))
    truth = 50.0 * np.exp(-(distance**2) / (2 * sigma**2))
    return truth, np.roll(truth, shift), np.full(n, truth.mean())


def test_metric_scorer_matches_the_shift_error_closed_form(
    plume: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    truth, shifted, flat = plume
    sigma, shift = 3.0, 6
    # RMSE(A)^2 = 2 <x^2> (1 - exp(-d^2 / 4 sigma^2)) for a well-sampled
    # Gaussian; the sampled sums agree with it to about 1e-13 here.
    expected = np.sqrt(
        2 * np.mean(truth**2) * (1 - np.exp(-(shift**2) / (4 * sigma**2)))
    )
    scorer = MetricScorer(rmse)
    assert scorer(shifted, truth)["value"] == pytest.approx(expected, rel=1e-12)
    # Report 14's table: the displaced plume scores 20.50, the flat field 15.62.
    assert round(scorer(shifted, truth)["value"], 2) == 20.50
    assert round(scorer(flat, truth)["value"], 2) == 15.62


def test_metric_scorer_scores_an_estimates_point(
    plume: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    truth, shifted, _ = plume
    scorer = MetricScorer(rmse)
    direct = scorer(shifted, truth)
    assert scorer(PointEstimate(shifted), PointEstimate(truth)) == direct


def test_metric_scorer_refuses_an_estimate_without_a_point() -> None:
    with pytest.raises(ValueError, match="point estimate"):
        MetricScorer(rmse)(PointEstimate(None), np.zeros(3))


# --------------------------------------------------------------------------- #
# Result shapes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (DatasetLike({"ch4": np.float64(2.0), "co2": 3}), {"ch4": 2.0, "co2": 3.0}),
        ({"shift_x": 6, "residual_rmse": 0.0}, {"shift_x": 6.0, "residual_rmse": 0.0}),
        (NamedScalar("ch4", 20.5), {"ch4": 20.5}),
        (NamedScalar(None, 20.5), {"value": 20.5}),
        (np.float64(1.5), {"value": 1.5}),
        (np.array(1.5), {"value": 1.5}),
    ],
)
def test_metric_scorer_flattens_each_result_shape(
    result: Any, expected: dict[str, float]
) -> None:
    scores = MetricScorer(lambda p, r: result, name="m")(None, None)
    assert scores == expected
    assert all(type(v) is float for v in scores.values())


@pytest.mark.parametrize("value", [np.ones(3), np.ones(1), np.ones((1, 1))])
def test_metric_scorer_rejects_a_value_that_keeps_a_dimension(value: Any) -> None:
    # a singleton dimension (one lead time) is still a dimension
    scorer = MetricScorer(lambda p, r: {"by_lead": value}, name="skill")
    with pytest.raises(ValueError, match="'by_lead' is not a scalar"):
        scorer(None, None)


def test_metric_scorer_is_never_differentiable() -> None:
    # its values become Python floats, which no autodiff can see through
    assert tags_of(MetricScorer(rmse)).differentiable is False


# --------------------------------------------------------------------------- #
# What a scorer takes from an Estimate
# --------------------------------------------------------------------------- #
@dataclass
class SampledEstimate(PointEstimate):
    draws: Any = None

    def sample(self, n: int, *, key: Any = None) -> Any:
        return self.draws[:n]


def test_probabilistic_scorer_refuses_to_score_only_the_point() -> None:
    def spread(prediction: Any, reference: Any) -> float:
        return float(np.std(prediction))

    scorer = MetricScorer(spread, lens="probabilistic")
    estimate = SampledEstimate(0.0, draws=np.array([-1.0, 1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="from_estimate="):
        scorer(estimate, 0.0)
    # raw carriers (an ensemble array) need no conversion
    assert scorer(estimate.draws, 0.0) == {"value": 1.0}


def test_from_estimate_says_what_the_metric_is_handed() -> None:
    def spread(prediction: Any, reference: Any) -> float:
        return float(np.std(prediction))

    scorer = MetricScorer(
        spread, lens="probabilistic", from_estimate=lambda e: e.sample(4)
    )
    estimate = SampledEstimate(0.0, draws=np.array([-1.0, 1.0, -1.0, 1.0]))
    assert scorer(estimate, PointEstimate(0.0)) == {"value": 1.0}
