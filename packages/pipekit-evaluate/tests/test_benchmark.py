"""Tests for the benchmark orchestration layer.

Uses tiny synthetic fakes that satisfy the carrier-agnostic protocols, so
the framework is exercised end-to-end without any domain package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest
from pipekit_evaluate.benchmark import (
    Axis,
    Benchmark,
    Cell,
    Estimate,
    Estimator,
    ReferenceRule,
    ReferenceSource,
    Scorer,
    Task,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
@dataclass
class FakeEstimate:
    value: float

    def point(self) -> float:
        return self.value

    def sample(self, n: int, *, key: Any = None) -> list[float]:
        return [self.value] * n

    def log_prob(self, theta: Any) -> float:
        return 0.0


@dataclass
class FakeEstimator:
    rung: int
    complexity: int
    value: float

    def fit(self, task: Task) -> FakeEstimator:
        return self

    def __call__(self, obs: Any) -> FakeEstimate:
        return FakeEstimate(self.value)


@dataclass
class FakeTask:
    name: str = "toy"
    domain: str = "test"
    data: Mapping[str, Any] = field(
        default_factory=lambda: {"obs": 1.0, "theta": 10.0, "state": 5.0}
    )

    def prior(self) -> Any:
        return None

    def simulator(self) -> Any:
        return None

    def observation_operator(self) -> Any:
        return None

    def datasets(self) -> Mapping[str, Any]:
        return self.data


@dataclass
class AbsError:
    name: str = "abs"

    def __call__(self, prediction: Estimate, reference: Any) -> Mapping[str, float]:
        ref = reference.point() if isinstance(reference, FakeEstimate) else reference
        return {"err": abs(prediction.point() - float(ref))}


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #
def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeEstimator(2, 0, 1.0), Estimator)
    assert isinstance(FakeEstimate(1.0), Estimate)
    assert isinstance(FakeTask(), Task)
    assert isinstance(AbsError(), Scorer)


# --------------------------------------------------------------------------- #
# Cube
# --------------------------------------------------------------------------- #
def test_cell_label_and_rung_name() -> None:
    cell = Cell("L4", 2, 1)
    assert cell.label == "L4/r2/c1"
    assert cell.rung_name == "model_based_inference"


@pytest.mark.parametrize(
    ("level", "rung", "complexity"),
    [("L9", 2, 0), ("L4", 7, 0), ("L4", 2, -1)],
)
def test_cell_validation(level: str, rung: int, complexity: int) -> None:
    with pytest.raises(ValueError):
        Cell(level, rung, complexity)


# --------------------------------------------------------------------------- #
# Reference rule (both axes)
# --------------------------------------------------------------------------- #
def test_staged_reference_table() -> None:
    rule = ReferenceRule()
    assert rule.source(1) is ReferenceSource.NONE
    assert rule.source(2) is ReferenceSource.KNOWN_THETA
    assert rule.source(4) is ReferenceSource.ORACLE_POSTERIOR
    assert rule.source(5) is ReferenceSource.ORACLE_POSTERIOR


def test_complexity_axis_step_zero_has_no_simpler_neighbour() -> None:
    rule = ReferenceRule()
    assert rule.source(2, axis=Axis.COMPLEXITY, complexity=0) is ReferenceSource.NONE
    assert (
        rule.source(2, axis=Axis.COMPLEXITY, complexity=1)
        is ReferenceSource.SIMPLER_STEP
    )
    # the amortized head is judged against the oracle even on the complexity axis
    assert (
        rule.source(5, axis=Axis.COMPLEXITY, complexity=1)
        is ReferenceSource.ORACLE_POSTERIOR
    )


# --------------------------------------------------------------------------- #
# End-to-end run
# --------------------------------------------------------------------------- #
def test_run_scores_against_resolved_references() -> None:
    task = FakeTask()
    oracle = FakeEstimator(rung=2, complexity=0, value=9.0)
    amortized = FakeEstimator(rung=5, complexity=0, value=8.0)
    report = Benchmark(
        task=task, estimators=[oracle, amortized], scorers=[AbsError()]
    ).run()

    by_label = {r.cell.label: r for r in report.results}

    # rung 2 → KNOWN_THETA (theta=10) → |9 - 10| = 1
    r2 = by_label["L4/r2/c0"]
    assert r2.reference is ReferenceSource.KNOWN_THETA
    assert r2.scores["abs.err"] == pytest.approx(1.0)

    # rung 5 → ORACLE_POSTERIOR (oracle point=9) → |8 - 9| = 1
    r5 = by_label["L4/r5/c0"]
    assert r5.reference is ReferenceSource.ORACLE_POSTERIOR
    assert r5.scores["abs.err"] == pytest.approx(1.0)

    assert report.to_dict()["L4/r2/c0"]["abs.err"] == pytest.approx(1.0)


def test_generative_rung_is_not_scored() -> None:
    task = FakeTask()
    simple = FakeEstimator(rung=1, complexity=0, value=3.0)
    report = Benchmark(task=task, estimators=[simple], scorers=[AbsError()]).run()
    (result,) = report.results
    assert result.reference is ReferenceSource.NONE
    assert result.scores == {}
    assert "generative" in result.note


def test_complexity_axis_scores_against_simpler_step() -> None:
    task = FakeTask()
    cheap = FakeEstimator(rung=2, complexity=0, value=9.0)
    rich = FakeEstimator(rung=2, complexity=1, value=9.5)
    report = Benchmark(
        task=task,
        estimators=[cheap, rich],
        scorers=[AbsError()],
        axis=Axis.COMPLEXITY,
    ).run()
    by_label = {r.cell.label: r for r in report.results}

    # the simplest step (c0) has no simpler neighbour → not scored
    assert by_label["L4/r2/c0"].reference is ReferenceSource.NONE
    # the richer step (c1) is scored against the simpler step (point=9.0)
    rich_result = by_label["L4/r2/c1"]
    assert rich_result.reference is ReferenceSource.SIMPLER_STEP
    assert rich_result.scores["abs.err"] == pytest.approx(0.5)


def test_missing_reference_is_noted_not_raised() -> None:
    # amortized rung with no rung-2 oracle present → ORACLE_POSTERIOR unresolved
    task = FakeTask()
    amortized = FakeEstimator(rung=5, complexity=0, value=8.0)
    report = Benchmark(task=task, estimators=[amortized], scorers=[AbsError()]).run()
    (result,) = report.results
    assert result.reference is ReferenceSource.ORACLE_POSTERIOR
    assert result.scores == {}
    assert "oracle" in result.note


def test_explicit_reference_estimator_used_as_oracle() -> None:
    task = FakeTask()
    explicit_oracle = FakeEstimator(rung=2, complexity=0, value=7.0)
    amortized = FakeEstimator(rung=5, complexity=0, value=8.0)
    report = Benchmark(
        task=task,
        estimators=[amortized],
        scorers=[AbsError()],
        reference=explicit_oracle,
    ).run()
    (result,) = report.results
    # scored against the explicit oracle's point (7.0): |8 - 7| = 1
    assert result.scores["abs.err"] == pytest.approx(1.0)


def test_duplicate_cells_are_rejected() -> None:
    task = FakeTask()
    a = FakeEstimator(rung=2, complexity=0, value=9.0)
    b = FakeEstimator(rung=2, complexity=0, value=8.0)
    with pytest.raises(ValueError, match="duplicate benchmark cells"):
        Benchmark(task=task, estimators=[a, b], scorers=[AbsError()]).run()


def test_improve_uses_validated_lower_rung_as_reference() -> None:
    task = FakeTask()
    oracle = FakeEstimator(rung=2, complexity=0, value=9.0)
    improve = FakeEstimator(rung=6, complexity=0, value=8.5)
    report = Benchmark(
        task=task, estimators=[oracle, improve], scorers=[AbsError()]
    ).run()
    by_label = {r.cell.label: r for r in report.results}
    r6 = by_label["L4/r6/c0"]
    assert r6.reference is ReferenceSource.BEST_VALIDATED_BELOW
    # rung 2 validated (point=9); |8.5 - 9| = 0.5
    assert r6.scores["abs.err"] == pytest.approx(0.5)


def test_improve_skips_unvalidated_lower_rung() -> None:
    # rung 5 cannot validate (no oracle), so rung 6 must NOT score against it.
    task = FakeTask()
    amortized = FakeEstimator(rung=5, complexity=0, value=8.0)
    improve = FakeEstimator(rung=6, complexity=0, value=7.0)
    report = Benchmark(
        task=task, estimators=[amortized, improve], scorers=[AbsError()]
    ).run()
    by_label = {r.cell.label: r for r in report.results}
    assert by_label["L4/r5/c0"].scores == {}  # unvalidated
    r6 = by_label["L4/r6/c0"]
    assert r6.reference is ReferenceSource.BEST_VALIDATED_BELOW
    assert r6.scores == {}
    assert "no validated rung below" in r6.note
