"""Shared fixtures for pipekit-cycle tests."""

from __future__ import annotations

from typing import Any

import pytest
from pipekit import StatefulOperator


class AddOneStateful(StatefulOperator):
    """Per-call: carrier += 1, state.count += 1. Canonical test step."""

    def __init__(self) -> None:
        self.calls = 0

    def _apply(self, carrier: float, state: Any) -> tuple[float, Any]:
        self.calls += 1
        if state is not None and hasattr(state, "count"):
            state.count = state.count + 1
        return carrier + 1, state


class SimpleState:
    """Tiny mutable state object — just a counter."""

    def __init__(self, count: int = 0) -> None:
        self.count = count


class ToyForward:
    """Plain forward-model satisfying `ForwardModel` structurally."""

    def __init__(self, dt: float = 1.0, factor: float = 2.0) -> None:
        self._dt = dt
        self.factor = factor

    def step(self, state, dt):
        return state * self.factor

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def state_signature(self):
        return None


class ToyObsOp:
    """Plain obs operator satisfying `ObservationOperator` structurally."""

    def __call__(self, state):
        return state

    def linearize(self, state):
        return self


class ToyAnalysis:
    """Plain analysis step satisfying `AnalysisStep` structurally.

    Returns a weighted average of forecast and obs.
    """

    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight

    def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
        return self.weight * forecast + (1 - self.weight) * obs


@pytest.fixture
def AddOneStateful_cls() -> type[StatefulOperator]:
    return AddOneStateful


@pytest.fixture
def SimpleState_cls() -> type[SimpleState]:
    return SimpleState


@pytest.fixture
def ToyForward_cls() -> type[ToyForward]:
    return ToyForward


@pytest.fixture
def ToyObsOp_cls() -> type[ToyObsOp]:
    return ToyObsOp


@pytest.fixture
def ToyAnalysis_cls() -> type[ToyAnalysis]:
    return ToyAnalysis
