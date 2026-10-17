"""Gate A1, config round-trip: every shipped pipekit-cycle operator survives it.

For every operator in ``pipekit_cycle.__all__`` not marked
``forbid_in_yaml``, ``loads(dumps(op))`` has the same `pipeline_hash`
and returns the same output on a fixed input (`assert_round_trip`).
A new operator without a case in `CASES` fails
`test_every_operator_has_a_case`.

The DA cycles round-trip only when their components are themselves
operators, so the cases build them from the small operators below,
each satisfying its `pipekit_cycle` protocol structurally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pipekit_cycle
import pytest
from pipekit import Operator, StatefulOperator
from pipekit.testing import assert_round_trip, public_operators
from pipekit_cycle import (
    CompositeForward,
    CompositeObs,
    Cycle,
    DACycle,
    DAState,
    EnsembleCycle,
    EnsembleDACycle,
    IdentityObs,
    IterationState,
    NeuralForward,
    Recurrence,
    SmootherCycle,
    WindowedCycle,
)


class _Grow(StatefulOperator):
    """Step: carrier * rate; count the iteration in a fresh state."""

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def _apply(self, carrier: float, state: IterationState) -> tuple[Any, Any]:
        return carrier * self.rate, IterationState(iter_count=state.iter_count + 1)


class _Reached(Operator):
    """Condition: the iteration count has reached ``limit``."""

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def _apply(self, state: IterationState) -> bool:
        return state.iter_count >= self.limit


class _Decay(Operator):
    """A `ForwardModel`: state * factor every ``dt``."""

    state_signature = None

    def __init__(self, factor: float, dt: float = 1.0) -> None:
        self.factor = factor
        self.dt = dt

    def step(self, state: float, dt: float) -> float:
        return state * self.factor

    def _apply(self, state: float) -> float:
        return self.step(state, self.dt)


class _Halve(Operator):
    """An `ObservationOperator`: halve the state; linear, so its own tangent."""

    def _apply(self, state: float) -> float:
        return state / 2

    def linearize(self, state: float) -> _Halve:
        return self


class _ObsAt(Operator):
    """Observation source: ``scale * (step index + 1)``."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def _apply(self, i: int) -> float:
        return self.scale * (i + 1)


class _Blend(Operator):
    """An `AnalysisStep`: ``w * forecast + (1 - w) * obs``."""

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def _apply(self, forecast: float, obs: float, *, obs_op: Any, obs_err_cov: Any):
        return self.weight * forecast + (1 - self.weight) * obs


class _BlendMembers(Operator):
    """An ensemble `AnalysisStep`: blend each member with the obs."""

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def _apply(self, forecasts: list[float], obs: float, *, obs_op, obs_err_cov):
        return [self.weight * f + (1 - self.weight) * obs for f in forecasts]


class _BlendWindow(Operator):
    """A smoother `AnalysisStep`: blend the window's last forecast and mean obs."""

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def _apply(self, forecasts: list[float], obs: list[float], *, obs_op, obs_err_cov):
        return self.weight * forecasts[-1] + (1 - self.weight) * sum(obs) / len(obs)


#: Abstract bases: none ship in pipekit-cycle.
SKIP: dict[str, str] = {}

#: name -> (build the operator, positional inputs)
CASES: dict[str, tuple[Callable[[], Operator], tuple[Any, ...]]] = {
    "CompositeForward": (
        lambda: CompositeForward((_Decay(0.5, dt=1.0), _Decay(3.0, dt=2.0))),
        (8.0,),
    ),
    "CompositeObs": (lambda: CompositeObs((_Halve(), IdentityObs())), (8.0,)),
    "Cycle": (
        lambda: Cycle(_Grow(2.0), n_steps=3, save_history=True, history_stride=2),
        (1.0, IterationState()),
    ),
    "DACycle": (
        lambda: DACycle(
            _Decay(0.9), _Halve(), _Blend(0.25), obs_source=_ObsAt(2.0), n_steps=3
        ),
        (10.0, DAState(obs_err_cov=0.1)),
    ),
    "EnsembleCycle": (
        lambda: EnsembleCycle(_Grow(2.0), n_steps=2, n_members=3),
        ([1.0, 2.0, 3.0], IterationState()),
    ),
    "EnsembleDACycle": (
        lambda: EnsembleDACycle(
            _Decay(0.9),
            _Halve(),
            _BlendMembers(0.5),
            obs_source=_ObsAt(1.0),
            n_steps=2,
            n_members=2,
        ),
        ([1.0, 4.0], DAState()),
    ),
    "IdentityObs": (IdentityObs, (8.0,)),
    "NeuralForward": (lambda: NeuralForward(_Decay(0.5), dt=600.0), (8.0,)),
    "Recurrence": (
        lambda: Recurrence(_Grow(3.0), _Reached(4), max_iters=10),
        (1.0, IterationState()),
    ),
    "SmootherCycle": (
        lambda: SmootherCycle(
            _Decay(0.9),
            _Halve(),
            _BlendWindow(0.5),
            window=3,
            stride=2,
            obs_source=_ObsAt(1.0),
        ),
        (10.0, DAState()),
    ),
    "WindowedCycle": (
        lambda: WindowedCycle(_Grow(2.0), window=2, stride=2),
        ([1.0, 2.0, 3.0, 4.0, 5.0], IterationState()),
    ),
}

OPERATORS = public_operators(pipekit_cycle)
ROUND_TRIPPABLE = sorted(n for n, cls in OPERATORS.items() if not cls.forbid_in_yaml)


def test_every_operator_has_a_case():
    missing = [n for n in ROUND_TRIPPABLE if n not in CASES and n not in SKIP]
    assert not missing, (
        f"Gate A1: {missing} have no round-trip case. Add a constructor and an "
        "input to CASES in this file (or, for an abstract base only, to SKIP)."
    )


def test_no_stale_cases():
    stale = sorted(set(CASES) - set(ROUND_TRIPPABLE))
    assert not stale, f"CASES names operators that are not shipped: {stale}"


@pytest.mark.parametrize("name", ROUND_TRIPPABLE)
def test_operator_round_trips(name):
    if name in SKIP:
        pytest.skip(SKIP[name])
    make, args = CASES[name]
    assert_round_trip(make(), *args)


def test_forward_models_keep_their_dt():
    """`dt` is derived (CompositeForward) or configured (NeuralForward)."""
    composite = assert_round_trip(CASES["CompositeForward"][0](), 8.0)
    assert composite.dt == 3.0
    neural = assert_round_trip(CASES["NeuralForward"][0](), 8.0)
    assert neural.dt == 600.0
