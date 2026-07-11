"""Tests for `pipekit_cycle.da`."""

from __future__ import annotations

import pytest
from pipekit import Operator
from pipekit_cycle import (
    DACycle,
    DAState,
    EnsembleDACycle,
    IdentityObs,
    SmootherCycle,
)


class FixedObsSource(Operator):
    """Operator that yields a fixed obs per step index."""

    __config_mixin_auto__ = False

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def _apply(self, step_index: int) -> float:
        return self.values[step_index]


def test_dacycle_runs_one_cycle_with_obs(ToyForward_cls, ToyAnalysis_cls):
    fwd = ToyForward_cls(dt=1.0, factor=2.0)
    da = DACycle(
        forward_model=fwd,
        obs_op=IdentityObs(),
        analysis_step=ToyAnalysis_cls(weight=0.5),
        obs_source=FixedObsSource([10.0]),
        n_steps=1,
    )
    state = DAState(t=0.0, obs_err_cov=None)
    carrier, state = da(2.0, state)
    # forecast: 2 * 2 = 4; analysis: 0.5 * 4 + 0.5 * 10 = 7
    assert carrier == 7.0
    assert state.t == 1.0
    assert state.cycle_count == 1


def test_dacycle_no_obs_source_propagates_forecast(ToyForward_cls, ToyAnalysis_cls):
    fwd = ToyForward_cls(dt=1.0, factor=3.0)
    da = DACycle(
        forward_model=fwd,
        obs_op=IdentityObs(),
        analysis_step=ToyAnalysis_cls(),
        obs_source=None,
        n_steps=3,
    )
    state = DAState()
    carrier, state = da(1.0, state)
    # 1 * 3 * 3 * 3 = 27 (no analysis)
    assert carrier == 27.0
    assert state.cycle_count == 3


def test_dacycle_history(ToyForward_cls, ToyAnalysis_cls):
    fwd = ToyForward_cls(dt=1.0, factor=2.0)
    da = DACycle(
        forward_model=fwd,
        obs_op=IdentityObs(),
        analysis_step=ToyAnalysis_cls(),
        obs_source=FixedObsSource([10.0, 20.0]),
        n_steps=2,
        save_history=True,
    )
    da(1.0, DAState())
    assert len(da.history) == 2
    # each entry is (forecast, analysis)
    for entry in da.history:
        assert len(entry) == 2


def test_dacycle_rejects_bad_forward_model(ToyAnalysis_cls):
    with pytest.raises(TypeError, match="ForwardModel"):
        DACycle(
            forward_model=object(),  # type: ignore[arg-type]
            obs_op=IdentityObs(),
            analysis_step=ToyAnalysis_cls(),
            obs_source=None,
            n_steps=1,
        )


def test_dacycle_rejects_bad_obs_source(ToyForward_cls, ToyAnalysis_cls):
    with pytest.raises(TypeError, match="obs_source"):
        DACycle(
            forward_model=ToyForward_cls(),
            obs_op=IdentityObs(),
            analysis_step=ToyAnalysis_cls(),
            obs_source=lambda i: 0.0,  # type: ignore[arg-type]
            n_steps=1,
        )


def test_ensemble_dacycle_runs(ToyForward_cls):
    fwd = ToyForward_cls(dt=1.0, factor=2.0)

    class MeanAnalysis:
        """Per-member: average with the obs."""

        def __call__(self, forecasts, obs, *, obs_op, obs_err_cov):
            return [0.5 * f + 0.5 * obs for f in forecasts]

    da = EnsembleDACycle(
        forward_model=fwd,
        obs_op=IdentityObs(),
        analysis_step=MeanAnalysis(),
        obs_source=FixedObsSource([10.0]),
        n_steps=1,
        n_members=3,
    )
    members, state = da([1.0, 2.0, 3.0], DAState())
    # forecasts: [2, 4, 6]; analysis: [(2+10)/2, (4+10)/2, (6+10)/2] = [6, 7, 8]
    assert members == [6.0, 7.0, 8.0]
    assert state.cycle_count == 1


def test_ensemble_dacycle_validates_member_count(ToyForward_cls, ToyAnalysis_cls):
    da = EnsembleDACycle(
        forward_model=ToyForward_cls(),
        obs_op=IdentityObs(),
        analysis_step=ToyAnalysis_cls(),
        obs_source=None,
        n_steps=1,
        n_members=3,
    )
    with pytest.raises(ValueError, match="3 members"):
        da([1.0, 2.0], DAState())


def test_smoother_cycle_accumulates_window(ToyForward_cls):
    fwd = ToyForward_cls(dt=1.0, factor=2.0)

    class WindowedAnalysis:
        def __call__(self, forecasts, obs, *, obs_op, obs_err_cov):
            # Return the last forecast nudged by the mean obs.
            return forecasts[-1] + sum(obs) / len(obs)

    da = SmootherCycle(
        forward_model=fwd,
        obs_op=IdentityObs(),
        analysis_step=WindowedAnalysis(),
        window=2,
        stride=1,
        obs_source=FixedObsSource([10.0, 20.0]),
    )
    out, state = da(1.0, DAState())
    # window: forecasts [2, 4]; obs [10, 20] → 4 + 15 = 19
    assert out == 19.0
    assert state.cycle_count == 2


def test_smoother_cycle_without_obs(ToyForward_cls, ToyAnalysis_cls):
    fwd = ToyForward_cls(dt=1.0, factor=2.0)
    da = SmootherCycle(
        forward_model=fwd,
        obs_op=IdentityObs(),
        analysis_step=ToyAnalysis_cls(),
        window=3,
        stride=1,
        obs_source=None,
    )
    out, _ = da(1.0, DAState())
    # 1 * 2 * 2 * 2 = 8, no analysis
    assert out == 8.0


def test_dacycle_uses_state_obs_err_cov(ToyForward_cls):
    captured = {}

    class CapturingAnalysis:
        def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
            captured["err"] = obs_err_cov
            return forecast

    da = DACycle(
        forward_model=ToyForward_cls(),
        obs_op=IdentityObs(),
        analysis_step=CapturingAnalysis(),
        obs_source=FixedObsSource([0.0]),
        n_steps=1,
    )
    da(1.0, DAState(obs_err_cov="MY_R"))
    assert captured["err"] == "MY_R"


def test_dacycle_does_not_mutate_input_state(ToyForward_cls, ToyAnalysis_cls):
    """CarryState is a value object — snapshots held by callers must survive."""
    da = DACycle(
        forward_model=ToyForward_cls(),
        obs_op=IdentityObs(),
        analysis_step=ToyAnalysis_cls(),
        obs_source=FixedObsSource([0.0, 0.0, 0.0]),
        n_steps=3,
    )
    before = DAState(t=0.0, cycle_count=0)
    _, after = da(1.0, before)
    assert (before.t, before.cycle_count) == (0.0, 0)
    assert after.cycle_count == 3
    assert after is not before
