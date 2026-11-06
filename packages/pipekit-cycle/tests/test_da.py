"""Tests for `pipekit_cycle.da`."""

from __future__ import annotations

import pytest
from pipekit_cycle import (
    CallableForward,
    DACycle,
    DAState,
    EnsembleDACycle,
    IdentityObs,
    SmootherCycle,
)
from pipekit_ops import Lambda, Operator


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


# --- obs_clock: a resumed cycle reads the next observations ----------------


class _RecordingObs(Operator):
    __config_mixin_auto__ = False

    def __init__(self, values=None) -> None:
        self.values = values
        self.seen: list[int] = []

    def _apply(self, index):
        self.seen.append(index)
        return float(index) if self.values is None else self.values[index]

    def get_config(self):
        return {}


class _TakeObs:
    def __call__(self, forecast, obs, *, obs_op=None, obs_err_cov=None):
        return obs


def _identity_forward() -> CallableForward:
    return CallableForward(lambda x, dt: x, dt=1.0)


# Scalar Kalman filter on a carrier (mean m, variance p):
# forecast: m ← a m, p ← a² p + q
# analysis: g = p / (p + r), m ← m + g (y - m), p ← (1 - g) p
_A, _Q, _R = 0.9, 0.1, 0.5


class _ScalarKalman:
    def __call__(self, forecast, obs, *, obs_op=None, obs_err_cov=None):
        mean, variance = forecast
        gain = variance / (variance + _R)
        return mean + gain * (obs - mean), (1.0 - gain) * variance


def _kalman_cycle(obs_source: Operator, n_steps: int, obs_clock: str) -> DACycle:
    forward = CallableForward(
        lambda carrier, dt: (_A * carrier[0], _A**2 * carrier[1] + _Q), dt=1.0
    )
    return DACycle(
        forward,
        IdentityObs(),
        _ScalarKalman(),
        obs_source=obs_source,
        n_steps=n_steps,
        save_history=False,
        obs_clock=obs_clock,
    )


def test_two_resumed_kalman_calls_equal_one_continuous_run():
    obs = [1.0, 0.5, -0.2, 0.3, 1.1, 0.9, -0.4, 0.2, 0.7, 0.1]
    start = (0.0, 1.0)

    continuous, _ = _kalman_cycle(_RecordingObs(obs), 10, "state")(start, DAState())

    source = _RecordingObs(obs)
    half = _kalman_cycle(source, 5, "state")
    resumed, state = half(start, DAState())
    resumed, state = half(resumed, state)

    assert resumed == continuous
    assert source.seen == list(range(10))
    assert state.cycle_count == 10


def test_default_clock_still_restarts_on_every_call():
    source = _RecordingObs()
    cycle = DACycle(
        _identity_forward(),
        IdentityObs(),
        _TakeObs(),
        obs_source=source,
        n_steps=5,
        save_history=False,
    )
    x, state = cycle(0.0, DAState())
    cycle(x, state)
    assert source.seen == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert "obs_clock" not in cycle.get_config()


def test_ensemble_cycle_state_clock_continues():
    source = _RecordingObs()

    class TakeObsPerMember:
        def __call__(self, forecasts, obs, *, obs_op=None, obs_err_cov=None):
            return [obs for _ in forecasts]

    cycle = EnsembleDACycle(
        _identity_forward(),
        IdentityObs(),
        TakeObsPerMember(),
        obs_source=source,
        n_steps=2,
        n_members=2,
        obs_clock="state",
    )
    members, state = cycle([0.0, 0.0], DAState())
    cycle(members, state)
    assert source.seen == [0, 1, 2, 3]
    assert cycle.get_config()["obs_clock"] == "state"


def test_smoother_cycle_state_clock_continues():
    source = _RecordingObs()

    class LastForecast:
        def __call__(self, forecasts, obs, *, obs_op=None, obs_err_cov=None):
            return forecasts[-1]

    cycle = SmootherCycle(
        _identity_forward(),
        IdentityObs(),
        LastForecast(),
        window=2,
        stride=2,
        obs_source=source,
        obs_clock="state",
    )
    x, state = cycle(0.0, DAState())
    cycle(x, state)
    assert source.seen == list(range(8))
    assert state.cycle_count == 4


def test_state_clock_needs_a_cycle_count():
    cycle = DACycle(
        _identity_forward(),
        IdentityObs(),
        _TakeObs(),
        obs_source=_RecordingObs(),
        obs_clock="state",
    )
    with pytest.raises(TypeError, match="cycle_count"):
        cycle(0.0, None)


def test_unknown_obs_clock_is_rejected():
    with pytest.raises(ValueError, match="obs_clock"):
        DACycle(_identity_forward(), IdentityObs(), _TakeObs(), obs_clock="wall")


# --- obs_err_source: a per-step observation-error covariance ---------------


class _ScalarKalmanFromState:
    """The same filter, reading its observation error from ``obs_err_cov``."""

    def __call__(self, forecast, obs, *, obs_op=None, obs_err_cov=None):
        mean, variance = forecast
        gain = variance / (variance + obs_err_cov)
        return mean + gain * (obs - mean), (1.0 - gain) * variance


def _r_source() -> Lambda:
    # Σ_ε,t = 0.1 + 0.01 t
    return Lambda(lambda t: 0.1 + 0.01 * t)


def _time_varying_cycle(obs, n_steps: int, obs_clock: str = "call") -> DACycle:
    forward = CallableForward(
        lambda carrier, dt: (_A * carrier[0], _A**2 * carrier[1] + _Q), dt=1.0
    )
    return DACycle(
        forward,
        IdentityObs(),
        _ScalarKalmanFromState(),
        obs_source=_RecordingObs(obs),
        n_steps=n_steps,
        save_history=False,
        obs_clock=obs_clock,
        obs_err_source=_r_source(),
    )


def test_obs_err_source_equals_a_hand_written_kalman_filter():
    obs = [1.0, 0.5, -0.2, 0.3, 1.1]
    mean, variance = 0.0, 1.0
    for t, y in enumerate(obs):
        mean, variance = _A * mean, _A**2 * variance + _Q
        gain = variance / (variance + (0.1 + 0.01 * t))
        mean, variance = mean + gain * (y - mean), (1.0 - gain) * variance

    carrier, state = _time_varying_cycle(obs, 5)((0.0, 1.0), DAState())

    assert carrier == (mean, variance)
    assert state.obs_err_cov == 0.1 + 0.01 * 4


def test_obs_err_source_follows_the_state_clock():
    obs = [1.0, 0.5, -0.2, 0.3, 1.1, 0.9]
    continuous, _ = _time_varying_cycle(obs, 6, "state")((0.0, 1.0), DAState())

    half = _time_varying_cycle(obs, 3, "state")
    resumed, state = half((0.0, 1.0), DAState())
    resumed, _ = half(resumed, state)

    assert resumed == continuous


def test_obs_err_source_leaves_the_callers_state_alone():
    start = DAState(obs_err_cov="untouched")
    _time_varying_cycle([1.0], 1)((0.0, 1.0), start)
    assert start.obs_err_cov == "untouched"
    assert start.cycle_count == 0


def test_without_obs_err_source_the_state_covariance_is_used():
    seen = []

    class Record:
        def __call__(self, forecast, obs, *, obs_op=None, obs_err_cov=None):
            seen.append(obs_err_cov)
            return obs

    cycle = DACycle(
        _identity_forward(),
        IdentityObs(),
        Record(),
        obs_source=_RecordingObs(),
        n_steps=2,
        save_history=False,
    )
    cycle(0.0, DAState(obs_err_cov=0.5))

    assert seen == [0.5, 0.5]
    assert "obs_err_source" not in cycle.get_config()


def test_ensemble_cycle_passes_each_steps_covariance():
    seen = []

    class Record:
        def __call__(self, forecasts, obs, *, obs_op=None, obs_err_cov=None):
            seen.append(obs_err_cov)
            return forecasts

    cycle = EnsembleDACycle(
        _identity_forward(),
        IdentityObs(),
        Record(),
        obs_source=_RecordingObs(),
        n_steps=2,
        n_members=1,
        obs_err_source=Lambda(lambda t: float(t)),
    )
    cycle([0.0], DAState())

    assert seen == [0.0, 1.0]
    assert "obs_err_source" in cycle.get_config()


def test_smoother_cycle_passes_the_windows_covariances_as_a_list():
    seen = []

    class Record:
        def __call__(self, forecasts, obs, *, obs_op=None, obs_err_cov=None):
            seen.append(obs_err_cov)
            return forecasts[-1]

    SmootherCycle(
        _identity_forward(),
        IdentityObs(),
        Record(),
        window=2,
        stride=2,
        obs_source=_RecordingObs(),
        obs_err_source=Lambda(lambda t: float(t)),
    )(0.0, DAState())

    assert seen == [[0.0, 1.0], [2.0, 3.0]]


def test_obs_err_source_needs_an_obs_source():
    with pytest.raises(ValueError, match="needs an obs_source"):
        DACycle(
            _identity_forward(),
            IdentityObs(),
            _TakeObs(),
            obs_err_source=_r_source(),
        )


def test_obs_err_source_must_be_an_operator():
    with pytest.raises(TypeError, match="must be an Operator"):
        DACycle(
            _identity_forward(),
            IdentityObs(),
            _TakeObs(),
            obs_source=_RecordingObs(),
            obs_err_source=lambda t: t,
        )


def test_obs_err_source_needs_a_state():
    cycle = DACycle(
        _identity_forward(),
        IdentityObs(),
        _TakeObs(),
        obs_source=_RecordingObs(),
        obs_err_source=_r_source(),
    )
    with pytest.raises(TypeError, match="needs a state"):
        cycle(0.0, None)
