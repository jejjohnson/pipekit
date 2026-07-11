"""Tests for `pipekit_cycle.bfn.BFNCycle`.

The toy setup is an identity (time-reversible) forward model with
`IdentityObs`, so a corrective nudge ``x <- x + K(y - x)`` pulls the state
toward the observation each pass; back-and-forth iteration converges to the
observation. Most tests inject a numpy-free ``convergence_fn`` so the suite
runs without numpy; one exercises the default (numpy) metric.
"""

from __future__ import annotations

from typing import Any

import pytest
from pipekit import Operator
from pipekit_cycle import BFNCycle, CallableForward, DAState, IdentityObs


def _identity_model(dt: float = 1.0) -> CallableForward:
    return CallableForward(lambda x, _dt: x, dt=dt)


def _gain(k: float = 0.5):
    return lambda innov: k * innov


def _abs_change(new: Any, old: Any) -> float:
    return abs(new - old) / (abs(old) + 1e-12)


class _ConstObs(Operator):
    __config_mixin_auto__ = False

    def __init__(self, value: float) -> None:
        self.value = value

    def _apply(self, step_index: int) -> float:
        return self.value


def test_bfn_converges_toward_obs() -> None:
    bfn = BFNCycle(
        _identity_model(),
        IdentityObs(),
        _gain(0.5),
        window=1,
        max_iters=40,
        tol=1e-4,
        obs_source=_ConstObs(10.0),
        convergence_fn=_abs_change,
    )
    result, _ = bfn(0.0, DAState())
    assert abs(result - 10.0) < 0.1
    assert bfn.iterations >= 1


def test_bfn_early_stop_when_already_converged() -> None:
    bfn = BFNCycle(
        _identity_model(),
        IdentityObs(),
        _gain(0.5),
        window=1,
        max_iters=10,
        tol=1e-6,
        obs_source=_ConstObs(10.0),
        convergence_fn=_abs_change,
    )
    # start at the observation -> the nudge is zero -> converges in one sweep
    result, _ = bfn(10.0, DAState())
    assert bfn.iterations == 1
    assert abs(result - 10.0) < 1e-9


def test_bfn_hits_max_iter_when_not_converged() -> None:
    bfn = BFNCycle(
        _identity_model(),
        IdentityObs(),
        _gain(0.5),
        window=1,
        max_iters=3,
        tol=1e-12,
        obs_source=_ConstObs(10.0),
        convergence_fn=_abs_change,
    )
    bfn(0.0, DAState())
    assert bfn.iterations == 3


def test_bfn_without_obs_is_pure_integration() -> None:
    # no obs_source -> no nudging -> identity model leaves the carrier unchanged
    bfn = BFNCycle(
        _identity_model(),
        IdentityObs(),
        _gain(0.5),
        window=2,
        obs_source=None,
        convergence_fn=_abs_change,
    )
    result, _ = bfn(3.0, DAState())
    assert result == 3.0
    assert bfn.iterations == 1  # back-pass equals forward-pass -> immediate stop


def test_bfn_uses_backward_model_when_supplied() -> None:
    bfn = BFNCycle(
        _identity_model(),
        IdentityObs(),
        _gain(0.5),
        window=1,
        max_iters=20,
        tol=1e-4,
        obs_source=_ConstObs(5.0),
        backward_model=_identity_model(),
        convergence_fn=_abs_change,
    )
    result, _ = bfn(0.0, DAState())
    assert abs(result - 5.0) < 0.1


def test_bfn_advances_carry_state() -> None:
    # the threaded clock / cycle count advance over the window, like DACycle
    bfn = BFNCycle(
        _identity_model(dt=2.0),
        IdentityObs(),
        _gain(0.5),
        window=3,
        max_iters=5,
        obs_source=_ConstObs(10.0),
        convergence_fn=_abs_change,
    )
    _, state = bfn(0.0, DAState(t=0.0))
    assert state.cycle_count == 3
    assert state.t == 6.0  # 3 window steps * dt=2.0


def test_bfn_default_convergence_is_dependency_free() -> None:
    # no convergence_fn -> the default relative-L2 metric, which must work for
    # scalar carriers without importing numpy
    bfn = BFNCycle(
        _identity_model(),
        IdentityObs(),
        _gain(0.5),
        window=1,
        max_iters=40,
        tol=1e-4,
        obs_source=_ConstObs(10.0),
    )
    result, _ = bfn(0.0, DAState())
    assert abs(result - 10.0) < 0.1


def test_bfn_rejects_dt_ignoring_model_without_backward_model() -> None:
    from pipekit_cycle import CompositeForward

    comp = CompositeForward((_identity_model(),))  # CompositeForward ignores dt
    with pytest.raises(TypeError, match="backward_model"):
        BFNCycle(comp, IdentityObs(), _gain(), window=1)
    # accepted once an explicit backward_model is supplied
    bfn = BFNCycle(
        comp,
        IdentityObs(),
        _gain(0.5),
        window=1,
        max_iters=20,
        tol=1e-4,
        obs_source=_ConstObs(5.0),
        backward_model=_identity_model(),
        convergence_fn=_abs_change,
    )
    result, _ = bfn(0.0, DAState())
    assert abs(result - 5.0) < 0.1


def test_bfn_validation() -> None:
    with pytest.raises(ValueError, match="window"):
        BFNCycle(_identity_model(), IdentityObs(), _gain(), window=0)
    with pytest.raises(ValueError, match="max_iter"):
        BFNCycle(_identity_model(), IdentityObs(), _gain(), window=1, max_iters=0)
    with pytest.raises(TypeError, match="ForwardModel"):
        BFNCycle(object(), IdentityObs(), _gain(), window=1)
    with pytest.raises(TypeError, match="ObservationOperator"):
        BFNCycle(_identity_model(), object(), _gain(), window=1)
    with pytest.raises(TypeError, match="callable"):
        BFNCycle(_identity_model(), IdentityObs(), 42, window=1)
