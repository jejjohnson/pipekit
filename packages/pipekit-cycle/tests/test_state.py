"""Tests for `pipekit_cycle.state` — DAState, IterationState, WindowState."""

from __future__ import annotations

from pipekit_cycle import DAState, IterationState, WindowState


def test_iteration_state_defaults_and_round_trip():
    state = IterationState()
    assert state.iter_count == 0
    assert state.residual_norm == float("inf")
    assert state.extras == {}
    rebuilt = IterationState.from_dict(state.to_dict())
    assert rebuilt == state


def test_iteration_state_extras_copied_not_aliased():
    extras = {"lr": 0.1}
    state = IterationState(extras=extras)
    extras["lr"] = 99
    assert state.extras == {"lr": 0.1}


def test_window_state_defaults_and_round_trip():
    state = WindowState(window_index=2, last_analysis=[1.0, 2.0])
    data = state.to_dict()
    assert data["window_index"] == 2
    rebuilt = WindowState.from_dict(data)
    assert rebuilt == state


def test_da_state_value_semantics():
    a = DAState(t=1.0, cycle_count=2)
    b = DAState(t=1.0, cycle_count=2)
    assert a == b
    assert repr(a).startswith("DAState(")
