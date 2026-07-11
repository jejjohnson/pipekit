"""Tests for `pipekit_cycle.cycle`."""

from __future__ import annotations

import pytest
from pipekit_cycle import (
    Cycle,
    EnsembleCycle,
    Recurrence,
    WindowedCycle,
)


def test_cycle_advances_n_steps(AddOneStateful_cls, SimpleState_cls):
    cycle = Cycle(AddOneStateful_cls(), n_steps=5)
    carrier, state = cycle(0, SimpleState_cls())
    assert carrier == 5
    assert state.count == 5


def test_cycle_zero_steps_is_identity(AddOneStateful_cls, SimpleState_cls):
    cycle = Cycle(AddOneStateful_cls(), n_steps=0)
    carrier, state_out = cycle(7, SimpleState_cls(count=3))
    assert carrier == 7
    assert state_out.count == 3


def test_cycle_save_history(AddOneStateful_cls, SimpleState_cls):
    cycle = Cycle(AddOneStateful_cls(), n_steps=4, save_history=True)
    cycle(0, SimpleState_cls())
    assert len(cycle.history) == 4
    # carriers at each saved step
    assert [c for c, _ in cycle.history] == [1, 2, 3, 4]


def test_cycle_history_stride(AddOneStateful_cls, SimpleState_cls):
    cycle = Cycle(AddOneStateful_cls(), n_steps=6, save_history=True, history_stride=2)
    cycle(0, SimpleState_cls())
    # steps 0, 2, 4 are saved; carriers seen at those moments
    assert len(cycle.history) == 3


def test_cycle_wraps_forward_model(ToyForward_cls, SimpleState_cls):
    """Cycle accepts a ForwardModel directly via the adapter shim."""
    cycle = Cycle(ToyForward_cls(dt=1.0, factor=3.0), n_steps=3)
    carrier, _ = cycle(2.0, SimpleState_cls())
    # 2 * 3 * 3 * 3 = 54
    assert carrier == 54.0


def test_cycle_rejects_invalid_step_op():
    with pytest.raises(TypeError):
        Cycle(step_op=object(), n_steps=1)


def test_cycle_rejects_negative_n_steps(AddOneStateful_cls):
    with pytest.raises(ValueError):
        Cycle(AddOneStateful_cls(), n_steps=-1)


def test_ensemble_cycle_runs_each_member(AddOneStateful_cls, SimpleState_cls):
    ens = EnsembleCycle(AddOneStateful_cls(), n_steps=3, n_members=4)
    members, state = ens([0, 10, 20, 30], SimpleState_cls())
    assert members == [3, 13, 23, 33]
    # each member contributed to state.count
    assert state.count == 3 * 4


def test_ensemble_cycle_validates_member_count(AddOneStateful_cls, SimpleState_cls):
    ens = EnsembleCycle(AddOneStateful_cls(), n_steps=1, n_members=3)
    with pytest.raises(ValueError, match="3 members"):
        ens([1, 2], SimpleState_cls())


def test_windowed_cycle_emits_one_output_per_window(
    AddOneStateful_cls, SimpleState_cls
):
    win = WindowedCycle(AddOneStateful_cls(), window=2, stride=2)
    out, state = win([10, 20, 30, 40, 50, 60], SimpleState_cls())
    # windows at start=0 ([10,20]) and start=2 ([30,40]) and start=4 ([50,60])
    # in each window: step applied twice to stream[start] / stream[start+1]
    # carrier starts at stream[start], then += 1 twice
    assert out == [12, 32, 52]
    assert state.count == 6


def test_windowed_cycle_stride_skip(AddOneStateful_cls, SimpleState_cls):
    win = WindowedCycle(AddOneStateful_cls(), window=2, stride=3)
    out, _ = win([10, 20, 30, 40, 50, 60], SimpleState_cls())
    # windows at start=0, start=3
    assert len(out) == 2


def test_recurrence_stops_on_condition(AddOneStateful_cls, SimpleState_cls):
    from pipekit import Lambda

    # Stop once state.count >= 5.
    stop_cond = Lambda(lambda s: s.count >= 5)
    rec = Recurrence(AddOneStateful_cls(), condition_op=stop_cond, max_iters=100)
    _, state_out = rec(0, SimpleState_cls())
    assert state_out.count == 5
    assert rec.converged is True
    assert rec.iters == 5


def test_recurrence_hits_max_iters(AddOneStateful_cls, SimpleState_cls):
    from pipekit import Lambda

    never_stop = Lambda(lambda s: False)
    rec = Recurrence(AddOneStateful_cls(), condition_op=never_stop, max_iters=3)
    rec(0, SimpleState_cls())
    assert rec.converged is False
    assert rec.iters == 3


def test_windowed_cycle_short_stream_raises(AddOneStateful_cls):
    """A stream shorter than one window is a misconfiguration, not a no-op."""
    cyc = WindowedCycle(step_op=AddOneStateful_cls(), window=5)
    with pytest.raises(ValueError, match="shorter than one window"):
        cyc([1.0, 2.0], None)
