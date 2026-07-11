"""Tests for `pipekit.state` — Group M (StatefulOperator + CarryState)."""

from __future__ import annotations

import pytest
from pipekit import CarryState, Operator, Sequential, StatefulOperator


class MeanState(CarryState):
    """Streaming-mean state, mirroring the module docstring's example."""

    def __init__(self, sum: float = 0.0, n: int = 0) -> None:
        self.sum = sum
        self.n = n


class StreamMean(StatefulOperator):
    """Per-call: fold ``x`` into the running mean, return (mean, new_state)."""

    def _apply(self, x, state):
        new = MeanState(sum=state.sum + x, n=state.n + 1)
        return new.sum / new.n, new


class AddOne(Operator):
    """Stateless step for mixed-pipeline tests."""

    def _apply(self, x):
        return x + 1


# ---------------------------------------------------------------------
# CarryState
# ---------------------------------------------------------------------


def test_carry_state_to_dict_from_dict_round_trip():
    state = MeanState(sum=3.5, n=2)
    data = state.to_dict()
    assert data == {"sum": 3.5, "n": 2}
    rebuilt = MeanState.from_dict(data)
    assert rebuilt == state


def test_carry_state_repr_lists_fields():
    assert repr(MeanState(sum=1.0, n=4)) == "MeanState(sum=1.0, n=4)"


def test_carry_state_eq_by_value_and_type():
    assert MeanState(sum=1.0, n=1) == MeanState(sum=1.0, n=1)
    assert MeanState(sum=1.0, n=1) != MeanState(sum=2.0, n=1)

    class OtherState(CarryState):
        def __init__(self) -> None:
            self.sum = 1.0
            self.n = 1

    # Same fields, different type — not equal.
    assert MeanState(sum=1.0, n=1) != OtherState()


def test_carry_state_hash_consistent_with_eq():
    a = MeanState(sum=1.0, n=1)
    b = MeanState(sum=1.0, n=1)
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


# ---------------------------------------------------------------------
# StatefulOperator
# ---------------------------------------------------------------------


def test_stateful_operator_is_flagged():
    assert StreamMean._is_stateful is True
    assert AddOne._is_stateful is False


def test_stateful_operator_apply_not_implemented():
    class Bare(StatefulOperator):
        pass

    with pytest.raises(NotImplementedError, match="Bare"):
        Bare()(1.0, MeanState())


def test_initial_state_fn_bootstraps_state():
    op = StreamMean()
    op.initial_state_fn = lambda: MeanState(sum=10.0, n=1)
    mean, state = op(20.0, op.initial_state_fn())
    assert mean == 15.0
    assert state == MeanState(sum=30.0, n=2)


# ---------------------------------------------------------------------
# Sequential state threading — the success paths
# ---------------------------------------------------------------------


def test_sequential_threads_state_through_stateful_ops():
    pipe = Sequential([StreamMean(), StreamMean()])
    carrier, state = pipe(4.0, MeanState())
    # First op: mean(4) = 4, state (4, 1); second folds 4 again: (8, 2) → 4.
    assert carrier == 4.0
    assert state == MeanState(sum=8.0, n=2)


def test_sequential_mixed_stateless_ops_pass_state_through():
    pipe = Sequential([AddOne(), StreamMean(), AddOne()])
    carrier, state = pipe(1.0, MeanState())
    # AddOne: 2.0 → StreamMean: mean 2.0, state (2, 1) → AddOne: 3.0.
    assert carrier == 3.0
    assert state == MeanState(sum=2.0, n=1)


def test_sequential_explicit_state_with_only_stateless_ops():
    """Supplying state forces the stateful path; state passes through."""
    pipe = Sequential([AddOne()])
    carrier, state = pipe(1.0, state=MeanState(sum=9.0, n=3))
    assert carrier == 2.0
    assert state == MeanState(sum=9.0, n=3)


def test_sequential_stateful_without_state_raises():
    pipe = Sequential([StreamMean()])
    with pytest.raises(TypeError, match="without a `state`"):
        pipe(1.0)


def test_empty_sequential_is_not_stateful():
    assert Sequential([])._is_stateful is False
