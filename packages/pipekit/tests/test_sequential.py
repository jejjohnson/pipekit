"""Tests for `pipekit._base.sequential` — Sequential."""

from __future__ import annotations

import pytest
from pipekit import Operator, Sequential


def test_basic_left_to_right(Scale_cls, AddConst_cls):
    pipe = Sequential([Scale_cls(2.0), AddConst_cls(1.0)])
    assert pipe(3.0) == 7.0


def test_empty_sequential_requires_no_input():
    with pytest.raises(TypeError, match="empty pipeline"):
        Sequential([])()


def test_empty_sequential_is_identity_when_input_given():
    """Sequential([]) with an input returns the input unchanged."""
    pipe = Sequential([])
    assert pipe(42) == 42


def test_construction_rejects_non_operator():
    with pytest.raises(
        TypeError, match=r"Sequential.operators\[1\] must be an Operator"
    ):
        Sequential([_Identity(), 5])  # type: ignore[list-item]


def test_construction_rejects_terminal_in_non_terminal_position(Sink_cls, Scale_cls):
    with pytest.raises(TypeError, match="terminal operator"):
        Sequential([Sink_cls("first"), Scale_cls(2.0)])


def test_terminal_at_end_is_allowed(Sink_cls, Scale_cls):
    pipe = Sequential([Scale_cls(2.0), Sink_cls("last")])
    assert pipe(3.0) is None


def test_or_flattens_nested_sequentials(Scale_cls):
    inner = Sequential([Scale_cls(2.0), Scale_cls(3.0)])
    outer = inner | Scale_cls(4.0)
    assert isinstance(outer, Sequential)
    assert len(outer) == 3
    assert outer(1.0) == 24.0


def test_or_with_sequential_on_rhs_flattens(Scale_cls):
    a = Sequential([Scale_cls(2.0)])
    b = Sequential([Scale_cls(3.0), Scale_cls(4.0)])
    combined = a | b
    assert len(combined) == 3
    assert combined(1.0) == 24.0


def test_len(Scale_cls):
    assert len(Sequential([Scale_cls(2.0), Scale_cls(3.0)])) == 2


def test_getitem_index_returns_operator(Scale_cls, AddConst_cls):
    pipe = Sequential([Scale_cls(2.0), AddConst_cls(1.0)])
    assert isinstance(pipe[0], Operator)
    assert pipe[0].get_config() == {"factor": 2.0}


def test_getitem_slice_returns_sequential(Scale_cls, AddConst_cls):
    pipe = Sequential([Scale_cls(2.0), AddConst_cls(1.0), Scale_cls(3.0)])
    sub = pipe[1:3]
    assert isinstance(sub, Sequential)
    assert len(sub) == 2
    assert sub(10.0) == 33.0


def test_get_config_lists_nested_operators(Scale_cls, AddConst_cls):
    pipe = Sequential([Scale_cls(2.0), AddConst_cls(1.0)])
    cfg = pipe.get_config()
    assert cfg == {
        "operators": [
            {"class": "Scale", "config": {"factor": 2.0}},
            {"class": "AddConst", "config": {"value": 1.0}},
        ],
    }


def test_compute_output_signature_threads_through(Scale_cls):
    """Sequential threads input_signature through each step (default passthrough)."""
    pipe = Sequential([Scale_cls(2.0), Scale_cls(3.0)])
    assert pipe.compute_output_signature("input_sig") == "input_sig"


def test_compute_output_signature_short_circuits_on_none():
    class Drops(Operator):
        def _apply(self, x):
            return x

        def compute_output_signature(self, sig):
            return None

    class Survives(Operator):
        called = False

        def _apply(self, x):
            return x

        def compute_output_signature(self, sig):
            type(self).called = True
            return sig

    pipe = Sequential([Drops(), Survives()])
    assert pipe.compute_output_signature("s") is None
    # The downstream op shouldn't have been queried after None
    assert Survives.called is False


def test_describe_emits_indented_tree(Scale_cls, AddConst_cls):
    pipe = Sequential([Scale_cls(2.0), Sequential([AddConst_cls(1.0)])])
    text = pipe.describe()
    assert "Sequential([" in text
    assert "Scale(factor=2.0)" in text
    assert "AddConst(value=1.0)" in text


def test_summary_renders_table(Scale_cls):
    pipe = Sequential([Scale_cls(2.0), Scale_cls(3.0)])
    text = pipe.summary(input_signature="sig0")
    assert "Operator" in text
    assert "Scale" in text
    # All operators preserve sig (default passthrough) so input and
    # output should be "sig0" everywhere.
    assert text.count("sig0") == 2 * len(pipe)


def test_summary_without_input_signature(Scale_cls):
    pipe = Sequential([Scale_cls(2.0)])
    text = pipe.summary()
    assert "?" in text


def test_repr(Scale_cls):
    pipe = Sequential([Scale_cls(2.0)])
    assert repr(pipe) == "Sequential([Scale(factor=2.0)])"
    assert repr(Sequential([])) == "Sequential([])"


class _Identity(Operator):
    def _apply(self, x):
        return x


def test_sequential_state_without_carrier_raises():
    """Calling `pipe(state=...)` without a carrier raises a clear TypeError."""
    from pipekit import CarryState, StatefulOperator

    class Step(StatefulOperator):
        def _apply(self, carrier, state):
            return carrier + 1, state

    class S(CarryState):
        def __init__(self) -> None:
            self.t = 0

    pipe = Sequential([Step()])
    with pytest.raises(TypeError, match="no carrier"):
        pipe(state=S())
