"""Tests for `pipekit_array.combinators` — `StackAlong`, `ConcatenateAlong`."""

from __future__ import annotations

import pytest
from pipekit import Operator
from pipekit_array import ConcatenateAlong, StackAlong


# --- StackAlong -----------------------------------------------------------


def test_stack_along_axis_zero(backend, make_array):
    a = make_array(backend, shape=(4, 8), seed=0)
    b = make_array(backend, shape=(4, 8), seed=1)
    out = StackAlong(axis=0)([a, b])
    assert out.shape == (2, 4, 8)


def test_stack_along_axis_negative(backend, make_array):
    a = make_array(backend, shape=(4,), seed=0)
    b = make_array(backend, shape=(4,), seed=1)
    out = StackAlong(axis=-1)([a, b])
    assert out.shape == (4, 2)


def test_stack_along_three_inputs(backend, make_array):
    arrs = [make_array(backend, shape=(3, 5), seed=i) for i in range(3)]
    out = StackAlong(axis=1)(arrs)
    assert out.shape == (3, 3, 5)


def test_stack_along_empty_list_raises():
    with pytest.raises(ValueError, match="at least one input"):
        StackAlong()([])


def test_stack_along_get_config_round_trip():
    op = StackAlong(axis=2)
    assert op.get_config() == {"axis": 2}
    revived = Operator.from_state(op.state)
    assert isinstance(revived, StackAlong)
    assert revived.axis == 2


# --- ConcatenateAlong ----------------------------------------------------


def test_concatenate_along_axis_zero(backend, make_array):
    a = make_array(backend, shape=(2, 8), seed=0)
    b = make_array(backend, shape=(3, 8), seed=1)
    out = ConcatenateAlong(axis=0)([a, b])
    assert out.shape == (5, 8)


def test_concatenate_along_axis_one(backend, make_array):
    a = make_array(backend, shape=(4, 3), seed=0)
    b = make_array(backend, shape=(4, 5), seed=1)
    out = ConcatenateAlong(axis=1)([a, b])
    assert out.shape == (4, 8)


def test_concatenate_along_single_input(backend, make_array):
    a = make_array(backend, shape=(4, 8), seed=0)
    out = ConcatenateAlong(axis=0)([a])
    assert out.shape == (4, 8)


def test_concatenate_along_empty_list_raises():
    with pytest.raises(ValueError, match="at least one input"):
        ConcatenateAlong()([])


def test_concatenate_along_get_config_round_trip():
    op = ConcatenateAlong(axis=-1)
    assert op.get_config() == {"axis": -1}
    revived = Operator.from_state(op.state)
    assert isinstance(revived, ConcatenateAlong)
    assert revived.axis == -1


# --- Cross-operator integration -----------------------------------------


def test_stack_then_concatenate(backend, make_array):
    """Stack two arrays, then concatenate two of those stacks."""
    a = make_array(backend, shape=(4, 8), seed=0)
    b = make_array(backend, shape=(4, 8), seed=1)
    stack_op = StackAlong(axis=0)
    stacked_1 = stack_op([a, b])  # (2, 4, 8)
    stacked_2 = stack_op([a, b])  # (2, 4, 8)
    out = ConcatenateAlong(axis=0)([stacked_1, stacked_2])
    assert out.shape == (4, 4, 8)
