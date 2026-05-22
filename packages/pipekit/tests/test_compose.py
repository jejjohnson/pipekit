"""Tests for `pipekit.compose` — Group B."""

from __future__ import annotations

import pytest
from pipekit import Sequential, complement, compose, compose_left, juxt, pipe


def test_pipe_applies_left_to_right(Scale_cls, AddConst_cls):
    """pipe(value, *ops) == Sequential(ops)(value)."""
    assert pipe(3.0, Scale_cls(2.0), AddConst_cls(1.0)) == 7.0


def test_pipe_no_operators_returns_value():
    """pipe(value) with zero operators returns the value unchanged."""
    assert pipe(42) == 42


def test_compose_is_right_to_left(Scale_cls, AddConst_cls):
    """compose(f, g)(x) == f(g(x)) — mathematical composition order."""
    composed = compose(Scale_cls(2.0), AddConst_cls(1.0))
    # AddConst runs first: (3 + 1) = 4; Scale runs second: 4 * 2 = 8
    assert composed(3.0) == 8.0


def test_compose_returns_sequential(Scale_cls):
    """compose() returns a Sequential so it can be further composed via |."""
    composed = compose(Scale_cls(2.0), Scale_cls(3.0))
    assert isinstance(composed, Sequential)
    assert len(composed) == 2


def test_compose_left_is_left_to_right(Scale_cls, AddConst_cls):
    """compose_left is an explicit alias of Sequential."""
    composed = compose_left(Scale_cls(2.0), AddConst_cls(1.0))
    # Scale first (3 * 2 = 6), then AddConst (6 + 1 = 7)
    assert composed(3.0) == 7.0


def test_compose_left_returns_sequential(Scale_cls):
    composed = compose_left(Scale_cls(2.0))
    assert isinstance(composed, Sequential)


def test_complement_negates_predicate():
    is_positive = lambda x: x > 0
    is_non_positive = complement(is_positive)
    assert is_non_positive(5) is False
    assert is_non_positive(-3) is True
    assert is_non_positive(0) is True


def test_complement_preserves_name():
    def my_check(x):
        return x > 0

    negated = complement(my_check)
    assert "my_check" in negated.__name__


def test_juxt_returns_tuple(Scale_cls, AddConst_cls):
    """juxt(f, g, h)(x) == (f(x), g(x), h(x))."""
    triple = juxt(Scale_cls(2.0), Scale_cls(3.0), AddConst_cls(1.0))
    assert triple(5.0) == (10.0, 15.0, 6.0)


def test_juxt_empty_returns_empty_tuple():
    """juxt() with no operators returns a callable producing an empty tuple."""
    nothing = juxt()
    assert nothing(42) == ()


def test_juxt_is_a_callable_not_an_operator(Scale_cls):
    """juxt returns a plain callable, not a Sequential / Operator."""
    from pipekit import Operator

    j = juxt(Scale_cls(2.0))
    assert callable(j)
    assert not isinstance(j, Operator)


def test_complement_with_args():
    """complement forwards *args, **kwargs to the predicate."""

    def both_positive(a, b):
        return a > 0 and b > 0

    neither_or_one = complement(both_positive)
    assert neither_or_one(1, 2) is False
    assert neither_or_one(-1, 2) is True


def test_pipe_rejects_non_operator():
    """pipe wraps in Sequential which enforces Operator instances."""
    with pytest.raises(TypeError, match="expected Operator"):
        pipe(1, lambda x: x)  # type: ignore[arg-type]
