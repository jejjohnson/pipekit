"""Tests for `pipekit_cycle.obs`."""

from __future__ import annotations

import pytest
from pipekit import Operator
from pipekit_cycle import (
    CallableObs,
    CompositeObs,
    IdentityObs,
    LinearObs,
    ObservationOperator,
)


def test_identity_obs_returns_input():
    op = IdentityObs()
    assert op(42) == 42


def test_identity_obs_satisfies_protocol():
    assert isinstance(IdentityObs(), ObservationOperator)


def test_identity_obs_linearize_is_self():
    op = IdentityObs()
    assert op.linearize(42) is op


def test_linear_obs_matrix_multiply():
    # Use a simple list-of-lists with __matmul__ via a fake.
    class FakeH:
        def __matmul__(self, x):
            return x * 2

    obs = LinearObs(FakeH())
    assert obs(3) == 6


def test_linear_obs_linearize_is_self():
    obs = LinearObs(H=object())
    assert obs.linearize(None) is obs


def test_linear_obs_satisfies_protocol():
    assert isinstance(LinearObs(H=object()), ObservationOperator)


def test_callable_obs_dispatches():
    op = CallableObs(lambda s: s + 100, name="add100")
    assert op(5) == 105
    assert op.name == "add100"


def test_callable_obs_linearize_raises_without_fn():
    op = CallableObs(lambda s: s)
    with pytest.raises(NotImplementedError):
        op.linearize(None)


def test_callable_obs_linearize_with_fn():
    op = CallableObs(lambda s: s, linearize_fn=lambda s: "lin")
    assert op.linearize(1) == "lin"


def test_callable_obs_is_forbid_in_yaml():
    assert CallableObs.forbid_in_yaml is True


def test_composite_obs_composes_left_to_right():
    plus_one = CallableObs(lambda s: s + 1)
    times_two = CallableObs(lambda s: s * 2)
    comp = CompositeObs((plus_one, times_two))
    # (3 + 1) * 2 == 8
    assert comp(3) == 8


def test_composite_obs_rejects_empty():
    with pytest.raises(ValueError):
        CompositeObs(())


def test_composite_obs_rejects_non_operator():
    with pytest.raises(TypeError):
        CompositeObs((lambda s: s,))  # type: ignore[arg-type]


def test_composite_obs_linearize_chains():
    # Identity components → composite is identity.
    comp = CompositeObs((IdentityObs(), IdentityObs()))
    lin = comp.linearize(5)
    assert isinstance(lin, CompositeObs)
    assert lin(5) == 5


def test_composite_obs_satisfies_operator_interface():
    comp = CompositeObs((IdentityObs(),))
    assert isinstance(comp, Operator)
