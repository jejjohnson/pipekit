"""Tests for `pipekit_cycle.forward`."""

from __future__ import annotations

import pytest
from pipekit import Identity, Operator
from pipekit_cycle import (
    CallableForward,
    CompositeForward,
    ForwardModel,
    NeuralForward,
)


def test_callable_forward_step_and_apply():
    fwd = CallableForward(fn=lambda s, dt: s + dt, dt=2.0)
    assert fwd.step(10.0, 2.0) == 12.0
    assert fwd(10.0) == 12.0
    assert fwd.dt == 2.0


def test_callable_forward_satisfies_protocol():
    fwd = CallableForward(fn=lambda s, dt: s, dt=1.0)
    assert isinstance(fwd, ForwardModel)


def test_callable_forward_is_forbid_in_yaml():
    assert CallableForward.forbid_in_yaml is True


def test_callable_forward_state_signature():
    fwd = CallableForward(lambda s, dt: s, dt=1.0, state_signature="sig")
    assert fwd.state_signature == "sig"


def test_composite_forward_sums_dt():
    a = CallableForward(lambda s, dt: s + dt, dt=1.0)
    b = CallableForward(lambda s, dt: s + dt, dt=2.0)
    c = CallableForward(lambda s, dt: s + dt, dt=3.0)
    comp = CompositeForward((a, b, c))
    assert comp.dt == 6.0


def test_composite_forward_runs_components_in_order():
    a = CallableForward(lambda s, dt: s + 1, dt=1.0)
    b = CallableForward(lambda s, dt: s * 2, dt=1.0)
    comp = CompositeForward((a, b))
    # (5 + 1) * 2 == 12
    assert comp.step(5, comp.dt) == 12


def test_composite_forward_satisfies_protocol():
    a = CallableForward(lambda s, dt: s, dt=1.0)
    comp = CompositeForward((a,))
    assert isinstance(comp, ForwardModel)


def test_composite_forward_rejects_empty():
    with pytest.raises(ValueError):
        CompositeForward(())


def test_composite_forward_rejects_non_forward_model():
    with pytest.raises(TypeError):
        CompositeForward((object(),))


def test_neural_forward_uses_model_op():
    class Double(Operator):
        def _apply(self, x):
            return x * 2

    fwd = NeuralForward(model_op=Double(), dt=1.0)
    assert fwd(5) == 10
    assert fwd.dt == 1.0


def test_neural_forward_satisfies_protocol():
    fwd = NeuralForward(model_op=Identity(), dt=1.0)
    assert isinstance(fwd, ForwardModel)


def test_neural_forward_rejects_non_operator():
    with pytest.raises(TypeError):
        NeuralForward(model_op=lambda x: x, dt=1.0)  # type: ignore[arg-type]
