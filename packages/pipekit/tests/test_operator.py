"""Tests for `pipekit._base.operator` — Operator + ConfigMixin."""

from __future__ import annotations

import pytest
from pipekit import ConfigMixin, Operator, Sequential


def test_operator_eager_call(Scale_cls):
    op = Scale_cls(2.0)
    assert op(3.0) == 6.0


def test_operator_must_implement_apply():
    class Empty(Operator):
        pass

    with pytest.raises(NotImplementedError, match="Empty must implement _apply"):
        Empty()(1.0)


def test_configmixin_auto_derives_from_init(Scale_cls):
    """ConfigMixin reads __init__ param names off `self` attrs."""
    op = Scale_cls(factor=2.5)
    assert op.get_config() == {"factor": 2.5}


def test_configmixin_opt_out():
    class Hidden(Operator):
        __config_mixin_auto__ = False

        def __init__(self, secret: str) -> None:
            self.secret = secret

        def _apply(self, x):
            return x

    assert Hidden(secret="hush").get_config() == {}


def test_configmixin_exclude():
    class Partial(Operator):
        __config_exclude__ = ("runtime_only",)

        def __init__(self, factor: float, runtime_only: object = None) -> None:
            self.factor = factor
            self.runtime_only = runtime_only

        def _apply(self, x):
            return x * self.factor

    op = Partial(2.0, runtime_only=object())
    assert op.get_config() == {"factor": 2.0}


def test_configmixin_can_be_used_standalone():
    """ConfigMixin is usable on non-Operator classes too."""

    class NotAnOperator(ConfigMixin):
        def __init__(self, a: int, b: str) -> None:
            self.a = a
            self.b = b

    assert NotAnOperator(1, "x").get_config() == {"a": 1, "b": "x"}


def test_repr_uses_config(Scale_cls):
    assert repr(Scale_cls(2.0)) == "Scale(factor=2.0)"


def test_state_roundtrip(Scale_cls):
    op = Scale_cls(2.5)
    record = op.state
    assert record["module"] == Scale_cls.__module__
    assert record["class"] == "Scale"
    assert record["config"] == {"factor": 2.5}

    revived = Operator.from_state(record)
    assert isinstance(revived, Scale_cls)
    assert revived.factor == 2.5
    assert revived(4.0) == 10.0


def test_from_state_rejects_unknown_class():
    with pytest.raises(TypeError, match="is not a loaded Operator"):
        Operator.from_state({"module": "no.such.module", "class": "Nope", "config": {}})


def test_from_state_rejects_non_primitive_config(Scale_cls):
    """Operators whose config nests objects aren't round-trippable via from_state."""
    bad = {
        "module": Scale_cls.__module__,
        "class": "Scale",
        "config": {"factor": {"nested": "dict"}},  # dicts are rejected
    }
    with pytest.raises(RuntimeError, match="non-primitive values"):
        Operator.from_state(bad)


def test_from_state_validates_record_shape():
    with pytest.raises(ValueError, match="module name"):
        Operator.from_state({"class": "X", "config": {}})
    with pytest.raises(ValueError, match="class name"):
        Operator.from_state({"module": "x", "config": {}})
    with pytest.raises(ValueError, match="config"):
        Operator.from_state({"module": "x", "class": "X", "config": "nope"})


def test_pipe_operator_returns_sequential(Scale_cls, AddConst_cls):
    pipe = Scale_cls(2.0) | AddConst_cls(1.0)
    assert isinstance(pipe, Sequential)
    assert pipe(3.0) == 7.0


def test_pipe_operator_flattens_nested(Scale_cls, AddConst_cls):
    """``a | (b | c)`` produces a flat 3-element Sequential."""
    a, b, c = Scale_cls(2.0), AddConst_cls(1.0), Scale_cls(3.0)
    pipe = a | (b | c)
    assert isinstance(pipe, Sequential)
    assert len(pipe) == 3
    # And the value is correct
    assert pipe(1.0) == ((1.0 * 2.0) + 1.0) * 3.0


def test_pipe_operator_rejects_non_operator(Scale_cls):
    """`a | 5` returns NotImplemented, which Python converts to TypeError."""
    with pytest.raises(TypeError):
        Scale_cls(2.0) | 5  # type: ignore[operator]


def test_compute_output_signature_default_is_passthrough(Scale_cls):
    """Default compute_output_signature returns the input unchanged."""
    sig_marker = object()
    assert Scale_cls(2.0).compute_output_signature(sig_marker) is sig_marker


def test_class_flags_default_false(Scale_cls):
    assert Scale_cls.forbid_in_yaml is False
    assert Scale_cls._terminal is False
