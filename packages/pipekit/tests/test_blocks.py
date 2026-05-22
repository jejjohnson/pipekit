"""Tests for `pipekit.blocks` — Group C."""

from __future__ import annotations

from pipekit import Const, Identity, Lambda, Operator, Sequential, Sink


def test_identity_returns_input_unchanged():
    op = Identity()
    assert op(42) == 42
    assert op("hello") == "hello"
    sentinel = object()
    assert op(sentinel) is sentinel


def test_identity_in_sequential_is_noop(Scale_cls):
    pipe = Sequential([Identity(), Scale_cls(2.0), Identity()])
    assert pipe(3.0) == 6.0


def test_identity_round_trips_via_state():
    """Identity has empty config and survives from_state."""
    record = Identity().state
    revived = Operator.from_state(record)
    assert isinstance(revived, Identity)
    assert revived(5) == 5


def test_const_returns_fixed_value():
    op = Const(0.0)
    assert op() == 0.0
    assert op(42) == 0.0
    assert op("anything") == 0.0


def test_const_round_trips_via_state():
    record = Const(42).state
    revived = Operator.from_state(record)
    assert isinstance(revived, Const)
    assert revived() == 42


def test_lambda_wraps_callable():
    twice = Lambda(lambda x: x * 2)
    assert twice(3) == 6


def test_lambda_uses_name_in_repr():
    twice = Lambda(lambda x: x * 2, name="double")
    assert repr(twice) == "Lambda(name='double')"


def test_lambda_defaults_name_from_callable():
    def named_fn(x):
        return x + 1

    op = Lambda(named_fn)
    assert op.name == "named_fn"


def test_lambda_is_forbid_in_yaml():
    """Lambda holds a closure; not YAML-serialisable."""
    assert Lambda.forbid_in_yaml is True


def test_lambda_get_config_is_debug_only():
    op = Lambda(lambda x: x, name="noop")
    cfg = op.get_config()
    # Config contains only the name (debug); the callable itself is gone.
    assert cfg == {"name": "noop"}
    assert "fn" not in cfg


def test_sink_returns_input_unchanged():
    """Sink's return value is the input, not the side-effect result."""
    seen = []

    log = Sink(lambda x: seen.append(x))
    assert log(42) == 42
    assert seen == [42]


def test_sink_in_pipeline_passes_through(Scale_cls, AddConst_cls):
    seen = []

    pipe = Sequential([Scale_cls(2.0), Sink(seen.append), AddConst_cls(1.0)])
    assert pipe(3.0) == 7.0
    # Sink saw the Scale output (6.0), not the AddConst output
    assert seen == [6.0]


def test_sink_is_forbid_in_yaml():
    assert Sink.forbid_in_yaml is True


def test_sink_is_not_terminal():
    """Sink returns the carrier so Sequential can continue past it."""
    assert Sink._terminal is False


def test_sink_uses_name_in_repr():
    sink = Sink(lambda x: None, name="checkpoint")
    assert repr(sink) == "Sink(name='checkpoint')"
