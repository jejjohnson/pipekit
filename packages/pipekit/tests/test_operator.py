"""Tests for `pipekit._base.operator` — Operator + ConfigMixin."""

from __future__ import annotations

import numpy as np
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


def test_from_state_rejects_values_that_are_not_json(Scale_cls):
    """A config value that is not JSON (here, an object) cannot be rebuilt."""
    bad = {
        "module": Scale_cls.__module__,
        "class": "Scale",
        "config": {"factor": object()},
    }
    with pytest.raises(RuntimeError, match="not a JSON value"):
        Operator.from_state(bad)


def test_from_state_rejects_a_nested_payload_without_a_module(Scale_cls):
    """Nested payloads written before nested round-trips had no `module`."""
    legacy = {
        "module": "pipekit._base.sequential",
        "class": "Sequential",
        "config": {"operators": [{"class": "Scale", "config": {"factor": 2.0}}]},
    }
    with pytest.raises(RuntimeError, match="without a `module`"):
        Operator.from_state(legacy)


def test_from_state_checks_the_top_level_class(Scale_cls):
    """`Sequential.from_state` refuses a record that is not a Sequential."""
    from pipekit import Sequential

    with pytest.raises(TypeError, match="is not a Sequential"):
        Sequential.from_state(Scale_cls(2.0).state)


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


def test_operator_rejects_mixed_node_and_literal_args(Scale_cls, TwoInput_cls):
    """Graph-construction mode is all-or-none: mixing Nodes and literals raises."""
    from pipekit import Input

    op = TwoInput_cls()
    a = Input("a")
    with pytest.raises(TypeError, match="every positional arg to be a Node"):
        op(a, 42)
    # Pure literals: eager path, normal _apply.
    assert op(1.0, 2.0) == 3.0
    # Pure Nodes: graph-construction path.
    from pipekit._base.graph import Node

    assert isinstance(op(a, Input("b")), Node)


# ---- backends: an operator refuses inputs on a backend it does not run on ----


class _NumpyOnly(Operator):
    backends = ("numpy",)

    def _apply(self, *xs, **kw):
        return xs[0] if xs else next(iter(kw.values()))


class _Carrier:
    """A named carrier whose raw data sits on whatever backend it holds."""

    def __init__(self, data):
        self.data = data


class _CarrierAdapter:
    def dims(self, x):
        return ("x",)

    def data(self, x):
        return x.data

    def labels(self, x, dim):
        return None

    def isel(self, x, indexers):
        return x

    def rewrap(self, x, data, dims):
        return _Carrier(data)

    def check_compatible(self, *xs):
        return xs

    def signature(self, x):
        return None


@pytest.fixture
def carrier_adapter():
    from pipekit import register_adapter, unregister_adapter

    register_adapter(_Carrier, _CarrierAdapter())
    yield
    unregister_adapter(_Carrier)


def test_an_operator_without_backends_checks_nothing() -> None:
    jnp = pytest.importorskip("jax.numpy")

    class Agnostic(Operator):
        def _apply(self, x):
            return x

    x = jnp.zeros(3)
    assert Agnostic()(x) is x


def test_a_numpy_operator_refuses_a_jax_array() -> None:
    from pipekit import BackendError

    jnp = pytest.importorskip("jax.numpy")
    with pytest.raises(BackendError, match=r"_NumpyOnly runs on numpy only, not jax"):
        _NumpyOnly()(jnp.zeros(3))


def test_a_carrier_is_checked_through_its_adapter(carrier_adapter) -> None:
    from pipekit import BackendError

    jnp = pytest.importorskip("jax.numpy")
    ok = _Carrier(np.zeros(3))
    assert _NumpyOnly()(ok) is ok
    with pytest.raises(BackendError, match="not jax"):
        _NumpyOnly()(_Carrier(jnp.zeros(3)))
    with pytest.raises(BackendError, match="not jax"):
        _NumpyOnly()(x=_Carrier(jnp.zeros(3)))


def test_mixed_backends_are_refused_even_in_a_list() -> None:
    from pipekit import BackendError

    jnp = pytest.importorskip("jax.numpy")

    class Both(_NumpyOnly):
        backends = ("numpy", "jax")

    with pytest.raises(BackendError, match=r"inputs on \['jax', 'numpy'\]"):
        Both()([np.zeros(3), jnp.zeros(3)])


def test_values_without_a_backend_pass() -> None:
    assert _NumpyOnly()("scene.tif") == "scene.tif"
    assert _NumpyOnly()(None, 3) is None


def test_graph_mode_is_not_checked() -> None:
    from pipekit import Input, Node

    assert isinstance(_NumpyOnly()(Input("x")), Node)
