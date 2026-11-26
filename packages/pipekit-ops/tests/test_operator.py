"""Tests for `pipekit_ops.core` — Operator + ConfigMixin."""

from __future__ import annotations

import numpy as np
import pytest
from pipekit_ops import ConfigMixin, Operator, Sequential


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
        "module": "pipekit_ops._base.sequential",
        "class": "Sequential",
        "config": {"operators": [{"class": "Scale", "config": {"factor": 2.0}}]},
    }
    with pytest.raises(RuntimeError, match="without a `module`"):
        Operator.from_state(legacy)


def test_from_state_checks_the_top_level_class(Scale_cls):
    """`Sequential.from_state` refuses a record that is not a Sequential."""
    from pipekit_ops import Sequential

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
    from pipekit_ops import Input

    op = TwoInput_cls()
    a = Input("a")
    with pytest.raises(TypeError, match="every positional arg to be a Node"):
        op(a, 42)
    # Pure literals: eager path, normal _apply.
    assert op(1.0, 2.0) == 3.0
    # Pure Nodes: graph-construction path.
    from pipekit_ops.core._graph import Node

    assert isinstance(op(a, Input("b")), Node)


# ---- backends: NumPy and JAX inputs go where the operator runs ----


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
    from pipekit_ops import register_adapter, unregister_adapter

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


class _Seen(_NumpyOnly):
    """Records the types it was called with."""

    def _apply(self, *xs, **named):
        self.types = [type(v).__module__ for v in (*xs, *named.values())]
        return xs[0] if len(xs) == 1 else list(xs)


def test_a_numpy_operator_takes_jax_data_and_returns_jax() -> None:
    jax = pytest.importorskip("jax")
    op = _Seen()
    out = op(jax.numpy.arange(3.0))
    assert op.types == ["numpy"]
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, op(np.arange(3.0)))
    assert isinstance(op(np.arange(3.0)), np.ndarray)


def test_the_host_copy_is_writable() -> None:
    jnp = pytest.importorskip("jax.numpy")

    class InPlace(_NumpyOnly):
        def _apply(self, x):
            x[0] = 1.0
            return x

    given = jnp.zeros(2)
    np.testing.assert_array_equal(InPlace()(given), [1.0, 0.0])
    np.testing.assert_array_equal(given, [0.0, 0.0])


def test_a_mix_runs_on_the_host_for_a_numpy_operator() -> None:
    jax = pytest.importorskip("jax")
    op = _Seen()
    out = op(np.zeros(2), jax.numpy.ones(2))
    assert op.types == ["numpy", "numpy"]
    assert all(isinstance(x, jax.Array) for x in out)


@pytest.mark.parametrize("backends", [("numpy", "jax"), ("jax",)])
def test_numpy_joins_jax_for_an_operator_that_lists_it(backends) -> None:
    jax = pytest.importorskip("jax")

    class Sums(Operator):
        def _apply(self, xs):
            assert all(isinstance(x, jax.Array) for x in xs)
            return sum(xs)

    Sums.backends = backends
    for xs in ([np.zeros(3), jax.numpy.ones(3)], [jax.numpy.ones(3), np.zeros(3)]):
        out = Sums()(xs)
        assert isinstance(out, jax.Array)
        np.testing.assert_array_equal(out, np.ones(3))


def test_a_jax_operator_takes_numpy_data() -> None:
    jax = pytest.importorskip("jax")

    class JaxOnly(_NumpyOnly):
        backends = ("jax",)

    assert isinstance(JaxOnly()(np.zeros(2)), jax.Array)


@pytest.mark.parametrize("transform", ["jit", "grad", "vmap"])
def test_a_numpy_operator_names_itself_under_a_trace(transform) -> None:
    from pipekit_ops import BackendError

    jax = pytest.importorskip("jax")
    run = getattr(jax, transform)(lambda x: _NumpyOnly()(x).sum())
    with pytest.raises(BackendError, match=r"_NumpyOnly runs on numpy only.*jax\.jit"):
        run(jax.numpy.ones(3))


def test_a_constant_joins_a_trace() -> None:
    jax = pytest.importorskip("jax")
    from pipekit_ops import guard

    class Scale(_NumpyOnly):
        backends = ("numpy", "jax")

        def _apply(self, x, weight):
            return x * weight

    weight = np.full(3, 2.0, np.float32)
    with guard("raise"):
        out = jax.jit(lambda x: Scale()(x, weight))(jax.numpy.ones(3))
    np.testing.assert_array_equal(out, [2.0, 2.0, 2.0])


def test_the_guard_names_the_operator() -> None:
    from pipekit_ops import BackendError, guard

    jnp = pytest.importorskip("jax.numpy")

    class Both(_NumpyOnly):
        backends = ("numpy", "jax")

    with guard("raise"):
        _NumpyOnly()(np.zeros(2))
        Both()(jnp.zeros(2))
        with pytest.raises(BackendError, match=r"^_NumpyOnly: JAX data copied to the"):
            _NumpyOnly()(jnp.zeros(2))
        with pytest.raises(BackendError, match=r"^Both: NumPy data joined jax"):
            Both()([np.zeros(2), jnp.zeros(2)])


def test_dask_keeps_its_rules() -> None:
    from pipekit_ops import BackendError

    da = pytest.importorskip("dask.array")
    jnp = pytest.importorskip("jax.numpy")

    class DaskOnly(_NumpyOnly):
        backends = ("dask",)

    with pytest.raises(BackendError, match="_NumpyOnly runs on numpy only, not dask"):
        _NumpyOnly()(da.zeros(2))
    with pytest.raises(BackendError, match="DaskOnly runs on dask only, not jax"):
        DaskOnly()(jnp.zeros(2))
    for other in (np.zeros(2), jnp.zeros(2)):
        with pytest.raises(BackendError, match="inputs on"):
            _NumpyOnly()([da.zeros(2), other])


def test_a_carrier_converts_through_its_adapter(carrier_adapter) -> None:
    jax = pytest.importorskip("jax")
    ok = _Carrier(np.zeros(3))
    assert _NumpyOnly()(ok) is ok
    op = _Seen()
    out = op(_Carrier(jax.numpy.zeros(3)))
    assert isinstance(out.data, jax.Array)
    assert isinstance(op(x0 := ok).data, np.ndarray) and x0 is ok


class _Located:
    """A carrier with a second array, which ``rewrap`` does not see."""

    def __init__(self, data, lat):
        self.data, self.lat = data, lat


class _LocatedAdapter(_CarrierAdapter):
    def to_backend(self, x, backend):
        from pipekit_ops import to_backend

        return _Located(to_backend(x.data, backend), to_backend(x.lat, backend))


def test_to_backend_of_an_adapter_moves_every_array() -> None:
    from pipekit_ops import ConvertingAdapter, register_adapter, unregister_adapter

    jax = pytest.importorskip("jax")
    assert isinstance(_LocatedAdapter(), ConvertingAdapter)
    register_adapter(_Located, _LocatedAdapter())
    try:
        seen = []

        class Reads(_NumpyOnly):
            def _apply(self, x):
                seen.extend([type(x.data), type(x.lat)])
                return x

        out = Reads()(_Located(jax.numpy.zeros(2), jax.numpy.ones(2)))
    finally:
        unregister_adapter(_Located)
    assert seen == [np.ndarray, np.ndarray]
    assert isinstance(out.data, jax.Array) and isinstance(out.lat, jax.Array)


def test_to_backend_of_containers_and_other_values() -> None:
    from typing import NamedTuple

    from pipekit_ops import to_backend

    jax = pytest.importorskip("jax")

    class Stats(NamedTuple):
        mean: object
        name: str

    given = {
        "stats": Stats(np.zeros(2), "b4"),
        "list": [np.ones(1), 3],
        "time": np.array(["2026-11-24"], "datetime64[D]"),
        "text": np.array(["a"]),
        "key": jax.random.key(0),
        "bfloat": np.array(jax.numpy.ones(1, "bfloat16")),
    }
    out = to_backend(given, "jax")
    assert isinstance(out["stats"], Stats) and out["stats"].name == "b4"
    assert isinstance(out["stats"].mean, jax.Array)
    assert isinstance(out["list"][0], jax.Array) and out["list"][1] == 3
    assert out["time"] is given["time"] and out["text"] is given["text"]
    assert out["bfloat"].dtype == jax.numpy.bfloat16
    back = to_backend(out, "numpy")
    assert isinstance(back["list"][0], np.ndarray)
    assert back["key"] is given["key"]


def test_values_without_a_backend_pass() -> None:
    assert _NumpyOnly()("scene.tif") == "scene.tif"
    assert _NumpyOnly()(None, 3) is None


def test_graph_mode_is_not_checked() -> None:
    from pipekit_ops import Input, Node

    assert isinstance(_NumpyOnly()(Input("x")), Node)


def test_numpy_scalars_and_dtypes_have_no_backend() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from pipekit_ops import backend_of

    class Both(_NumpyOnly):
        backends = ("jax",)

    x = jnp.zeros(3)
    assert backend_of(np.int64(0)) is None
    assert backend_of(np.dtype("float32")) is None
    assert Both()(x, axis=np.int64(0), dtype=np.dtype("float32")) is x


def test_async_map_places_the_input_of_a_coroutine_apply() -> None:
    from pipekit_ops import AsyncMap

    jax = pytest.importorskip("jax")

    class AsyncNumpy(Operator):
        backends = ("numpy",)

        async def _apply(self, x):
            assert isinstance(x, np.ndarray)
            return x

    assert AsyncMap(AsyncNumpy())([np.zeros(2)])[0].shape == (2,)
    assert isinstance(AsyncMap(AsyncNumpy())([jax.numpy.zeros(2)])[0], jax.Array)
