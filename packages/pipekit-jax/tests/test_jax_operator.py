"""Tests for `pipekit_jax.JaxOperator` and `pipekit_jax.testing`."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest


pytest.importorskip("equinox")
pytest.importorskip("jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit import Graph, Input, Operator, Sequential
from pipekit_jax import JaxOperator
from pipekit_jax.operator import jsonable
from pipekit_jax.testing import check_jax_operator


class Diffuse(JaxOperator):
    """u ↦ u + κ ∇²u on a periodic 1-D grid (κ learnable)."""

    dx: float = eqx.field(static=True)
    kappa: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, u):
        lap = (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / self.dx**2
        return u + self.kappa * lap


class Weighted(JaxOperator):
    """u ↦ w ⊙ u with an array-valued learnable field."""

    w: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, u):
        return self.w * u


class NoConverter(JaxOperator):
    """Learnable field *without* a converter — loses leafness on reload."""

    a: jax.Array

    def _apply(self, u):
        return self.a * u


class Branchy(JaxOperator):
    """Python control flow on a traced value — not jit-safe."""

    def _apply(self, u):
        return u if float(u.sum()) > 0 else -u


class Holder(JaxOperator):
    """Holds a callable — not JSON-serialisable, so forbid_in_yaml."""

    forbid_in_yaml: ClassVar[bool] = True
    fn: object = eqx.field(static=True)

    def _apply(self, u):
        return self.fn(u)


U = jnp.sin(jnp.linspace(0.0, 2 * jnp.pi, 16, endpoint=False))


# ---------------------------------------------------------------------
# Operator + Module surface
# ---------------------------------------------------------------------


def test_is_operator_and_module_with_pipekit_call():
    op = Diffuse(dx=0.1, kappa=0.01)
    assert isinstance(op, Operator) and isinstance(op, eqx.Module)
    assert isinstance(op.kappa, jax.Array)  # converter ran
    # eager call
    np.testing.assert_allclose(
        op(U), U + 0.01 * (jnp.roll(U, -1) - 2 * U + jnp.roll(U, 1)) / 0.01
    )
    # symbolic call builds a graph node
    x = Input("u")
    g = Graph(inputs={"u": x}, outputs={"v": op(x)})
    np.testing.assert_allclose(g(u=U)["v"], op(U))


def test_pipe_operator_composes_into_sequential():
    pipe = Diffuse(dx=0.1, kappa=0.01) | Weighted(w=jnp.full(16, 2.0))
    assert isinstance(pipe, Sequential)
    np.testing.assert_allclose(pipe(U), 2.0 * Diffuse(dx=0.1, kappa=0.01)(U))


def test_get_config_is_json_builtins():
    cfg = Weighted(w=jnp.arange(3.0)).get_config()
    assert cfg == {"w": [0.0, 1.0, 2.0]}
    cfg = Diffuse(dx=0.1, kappa=0.5).get_config()
    assert cfg == {"dx": 0.1, "kappa": 0.5}
    json.dumps(Diffuse(dx=0.1, kappa=0.5).state)


def test_jsonable_handles_nested_containers():
    value = {"a": (jnp.asarray(1.5), [np.arange(2)]), "b": "s"}
    assert jsonable(value) == {"a": (1.5, [[0, 1]]), "b": "s"}


def test_from_state_round_trip_keeps_leaves_learnable():
    op = Diffuse(dx=0.1, kappa=0.5)
    rebuilt = Operator.from_state(json.loads(json.dumps(op.state)))
    assert isinstance(rebuilt, Diffuse)
    assert eqx.is_inexact_array(rebuilt.kappa)
    grads = eqx.filter_grad(lambda o: jnp.sum(o(U) ** 2))(rebuilt)
    assert grads.kappa is not None


# ---------------------------------------------------------------------
# Weights round trip
# ---------------------------------------------------------------------


def test_serialize_and_with_weights_round_trip():
    trained = Weighted(w=jnp.linspace(0.0, 1.0, 16))
    blob = trained.serialize_weights()
    skeleton = Weighted(w=jnp.zeros(16))
    restored = skeleton.with_weights(blob)
    assert isinstance(restored, Weighted)
    np.testing.assert_array_equal(restored.w, trained.w)


def test_with_weights_rejects_non_bytes():
    with pytest.raises(TypeError, match="bytes"):
        Weighted(w=jnp.zeros(3)).with_weights("not bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Inside containers (needs the pytree registration)
# ---------------------------------------------------------------------


def test_filter_grad_reaches_jax_operator_inside_graph():
    x = Input("u")
    g = Graph(
        inputs={"u": x},
        outputs={"v": Weighted(w=jnp.full(16, 0.5))(Diffuse(dx=0.1, kappa=0.01)(x))},
    )
    grads = eqx.filter_grad(lambda m: jnp.sum(m(u=U)["v"] ** 2))(g)
    leaves = [leaf for leaf in jax.tree.leaves(grads) if eqx.is_array(leaf)]
    assert len(leaves) == 2  # kappa and w


# ---------------------------------------------------------------------
# check_jax_operator
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [Diffuse(dx=0.1, kappa=0.01), Weighted(w=jnp.linspace(1.0, 2.0, 16))],
    ids=["scalar-leaf", "array-leaf"],
)
def test_check_passes_for_well_formed_operators(op):
    check_jax_operator(op, U)


def test_check_skips_round_trip_for_forbid_in_yaml():
    check_jax_operator(Holder(fn=jnp.tanh), U)


def test_check_flags_a_learnable_field_lost_on_reload():
    with pytest.raises(AssertionError, match=r"converter=jnp\.asarray"):
        check_jax_operator(NoConverter(a=jnp.asarray(2.0)), U)


def test_check_flags_non_jittable_operator():
    with pytest.raises(jax.errors.ConcretizationTypeError):
        check_jax_operator(Branchy(), U)


# ---------------------------------------------------------------------
# Review fixes (#68)
# ---------------------------------------------------------------------


class SpectralGain(JaxOperator):
    """û ↦ g ⊙ û with a complex learnable gain."""

    g: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, u):
        return self.g * u


class SpectralGainForbidden(SpectralGain):
    forbid_in_yaml: ClassVar[bool] = True


class WithOptions(JaxOperator):
    options: dict = eqx.field(static=True)

    def _apply(self, u):
        return u


class OnGrid(JaxOperator):
    """Static tuple config — reloads from JSON as a list."""

    grid: tuple = eqx.field(static=True)
    a: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, u):
        return self.a * u * self.grid[0]


class Mode(JaxOperator):
    """A static (string) positional input next to an array input."""

    def _apply(self, u, mode):
        return u if mode == "identity" else -u


def test_state_rejects_complex_config_with_actionable_message():
    op = SpectralGain(g=jnp.full(4, 1 + 2j))
    with pytest.raises(TypeError, match="forbid_in_yaml"):
        _ = op.state
    # flagged operators keep their (debug) state and pass the contract check
    _ = SpectralGainForbidden(g=jnp.full(16, 1 + 2j)).state
    check_jax_operator(
        SpectralGainForbidden(g=jnp.full(16, 0.5 + 0j)), U.astype(jnp.complex64)
    )


def test_state_rejects_dict_config():
    with pytest.raises(TypeError, match="dicts"):
        _ = WithOptions(options={"axis": 0}).state


def test_check_accepts_static_tuple_reloaded_as_list():
    check_jax_operator(OnGrid(grid=(4, 2.0), a=0.5), U)


def test_check_holds_non_array_inputs_static_under_vmap():
    check_jax_operator(Mode(), U, "identity")
    check_jax_operator(Mode(), U, "negate")


def test_with_weights_rejects_a_blob_with_extra_leaves():
    from pipekit_jax import JaxModelOp

    saved = JaxModelOp(eqx.nn.Linear(4, 4, use_bias=True, key=jax.random.key(0)))
    skeleton = JaxModelOp(eqx.nn.Linear(4, 4, use_bias=False, key=jax.random.key(1)))
    with pytest.raises(ValueError, match="structure"):
        skeleton.with_weights(saved.serialize_weights())
    # a legacy (header-less) blob is caught by the unread-bytes check
    import io

    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, saved.module)
    with pytest.raises(ValueError, match="not consumed"):
        skeleton.with_weights(buf.getvalue())


class Holds(JaxOperator):
    """An operator whose config nests another operator (like Cycle)."""

    inner: JaxOperator

    def get_config(self):
        from pipekit._base.operator import nested_config

        return {"inner": nested_config(self.inner)}

    def _apply(self, u):
        return self.inner(u)


def test_state_allows_nested_operator_payloads():
    record = Holds(inner=Diffuse(dx=0.1, kappa=0.01)).state
    json.dumps(record)
    assert record["config"]["inner"]["class"] == "Diffuse"


# ---------------------------------------------------------------------
# Review fixes, round 2 (#68)
# ---------------------------------------------------------------------


class TwoFields(JaxOperator):
    a: jax.Array = eqx.field(converter=jnp.asarray)
    b: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, u):
        return self.a * u + self.b


class TwoFieldsSwapped(JaxOperator):
    """Same leaf count / shapes / dtypes as TwoFields, fields in the other order."""

    b: jax.Array = eqx.field(converter=jnp.asarray)
    a: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, u):
        return self.a * u + self.b


def test_with_weights_refuses_a_skeleton_with_a_different_field_layout():
    saved = TwoFields(a=jnp.full(3, 2.0), b=jnp.full(3, 5.0))
    skeleton = TwoFieldsSwapped(b=jnp.zeros(3), a=jnp.zeros(3))
    with pytest.raises(ValueError, match="structure"):
        skeleton.with_weights(saved.serialize_weights())
    # the matching skeleton still restores exactly
    restored = TwoFields(a=jnp.zeros(3), b=jnp.zeros(3)).with_weights(
        saved.serialize_weights()
    )
    np.testing.assert_array_equal(restored.a, saved.a)
    np.testing.assert_array_equal(restored.b, saved.b)


def test_with_weights_still_loads_legacy_blobs_without_a_header():
    import io

    saved = TwoFields(a=jnp.full(3, 2.0), b=jnp.full(3, 5.0))
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, saved)  # the pre-header format
    restored = TwoFields(a=jnp.zeros(3), b=jnp.zeros(3)).with_weights(buf.getvalue())
    np.testing.assert_array_equal(restored.b, saved.b)


def test_jsonable_turns_named_tuples_into_plain_tuples():
    from typing import NamedTuple

    class Shape(NamedTuple):
        nx: int
        ny: int

    assert jsonable(Shape(4, 2)) == (4, 2)
    assert jsonable([Shape(1, 2)]) == [(1, 2)]


def test_state_rejects_a_nested_payload_hiding_a_complex_value():
    with pytest.raises(TypeError, match="cannot round-trip"):
        _ = Holds(inner=SpectralGain(g=jnp.full(4, 1 + 2j))).state


def test_check_accepts_operators_with_nested_operator_payloads():
    check_jax_operator(Holds(inner=Diffuse(dx=0.1, kappa=0.01)), U)


# ---------------------------------------------------------------------
# Review fixes, round 3 (#68)
# ---------------------------------------------------------------------


class TwoFieldsLookalike(JaxOperator):
    """A different class with TwoFields' exact field names, shapes and dtypes."""

    a: jax.Array = eqx.field(converter=jnp.asarray)
    b: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, u):
        return self.b * u - self.a


def test_with_weights_refuses_a_different_class_with_identical_leaves():
    saved = TwoFields(a=jnp.full(3, 2.0), b=jnp.full(3, 5.0))
    lookalike = TwoFieldsLookalike(a=jnp.zeros(3), b=jnp.zeros(3))
    with pytest.raises(ValueError, match="node types"):
        lookalike.with_weights(saved.serialize_weights())


def test_jax_model_op_refuses_a_different_module_class_with_identical_leaves():
    from pipekit_jax import JaxModelOp

    class LinearLookalike(eqx.Module):
        weight: jax.Array
        bias: jax.Array

        def __call__(self, x):
            return self.weight @ x + self.bias

    saved = JaxModelOp(eqx.nn.Linear(4, 4, key=jax.random.key(0)))
    skeleton = JaxModelOp(LinearLookalike(jnp.zeros((4, 4)), jnp.zeros(4)))
    with pytest.raises(ValueError, match="node types"):
        skeleton.with_weights(saved.serialize_weights())


class Branches(JaxOperator):
    """Named child operators, configured as {"branches": {name: nested_config}}."""

    left: JaxOperator
    right: JaxOperator

    def get_config(self):
        from pipekit._base.operator import nested_config

        return {
            "branches": {
                "left": nested_config(self.left),
                "right": nested_config(self.right),
            }
        }

    def _apply(self, u):
        return self.left(u) + self.right(u)


def test_state_and_check_accept_mappings_of_nested_operators():
    op = Branches(left=Diffuse(dx=0.1, kappa=0.01), right=Weighted(w=jnp.ones(16)))
    record = op.state
    assert set(record["config"]["branches"]) == {"left", "right"}
    check_jax_operator(op, U)


class HalfDefault(JaxOperator):
    w: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, u):
        return self.w * u


class HalfCasting(JaxOperator):
    w: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, jnp.float16))

    def _apply(self, u):
        return self.w * u


def test_state_refuses_a_field_whose_dtype_json_would_lose():
    with pytest.raises(TypeError, match="different dtype"):
        _ = HalfDefault(w=jnp.ones(4, dtype=jnp.float16)).state


def test_casting_converter_round_trips_the_dtype():
    op = HalfCasting(w=jnp.full(16, 0.5))
    assert op.w.dtype == jnp.float16
    rebuilt = Operator.from_state(json.loads(json.dumps(op.state)))
    assert rebuilt.w.dtype == jnp.float16
    check_jax_operator(op, U.astype(jnp.float16))


# ---------------------------------------------------------------------
# Review fixes, round 4 (#68)
# ---------------------------------------------------------------------


class Balanced(JaxOperator):
    options: dict = eqx.field(static=True)

    def _apply(self, u):
        return u


def test_class_config_keys_alone_do_not_make_a_nested_payload():
    with pytest.raises(TypeError, match="cannot round-trip"):
        _ = Balanced(options={"class": "balanced", "config": 1}).state


class HoldsWithBias(JaxOperator):
    """A container with a nested child *and* a local learnable field."""

    inner: JaxOperator
    bias: jax.Array

    def get_config(self):
        from pipekit._base.operator import nested_config

        return {"inner": nested_config(self.inner), "bias": jsonable(self.bias)}

    def _apply(self, u):
        return self.inner(u) + self.bias


class HoldsWithBiasConverted(HoldsWithBias):
    bias: jax.Array = eqx.field(converter=jnp.asarray)


def test_check_still_validates_local_fields_of_containers():
    inner = Diffuse(dx=0.1, kappa=0.01)
    with pytest.raises(AssertionError, match="converter"):
        check_jax_operator(HoldsWithBias(inner=inner, bias=jnp.asarray(0.5)), U)
    check_jax_operator(HoldsWithBiasConverted(inner=inner, bias=0.5), U)


class Tempered(JaxOperator):
    w: jax.Array = eqx.field(converter=jnp.asarray)
    temperature: float = 1.0  # a non-static Python scalar: configuration

    def _apply(self, u):
        return self.w * u / self.temperature


def test_weights_carry_arrays_only_and_keep_skeleton_config():
    trained = Tempered(w=jnp.full(4, 3.0), temperature=2.0)
    restored = Tempered(w=jnp.zeros(4), temperature=0.5).with_weights(
        trained.serialize_weights()
    )
    np.testing.assert_array_equal(restored.w, trained.w)
    assert restored.temperature == 0.5


class HiddenParam(JaxOperator):
    a: jax.Array = eqx.field(static=True)

    def _apply(self, u):
        return self.a * u


def test_check_flags_an_inexact_array_hidden_in_a_static_field():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # equinox warns about array statics
        op = HiddenParam(a=jnp.asarray(2.0))
    with pytest.raises(AssertionError, match="static"):
        check_jax_operator(op, U)


def test_flagged_complex_operator_round_trips_through_a_registry(tmp_path):
    from pipekit_experiment import LocalModelRegistry

    op = SpectralGainForbidden(g=jnp.full(4, 1 + 2j))
    json.dumps(op.state)  # JSON-safe debug record
    registry = LocalModelRegistry(root=str(tmp_path))
    ref = registry.store(op, weights=op.serialize_weights())
    restored = SpectralGainForbidden(g=jnp.zeros(4, dtype=op.g.dtype)).from_registry(
        registry, ref
    )
    np.testing.assert_array_equal(restored.g, op.g)


class GraphHolder(JaxOperator):
    graph: Any

    def _apply(self, u):
        return self.graph(x=u)["y"]


def _scale_graph(chain: bool):
    from pipekit import Graph, Input

    # Same nodes, outputs, leaf count and shapes; only the wiring differs.
    x = Input("x")
    first = Weighted(w=jnp.full(16, 2.0))(x)
    second = Weighted(w=jnp.full(16, 3.0))(first if chain else x)
    return Graph(inputs={"x": x}, outputs={"y": second, "z": first})


def test_weights_refuse_a_graph_with_different_wiring():
    saved = GraphHolder(graph=_scale_graph(chain=True))
    rewired = GraphHolder(graph=_scale_graph(chain=False))
    with pytest.raises(ValueError, match="node types"):
        rewired.with_weights(saved.serialize_weights())
    same = GraphHolder(graph=_scale_graph(chain=True)).with_weights(
        saved.serialize_weights()
    )
    np.testing.assert_allclose(same(U), saved(U))
