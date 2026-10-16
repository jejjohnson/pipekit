"""Tests for `pipekit_jax.JaxOperator` and `pipekit_jax.testing`."""

from __future__ import annotations

import json
from typing import ClassVar

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
    with pytest.raises(ValueError, match="not consumed"):
        skeleton.with_weights(saved.serialize_weights())
