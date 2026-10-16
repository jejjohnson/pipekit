"""Tests for `pipekit_jax.pytree` — Sequential / Graph as JAX pytrees."""

from __future__ import annotations

import subprocess
import sys

import pytest


pytest.importorskip("equinox")
pytest.importorskip("jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pipekit_jax  # noqa: F401  (registers the containers)
from pipekit import Graph, Input, Operator, Sequential


class Scale(Operator, eqx.Module):
    """x ↦ a·x with a learnable scalar ``a``."""

    a: jax.Array

    def _apply(self, x):
        return self.a * x


class Add(Operator, eqx.Module):
    """(x, y) ↦ x + b·y with a learnable scalar ``b``."""

    b: jax.Array

    def _apply(self, x, y):
        return x + self.b * y


class Square(Operator):
    """A plain (non-pytree) operator: one opaque leaf."""

    def _apply(self, x):
        return x * x


class Pick(Operator):
    """Select one output from a graph's dict output."""

    def __init__(self, key: str) -> None:
        self.key = key

    def _apply(self, d):
        return d[self.key]


def _diamond() -> Graph:
    #        x
    #      ┌─┴─┐
    #   Scale  Scale        (shared parent, two consumers)
    #      └─┬─┘
    #       Add            (fused output)
    x = Input("x")
    left = Scale(jnp.asarray(2.0))(x)
    right = Scale(jnp.asarray(3.0))(x)
    out = Add(jnp.asarray(0.5))(left, right)
    return Graph(inputs={"x": x}, outputs={"y": out, "left": left})


X = jnp.linspace(-1.0, 1.0, 5)


def _allclose(a, b):
    jax.tree.map(np.testing.assert_allclose, a, b)


# ---------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------


def test_graph_round_trip_evaluates_identically():
    g = _diamond()
    leaves, treedef = jax.tree.flatten(g)
    assert len(leaves) == 3
    rebuilt = jax.tree.unflatten(treedef, leaves)
    _allclose(rebuilt(x=X), g(x=X))


def test_sequential_round_trip_evaluates_identically():
    pipe = Sequential([Scale(jnp.asarray(2.0)), Scale(jnp.asarray(-1.0))])
    rebuilt = jax.tree.unflatten(*reversed(jax.tree.flatten(pipe)))
    assert isinstance(rebuilt, Sequential)
    _allclose(rebuilt(X), pipe(X))


def test_structure_is_stable_across_tree_map():
    for container in (_diamond(), Sequential([Scale(jnp.asarray(1.0))])):
        mapped = jax.tree.map(lambda leaf: leaf + 1.0, container)
        assert jax.tree.structure(mapped) == jax.tree.structure(container)
        # ... and across a second round trip from the rebuilt object.
        remapped = jax.tree.map(lambda leaf: leaf, mapped)
        assert jax.tree.structure(remapped) == jax.tree.structure(container)


def test_tree_map_changes_values_not_topology():
    g = jax.tree.map(lambda leaf: 2 * leaf, _diamond())
    # a: 4, 6; b: 1.0  →  y = 4x + 1.0 * 6x = 10x
    _allclose(g(x=X)["y"], 10.0 * X)
    assert set(g(x=X)) == {"y", "left"}


def test_nested_graph_in_graph_and_in_sequential():
    inner = _diamond()
    x = Input("x")
    outer = Graph(inputs={"x": x}, outputs={"z": Scale(jnp.asarray(-1.0))(x)})
    pipe = Sequential([Scale(jnp.asarray(2.0)), outer])
    # Graph nested in a Graph node
    xi = Input("x")
    wrapper = Graph(inputs={"x": xi}, outputs={"inner": inner(xi)})

    for container in (pipe, wrapper):
        rebuilt = jax.tree.unflatten(*reversed(jax.tree.flatten(container)))
        _allclose(rebuilt(X), container(X))
    assert len(jax.tree.leaves(pipe)) == 2
    assert len(jax.tree.leaves(wrapper)) == 3


def test_input_passthrough_output():
    x = Input("x")
    g = Graph(
        inputs={"x": x}, outputs={"same": x, "scaled": Scale(jnp.asarray(2.0))(x)}
    )
    rebuilt = jax.tree.map(lambda leaf: leaf, g)
    _allclose(rebuilt(x=X), {"same": X, "scaled": 2.0 * X})


# ---------------------------------------------------------------------
# Plain (non-pytree) operators and Equinox filtering
# ---------------------------------------------------------------------


def test_plain_operator_is_an_opaque_leaf():
    pipe = Sequential([Square(), Scale(jnp.asarray(2.0))])
    leaves = jax.tree.leaves(pipe)
    assert any(isinstance(leaf, Square) for leaf in leaves)
    assert len(leaves) == 2


def test_eqx_partition_combine_round_trips_with_placeholders():
    # partition replaces filtered-out leaves with None; unflatten must not
    # run Sequential's constructor validation on them.
    pipe = Sequential([Square(), Scale(jnp.asarray(2.0))])
    params, static = eqx.partition(pipe, eqx.is_inexact_array)
    _allclose(eqx.combine(params, static)(X), pipe(X))


# ---------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------


def test_filter_jit_does_not_retrace_when_only_leaf_values_change():
    traces = []

    @eqx.filter_jit
    def run(container, x):
        traces.append(1)
        return container(x=x)["y"]

    g = _diamond()
    run(g, X)
    run(jax.tree.map(lambda leaf: leaf * 1.5, g), X)
    assert len(traces) == 1


def test_filter_grad_through_graph_in_sequential_matches_finite_differences():
    pipe = Sequential([_diamond(), Pick("y"), Scale(jnp.asarray(0.5))])

    def loss(p):
        return jnp.sum(p(X) ** 2)

    grads = eqx.filter_grad(loss)(pipe)
    leaves = jax.tree.leaves(grads)
    assert len(leaves) == 4 and all(float(jnp.abs(g)) > 0 for g in leaves)

    # Central differences on the Add coefficient b (third graph leaf).
    def with_b(b):
        g = pipe.operators[0]
        new_g = jax.tree.unflatten(
            jax.tree.structure(g), [*jax.tree.leaves(g)[:2], jnp.asarray(b)]
        )
        return loss(Sequential([new_g, *pipe.operators[1:]]))

    eps = 1e-3
    fd = (with_b(0.5 + eps) - with_b(0.5 - eps)) / (2 * eps)
    np.testing.assert_allclose(jax.tree.leaves(grads.operators[0])[2], fd, rtol=1e-3)


def test_jax_jit_with_all_pytree_operators_as_argument():
    g = _diamond()
    out = jax.jit(lambda graph, x: graph(x=x)["y"])(g, X)
    _allclose(out, g(x=X)["y"])


def test_vmap_over_a_leaf_of_the_pipeline():
    pipe = Sequential([Scale(jnp.asarray(1.0))])
    stacked = jax.tree.map(lambda leaf: jnp.stack([leaf, 2 * leaf, 3 * leaf]), pipe)
    out = jax.vmap(lambda p: p(X))(stacked)
    _allclose(out, jnp.stack([X, 2 * X, 3 * X]))


# ---------------------------------------------------------------------
# Core stays JAX-free
# ---------------------------------------------------------------------


def test_pipekit_core_does_not_import_jax():
    code = (
        "import sys, pipekit\n"
        "from pipekit import Sequential, Identity\n"
        "Sequential([Identity()])(1)\n"
        "assert 'jax' not in sys.modules, 'pipekit core imported jax'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------------
# Aux data is topology only (review: #67)
# ---------------------------------------------------------------------


def test_identically_wired_graphs_share_a_treedef_and_compile_once():
    traces = []

    @eqx.filter_jit
    def run(container, x):
        traces.append(1)
        return container(x=x)["y"]

    a, b = _diamond(), _diamond()  # built separately, same wiring
    assert jax.tree.structure(a) == jax.tree.structure(b)
    run(a, X)
    run(b, X)
    assert len(traces) == 1


def test_rebuilt_graph_does_not_keep_the_original_alive():
    import gc
    import weakref

    g = _diamond()
    ref = weakref.ref(g)
    rebuilt = jax.tree.map(lambda leaf: leaf + 1.0, g)
    del g
    gc.collect()
    assert ref() is None
    _allclose(rebuilt(x=X)["y"], (3.0 * X) + 1.5 * (4.0 * X))  # a: 3, 4; b: 1.5


def test_differently_wired_graphs_have_different_treedefs():
    x = Input("x")
    straight = Graph(inputs={"x": x}, outputs={"y": Scale(jnp.asarray(1.0))(x)})
    x2 = Input("x")
    passthrough = Graph(
        inputs={"x": x2}, outputs={"y": x2, "z": Scale(jnp.asarray(1.0))(x2)}
    )
    assert jax.tree.structure(straight) != jax.tree.structure(passthrough)


# ---------------------------------------------------------------------
# JaxModelOp exposes its module (review: #67)
# ---------------------------------------------------------------------


def test_filter_grad_reaches_jax_model_op_weights_inside_a_pipeline():
    from pipekit_jax import JaxModelOp

    mlp = eqx.nn.MLP(
        in_size=5, out_size=5, width_size=8, depth=1, key=jax.random.key(0)
    )
    pipe = Sequential([JaxModelOp(mlp), Scale(jnp.asarray(2.0))])

    grads = eqx.filter_grad(lambda p: jnp.sum(p(X) ** 2))(pipe)
    direct = eqx.filter_grad(lambda m: jnp.sum((2.0 * m(X)) ** 2))(mlp)
    _allclose(
        eqx.filter(grads.operators[0].module, eqx.is_array),
        eqx.filter(direct, eqx.is_array),
    )


def test_jax_model_op_partition_combine_round_trip():
    from pipekit_jax import JaxModelOp

    op = JaxModelOp(eqx.nn.Linear(5, 5, key=jax.random.key(1)))
    params, static = eqx.partition(op, eqx.is_inexact_array)
    rebuilt = eqx.combine(params, static)
    assert isinstance(rebuilt, JaxModelOp)
    _allclose(rebuilt(X), op(X))
