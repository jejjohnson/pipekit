"""Contract checks for JAX operators built on `JaxOperator`.

Call `check_jax_operator` from a library's test suite on every operator
it ships::

    def test_contract():
        check_jax_operator(Diffuse(dx=0.1, kappa=0.5), jnp.sin(x))
"""

from __future__ import annotations

import json
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit import Operator


def _assert_tree_allclose(actual: Any, expected: Any, what: str) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected), (
        f"{what}: output structure differs"
    )
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(
            np.asarray(a), np.asarray(e), rtol=1e-5, atol=1e-6, err_msg=what
        )


def check_jax_operator(op: Any, *inputs: Any, vmap: bool = True) -> None:
    """Assert that ``op`` behaves as a JAX-native pipekit operator.

    Checks, in order:

    1. ``op`` is a pipekit ``Operator`` and an ``eqx.Module``.
    2. Eager ≡ jit: ``op(*inputs)`` equals ``eqx.filter_jit`` of the same
       call, with ``op`` passed as an argument (so its leaves are traced).
    3. Batching (when ``vmap``): stacking every input array along a new
       leading axis and ``jax.vmap``-ing gives the stacked eager outputs.
    4. Leaves visible: every inexact-array field of ``op`` is a leaf of
       ``jax.tree.leaves(op)``, and ``eqx.filter_grad`` of a scalar loss
       returns a gradient for each of them.
    5. Config round trip (unless ``forbid_in_yaml``): ``op.state``
       survives ``json.dumps`` → ``Operator.from_state``; the rebuilt
       operator has the same pytree structure, the same number of
       inexact-array leaves (a learnable field that came back as a
       Python float fails here), and the same outputs.

    Args:
        op: The operator under test.
        *inputs: Example positional inputs for ``op``.
        vmap: Also check batching over a new leading axis.

    Raises:
        AssertionError: on the first violated property.
    """
    assert isinstance(op, Operator), f"{type(op).__name__} is not a pipekit Operator"
    assert isinstance(op, eqx.Module), f"{type(op).__name__} is not an eqx.Module"

    expected = op(*inputs)
    jitted = eqx.filter_jit(lambda o, *xs: o(*xs))(op, *inputs)
    _assert_tree_allclose(jitted, expected, "eager vs filter_jit")

    if vmap:
        batched_inputs = jax.tree.map(lambda x: jnp.stack([x, x]), inputs)
        batched = jax.vmap(lambda *xs: op(*xs))(*batched_inputs)
        _assert_tree_allclose(
            batched,
            jax.tree.map(lambda y: jnp.stack([y, y]), expected),
            "vmap over a leading axis",
        )

    params = [leaf for leaf in jax.tree.leaves(op) if eqx.is_inexact_array(leaf)]
    if params:

        def loss(o: Any) -> Any:
            out = o(*inputs)
            return sum(jnp.sum(jnp.real(y)) for y in jax.tree.leaves(out))

        grads = eqx.filter_grad(loss)(op)
        grad_leaves = [g for g in jax.tree.leaves(grads) if eqx.is_array(g)]
        assert len(grad_leaves) == len(params), (
            f"filter_grad returned {len(grad_leaves)} gradients for "
            f"{len(params)} inexact-array leaves"
        )

    if type(op).forbid_in_yaml:
        return
    state = json.loads(json.dumps(op.state))
    rebuilt = Operator.from_state(state)
    assert jax.tree.structure(rebuilt) == jax.tree.structure(op), (
        "config round trip changed the pytree structure"
    )
    rebuilt_params = [
        leaf for leaf in jax.tree.leaves(rebuilt) if eqx.is_inexact_array(leaf)
    ]
    assert len(rebuilt_params) == len(params), (
        f"config round trip turned {len(params) - len(rebuilt_params)} learnable "
        "field(s) into constants — declare them with "
        "eqx.field(converter=jnp.asarray)"
    )
    _assert_tree_allclose(rebuilt(*inputs), expected, "config round trip")
