"""Tests for `pipekit_jax.to_jax` — a KernelOp as a JaxOperator (#106)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest


pa = pytest.importorskip("pipekit_array")

from pipekit_array import DimArray, KernelOp, KernelSpec
from pipekit_jax import JaxKernelOp, JaxOperator, to_jax
from pipekit_jax.testing import check_jax_operator


@pytest.fixture(autouse=True)
def x64() -> Iterator[None]:
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def matched_filter_score(x: Any, mu: Any, sigma: Any, t: Any) -> Any:
    """alpha per pixel: x (..., C), mu (C,), sigma (C, C), t (C,) -> (...)."""
    # w = Sigma^-1 t, solved, never inverted
    w = jnp.linalg.solve(sigma, t)
    # alpha = (x - mu)^T w / (t^T w)
    return ((x - mu) @ w) / (t @ w)


def _problem(seed: int = 0) -> tuple[DimArray, dict[str, Any]]:
    """A 4-band toy scene and a well-conditioned background."""
    rng = np.random.default_rng(seed)
    scene = DimArray(jnp.asarray(rng.random((4, 6, 5))), ("band", "y", "x"))
    a = rng.random((4, 4))
    params = {
        "mu": rng.random(4),
        "sigma": a @ a.T + 4.0 * np.eye(4),
        "t": np.array([0.1, -0.2, 0.3, 0.05]),
    }
    return scene, params


def _score(params: dict[str, Any]) -> KernelOp:
    return KernelOp(
        matched_filter_score,
        KernelSpec("(c)->()", backends=("jax",)),
        core={"c": "band"},
        **params,
    )


def test_leaves_are_exactly_the_array_params() -> None:
    _, params = _problem()
    op = KernelOp(
        matched_filter_score,
        KernelSpec("(c)->()", backends=("jax",)),
        core={"c": "band"},
        mu=params["mu"],
        sigma=params["sigma"],
        t=params["t"],
    )
    jop = to_jax(op)
    assert isinstance(jop, JaxKernelOp)
    assert isinstance(jop, JaxOperator)
    assert sorted(jop.params) == ["mu", "sigma", "t"]
    leaves = jax.tree_util.tree_leaves(jop)
    assert len(leaves) == 3
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def test_config_params_are_static() -> None:
    def nd(a: Any, b: Any, eps: float, scale: Any) -> Any:
        return scale * (a - b) / (a + b + eps)

    jop = to_jax(KernelOp(nd, "(),()->()", core={}, eps=1e-10, scale=np.float64(2.0)))
    assert dict(jop.static_params) == {"eps": 1e-10}
    assert sorted(jop.params) == ["scale"]
    assert len(jax.tree_util.tree_leaves(jop)) == 1


def test_equals_the_kernel_op() -> None:
    scene, params = _problem()
    op = _score(params)
    np.testing.assert_allclose(
        np.asarray(to_jax(op)(scene).data), np.asarray(op(scene).data), rtol=1e-12
    )
    assert to_jax(op)(scene).dims == ("y", "x")


def test_check_jax_operator() -> None:
    scene, params = _problem()
    check_jax_operator(to_jax(_score(params)), scene)


def test_filter_jit_traces_once() -> None:
    """PK-K3 on the JAX path: same-shape carriers, one trace."""
    _, params = _problem()
    jop = to_jax(_score(params))
    traces = 0

    @eqx.filter_jit
    def run(o: JaxKernelOp, x: DimArray) -> Any:
        nonlocal traces
        traces += 1
        return o(x).data

    for seed in range(4):
        run(jop, _problem(seed)[0])
    assert traces == 1


def test_gradient_with_respect_to_the_target() -> None:
    """d loss / dt from filter_grad matches central finite differences."""
    scene, params = _problem()
    jop = to_jax(_score(params))

    def loss(o: JaxKernelOp) -> Any:
        alpha = o(scene).data
        return jnp.sum(alpha**2)

    grads = eqx.filter_grad(loss)(jop)
    dt = np.asarray(grads.params["t"])
    assert np.all(np.isfinite(dt))
    step = 1e-6
    numeric = np.zeros(4)
    for k in range(4):
        bump = np.zeros(4)
        bump[k] = step
        plus = eqx.tree_at(lambda o: o.params["t"], jop, jop.params["t"] + bump)
        minus = eqx.tree_at(lambda o: o.params["t"], jop, jop.params["t"] - bump)
        numeric[k] = (float(loss(plus)) - float(loss(minus))) / (2 * step)
    np.testing.assert_allclose(dt, numeric, rtol=1e-5, atol=1e-8)


def test_geometry_and_input_count() -> None:
    field = DimArray(jnp.outer(jnp.arange(6.0), jnp.ones(5)), ("y", "x"))

    def gradient(f: Any, *, spacing: Any) -> tuple[Any, Any]:
        gy, gx = jnp.gradient(f, axis=(-2, -1))
        return gy / spacing[0], gx / spacing[1]

    op = KernelOp(
        gradient,
        "(y,x)->(y,x),(y,x)",
        core={"y": "y", "x": "x"},
        geometry={"spacing": lambda _: jnp.asarray([2.0, 4.0])},
    )
    gy, _ = to_jax(op)(field)
    np.testing.assert_allclose(np.asarray(gy.data), 0.5)
    with pytest.raises(TypeError, match="input"):
        to_jax(op)(field, field)


def test_rejects_non_kernel_ops_and_unhashable_params() -> None:
    with pytest.raises(TypeError, match="KernelOp"):
        to_jax(object())
    op = KernelOp(lambda x, opts: x, "()->()", core={}, opts={"a": 1})
    with pytest.raises(TypeError, match="hashable"):
        to_jax(op)


def test_dimarray_parameter_is_a_leaf() -> None:
    w = DimArray(np.ones(2), ("q",))
    op = KernelOp(lambda x, w: x * w.data.sum(), "()->()", core={}, w=w)
    jop = to_jax(op)
    assert sorted(jop.params) == ["w"]
    assert jop.static_params == ()
    assert isinstance(jop.params["w"].data, jax.Array)
