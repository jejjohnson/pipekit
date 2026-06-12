"""truncated_scan gradient semantics + the spec -> diffrax mapping."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from pipekit_cycle import (
    BacksolveAdjoint,
    DirectAdjoint,
    ImplicitAdjoint,
    RecursiveCheckpointAdjoint,
    TruncatedAdjoint,
)
from pipekit_jax import to_diffrax_adjoint, truncated_scan


jax.config.update("jax_enable_x64", True)


def _step(theta):
    """Mildly expanding linear step so history matters to the gradient."""

    def f(carry, x):
        new = jnp.tanh(theta) * 1.5 * carry + x
        return new, new

    return f


def _loss_full(theta, init, xs):
    _, ys = jax.lax.scan(_step(theta), init, xs)
    return jnp.sum(ys[-1] ** 2)


def _loss_truncated(k):
    def loss(theta, init, xs):
        _, ys = truncated_scan(_step(theta), init, xs, k=k)
        return jnp.sum(ys[-1] ** 2)

    return loss


@pytest.fixture
def problem():
    init = jnp.array([0.3, -0.2])
    xs = jnp.linspace(-1.0, 1.0, 8)[:, None] * jnp.ones((8, 2))
    return jnp.asarray(0.7), init, xs


def test_forward_values_identical_to_scan(problem):
    theta, init, xs = problem
    _, ys_plain = jax.lax.scan(_step(theta), init, xs)
    _, ys_trunc = truncated_scan(_step(theta), init, xs, k=2)
    assert jnp.allclose(ys_plain, ys_trunc)


def test_k_geq_length_matches_full_gradient(problem):
    theta, init, xs = problem
    g_full = jax.grad(_loss_full)(theta, init, xs)
    g_trunc = jax.grad(_loss_truncated(k=len(xs)))(theta, init, xs)
    assert jnp.allclose(g_full, g_trunc)


def test_k1_matches_manual_one_step_construction(problem):
    theta, init, xs = problem

    def manual_one_step(theta):
        # All-but-last transitions under stop_gradient, then one live step.
        carry = init
        for x in xs[:-1]:
            carry, _ = _step(theta)(carry, x)
        carry = jax.lax.stop_gradient(carry)
        _, y = _step(theta)(carry, xs[-1])
        return jnp.sum(y**2)

    g_manual = jax.grad(manual_one_step)(theta)
    g_trunc = jax.grad(_loss_truncated(k=1))(theta, init, xs)
    assert jnp.allclose(g_manual, g_trunc)


def test_truncated_gradient_is_smaller_on_expanding_dynamics(problem):
    """On expanding dynamics the truncated gradient is tamer than the full one."""
    theta, init, xs = problem
    g_full = jnp.abs(jax.grad(_loss_full)(theta, init, xs))
    g_k1 = jnp.abs(jax.grad(_loss_truncated(k=1))(theta, init, xs))
    assert g_k1 < g_full


def test_init_gradient_blocked_when_truncated(problem):
    """With k < T no gradient reaches the initial carry."""
    theta, init, xs = problem
    g_init = jax.grad(lambda i: _loss_truncated(k=1)(theta, i, xs))(init)
    assert jnp.allclose(g_init, 0.0)
    g_init_full = jax.grad(lambda i: _loss_full(theta, i, xs))(init)
    assert not jnp.allclose(g_init_full, 0.0)


def test_invalid_k_raises():
    with pytest.raises(ValueError, match="k >= 1"):
        truncated_scan(lambda c, x: (c, c), 0.0, jnp.ones(3), k=0)


class TestToDiffraxAdjoint:
    def test_mapping(self):
        diffrax = pytest.importorskip("diffrax")
        assert isinstance(
            to_diffrax_adjoint(RecursiveCheckpointAdjoint(checkpoints=8)),
            diffrax.RecursiveCheckpointAdjoint,
        )
        assert isinstance(to_diffrax_adjoint(DirectAdjoint()), diffrax.DirectAdjoint)
        assert isinstance(
            to_diffrax_adjoint(BacksolveAdjoint()), diffrax.BacksolveAdjoint
        )
        assert isinstance(
            to_diffrax_adjoint(ImplicitAdjoint()), diffrax.ImplicitAdjoint
        )

    def test_raw_diffrax_adjoint_passes_through(self):
        diffrax = pytest.importorskip("diffrax")
        raw = diffrax.DirectAdjoint()
        assert to_diffrax_adjoint(raw) is raw

    def test_truncated_rejected_at_solve_layer(self):
        pytest.importorskip("diffrax")
        with pytest.raises(ValueError, match="outer rollout"):
            to_diffrax_adjoint(TruncatedAdjoint(k=2))

    def test_foreign_spec_matched_structurally(self):
        """Identically-named specs from other libraries are accepted."""
        diffrax = pytest.importorskip("diffrax")

        class RecursiveCheckpointAdjoint:
            checkpoints = 4

        mapped = to_diffrax_adjoint(RecursiveCheckpointAdjoint())
        assert isinstance(mapped, diffrax.RecursiveCheckpointAdjoint)

    def test_unknown_spec_rejected(self):
        pytest.importorskip("diffrax")
        with pytest.raises(ValueError, match="Unrecognised"):
            to_diffrax_adjoint(object())
