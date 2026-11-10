"""The adjoint forwarders warn and agree with integrix.adjoint."""

from __future__ import annotations

import sys
import warnings

import jax
import jax.numpy as jnp
import pytest
from pipekit_cycle import BacksolveAdjoint, RecursiveCheckpointAdjoint
from pipekit_jax import to_diffrax_adjoint, truncated_scan


def _step(theta):
    def f(carry, x):
        new = jnp.tanh(theta) * 1.5 * carry + x
        return new, new

    return f


def test_truncated_scan_warns_and_names_integrix():
    xs = jnp.ones((4, 2))
    with pytest.warns(DeprecationWarning, match=r"integrix\.adjoint\.truncated_scan"):
        truncated_scan(_step(0.7), jnp.zeros(2), xs, k=2)


def test_truncated_scan_matches_integrix():
    ix_adjoint = pytest.importorskip("integrix.adjoint")
    init = jnp.array([0.3, -0.2])
    xs = jnp.linspace(-1.0, 1.0, 6)[:, None] * jnp.ones((6, 2))

    def loss(fn, theta):
        _, ys = fn(_step(theta), init, xs, k=2)
        return jnp.sum(ys[-1] ** 2)

    with pytest.warns(DeprecationWarning):
        value = loss(truncated_scan, 0.7)
        grad = jax.grad(lambda t: loss(truncated_scan, t))(0.7)
    assert value == loss(ix_adjoint.truncated_scan, 0.7)
    assert grad == jax.grad(lambda t: loss(ix_adjoint.truncated_scan, t))(0.7)


def test_to_diffrax_adjoint_warns_and_forwards():
    ix_adjoint = pytest.importorskip("integrix.adjoint")
    spec = RecursiveCheckpointAdjoint(checkpoints=8)
    with pytest.warns(
        DeprecationWarning, match=r"integrix\.adjoint\.to_diffrax_adjoint"
    ):
        mapped = to_diffrax_adjoint(spec)
    assert mapped == ix_adjoint.to_diffrax_adjoint(spec)


def test_to_diffrax_adjoint_without_integrix_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "integrix.adjoint", None)
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ImportError, match=r"pipekit-jax\[diffrax\]"),
    ):
        to_diffrax_adjoint(BacksolveAdjoint())


def test_diffrax_forward_model_does_not_warn():
    pytest.importorskip("diffrax")
    from pipekit_jax import DiffraxForwardModel

    model = DiffraxForwardModel(vector_field=lambda t, y, args: -y, dt0=0.1, dt=1.0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=r"pipekit_jax\.", category=DeprecationWarning
        )
        model.step(jnp.ones(2), 1.0)
