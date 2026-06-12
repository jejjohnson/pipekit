"""DiffraxForwardModel — protocol conformance, correctness, adjoints, SDE."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest


diffrax = pytest.importorskip("diffrax")

from pipekit_cycle import (
    BacksolveAdjoint,
    DACycle,
    DAState,
    ForwardModel,
    IdentityObs,
    RecursiveCheckpointAdjoint,
    SupportsAdjoint,
)
from pipekit_jax import DiffraxForwardModel


def _decay(t, y, args):
    return -y


class _RateDecay(eqx.Module):
    """Parametrised drift; the rate is a pytree leaf of the terms."""

    rate: jax.Array

    def __call__(self, t, y, args):
        return -self.rate * y


@pytest.fixture
def model():
    return DiffraxForwardModel(vector_field=_decay, dt0=0.01, dt=1.0)


class TestProtocolConformance:
    def test_forward_model(self, model):
        assert isinstance(model, ForwardModel)
        assert model.dt == 1.0
        assert model.state_signature is None

    def test_supports_adjoint(self, model):
        assert isinstance(model, SupportsAdjoint)
        assert model.adjoint == RecursiveCheckpointAdjoint()
        swapped = model.with_adjoint(BacksolveAdjoint())
        assert swapped.adjoint == BacksolveAdjoint()
        assert model.adjoint == RecursiveCheckpointAdjoint()  # unchanged


class TestODECorrectness:
    def test_matches_closed_form(self, model):
        y0 = jnp.array([1.0, 2.0])
        out = model.step(y0, 1.0)
        assert jnp.allclose(out, y0 * jnp.exp(-1.0), atol=1e-4)

    def test_as_dynamics_explicit_span(self, model):
        dyn = model.as_dynamics()
        y0 = jnp.array([1.0])
        assert jnp.allclose(dyn(y0, 0.0, 2.0), y0 * jnp.exp(-2.0), atol=1e-4)

    def test_gradient_through_step(self):
        def loss(rate):
            m = DiffraxForwardModel(vector_field=_RateDecay(rate), dt0=0.01)
            return m.step(jnp.array([1.0]), 1.0)[0]

        # d/dr exp(-r) at r=1 is -exp(-1)
        g = jax.grad(loss)(jnp.asarray(1.0))
        assert jnp.allclose(g, -jnp.exp(-1.0), atol=1e-3)

    def test_gradient_with_backsolve(self):
        # BacksolveAdjoint differentiates w.r.t. inputs of the solve, so
        # the parameter rides the terms pytree (an eqx.Module field), not
        # a Python closure.
        def loss(rate):
            m = DiffraxForwardModel(
                vector_field=_RateDecay(rate),
                dt0=0.01,
                adjoint=BacksolveAdjoint(),
            )
            return m.step(jnp.array([1.0]), 1.0)[0]

        g = jax.grad(loss)(jnp.asarray(1.0))
        assert jnp.allclose(g, -jnp.exp(-1.0), atol=1e-3)


class TestSDE:
    def test_sde_smoke_and_shapes(self):
        m = DiffraxForwardModel(
            vector_field=_decay,
            # diffrax >= 0.7: diagonal noise is a lineax operator mapping
            # the Brownian increment into state space.
            diffusion=lambda t, y, args: lx.DiagonalLinearOperator(
                0.1 * jnp.ones_like(y)
            ),
            sde_key=jax.random.key(0),
            solver=diffrax.Heun(),
            dt0=0.01,
        )
        out = m.step(jnp.ones(3), 1.0)
        assert out.shape == (3,)
        assert jnp.all(jnp.isfinite(out))

    def test_sde_requires_key(self):
        with pytest.raises(ValueError, match="sde_key"):
            DiffraxForwardModel(
                vector_field=_decay,
                diffusion=lambda t, y, args: lx.DiagonalLinearOperator(
                    jnp.ones_like(y)
                ),
            )


class TestCycleIntegration:
    def test_runs_inside_dacycle(self, model):
        class _NoOpAnalysis:
            def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
                return forecast

        da = DACycle(
            forward_model=model,
            obs_op=IdentityObs(),
            analysis_step=_NoOpAnalysis(),
            obs_source=None,
            n_steps=2,
        )
        carrier, state = da(jnp.array([1.0]), DAState())
        assert jnp.allclose(carrier, jnp.exp(-2.0), atol=1e-3)
        assert state.cycle_count == 2
