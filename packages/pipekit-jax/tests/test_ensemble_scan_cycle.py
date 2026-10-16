"""Tests for `pipekit_jax.EnsembleScanCycle`."""

from __future__ import annotations

import pytest


pytest.importorskip("equinox")
pytest.importorskip("jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit import StatefulOperator
from pipekit_jax import EnsembleScanCycle, JaxOperator, ScanCycle


class Forced(StatefulOperator, JaxOperator):
    """c ↦ a·c + s (state is a read-only forcing); learnable ``a``."""

    a: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, c, s):
        return self.a * c + s, s


class Counting(StatefulOperator):
    """c ↦ c + 1, s ↦ s + 1 (per-member state)."""

    def _apply(self, c, s):
        return c + 1.0, s + 1.0


M = 16
MEMBERS = jax.random.normal(jax.random.key(0), (M, 4))


def test_shared_state_matches_a_loop_over_members():
    cycle = ScanCycle(step_op=Forced(a=0.9), n_steps=10)
    s = jnp.asarray(0.1)
    out, s_out = EnsembleScanCycle(cycle=cycle)(MEMBERS, s)
    ref = jnp.stack([cycle(m, s)[0] for m in MEMBERS])
    np.testing.assert_allclose(out, ref, rtol=1e-6)
    np.testing.assert_allclose(s_out, s)  # shared state is read-only


def test_per_member_state_matches_a_loop_over_members():
    cycle = ScanCycle(step_op=Counting(), n_steps=5)
    states = jnp.arange(float(M))
    out, s_out = EnsembleScanCycle(cycle=cycle, state_axis=0)(MEMBERS, states)
    for j in range(M):
        c_ref, s_ref = cycle(MEMBERS[j], states[j])
        np.testing.assert_allclose(out[j], c_ref, rtol=1e-6)
        np.testing.assert_allclose(s_out[j], s_ref)


def test_run_returns_per_member_histories():
    cycle = ScanCycle(step_op=Forced(a=0.9), n_steps=10, history_stride=5)
    _, _, (hc, _) = EnsembleScanCycle(cycle=cycle).run(MEMBERS, jnp.asarray(0.0))
    assert hc.shape == (M, 2, 4)
    np.testing.assert_allclose(
        hc[3], cycle.run(MEMBERS[3], jnp.asarray(0.0))[2][0], rtol=1e-6
    )


def test_whole_ensemble_compiles_once():
    traces = []
    ens = EnsembleScanCycle(cycle=ScanCycle(step_op=Forced(a=0.9), n_steps=20))

    @eqx.filter_jit
    def run(e, members):
        traces.append(1)
        return e(members, jnp.asarray(0.0))[0]

    run(ens, MEMBERS)
    run(ens, MEMBERS + 1.0)
    assert len(traces) == 1


def test_gradient_reaches_the_shared_step_parameters():
    ens = EnsembleScanCycle(cycle=ScanCycle(step_op=Forced(a=0.9), n_steps=10))

    def loss(e):
        return jnp.sum(e(MEMBERS, jnp.asarray(0.0))[0] ** 2)

    grad = eqx.filter_grad(loss)(ens).cycle.step_op.a
    eps = 1e-3

    def with_a(a):
        return loss(EnsembleScanCycle(cycle=ScanCycle(step_op=Forced(a=a), n_steps=10)))

    np.testing.assert_allclose(
        grad, (with_a(0.9 + eps) - with_a(0.9 - eps)) / (2 * eps), rtol=1e-3
    )


def test_rejects_bad_state_axis_and_reports_config():
    with pytest.raises(ValueError, match="state_axis"):
        EnsembleScanCycle(cycle=ScanCycle(step_op=Counting(), n_steps=1), state_axis=1)
    cfg = EnsembleScanCycle(cycle=ScanCycle(step_op=Counting(), n_steps=1)).get_config()
    assert cfg["cycle"]["class"] == "ScanCycle" and cfg["state_axis"] is None


class Drifting(StatefulOperator):
    """Writes state: c ↦ c + s, s ↦ s + 1. Under a shared read-only state
    every step must see the *incoming* s."""

    def _apply(self, c, s):
        return c + s, s + 1.0


def test_shared_state_is_read_only_between_steps_too():
    s = jnp.asarray(0.5)
    ens = EnsembleScanCycle(
        cycle=ScanCycle(step_op=Drifting(), n_steps=4, history_stride=1)
    )
    out, s_out = ens(MEMBERS, s)
    np.testing.assert_allclose(
        out, MEMBERS + 4 * 0.5, rtol=1e-6
    )  # not + 0.5+1.5+2.5+3.5
    np.testing.assert_allclose(s_out, s)
    _, _, (hc, _) = ens.run(MEMBERS, s)
    np.testing.assert_allclose(hc[:, -1], MEMBERS + 4 * 0.5, rtol=1e-6)
