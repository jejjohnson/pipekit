"""Tests for `pipekit_jax.ScanCycle` and `pipekit_jax.Rollout`."""

from __future__ import annotations

import pytest


pytest.importorskip("equinox")
pytest.importorskip("jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit import Operator, Sequential, StatefulOperator
from pipekit_cycle import Cycle
from pipekit_jax import JaxOperator, Rollout, ScanCycle, truncated_scan


class Relax(StatefulOperator):
    """c ↦ 0.9c + 1;  s ↦ s + Σc.   A plain (non-pytree) stateful step."""

    def _apply(self, c, s):
        return 0.9 * c + 1.0, s + jnp.sum(c)


class Decay(StatefulOperator, JaxOperator):
    """c ↦ a·c with a learnable ``a``; state passes through."""

    a: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, c, s):
        return jax.tree.map(lambda x: self.a * x, c), s


class Damp(JaxOperator):
    """Carrier-only step c ↦ a·c (learnable ``a``)."""

    a: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, c):
        return jax.tree.map(lambda x: self.a * x, c)


class Oscillator(eqx.Module):
    """A `ForwardModel`: explicit Euler for ẍ = -ω²x, state (x, v)."""

    omega: jax.Array
    dt: float = eqx.field(static=True, default=0.01)
    state_signature: None = eqx.field(static=True, default=None)

    def step(self, state, dt):
        x, v = state["x"], state["v"]
        return {"x": x + dt * v, "v": v - dt * self.omega**2 * x}


class Spectral(eqx.Module):
    """A custom pytree carrier (e.g. Fourier coefficients)."""

    hat: jax.Array


C0 = jnp.arange(3.0)
S0 = jnp.asarray(0.0)


def _close(a, b):
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-6), a, b)


# ---------------------------------------------------------------------
# Parity with pipekit_cycle.Cycle
# ---------------------------------------------------------------------


@pytest.mark.parametrize(("n", "k"), [(10, 3), (9, 3), (1, 1), (5, 1), (0, 2), (7, 10)])
def test_matches_cycle_final_state_and_history(n, k):
    ref = Cycle(Relax(), n, save_history=True, history_stride=k)
    c_ref, s_ref = ref(C0, S0)
    cyc = ScanCycle(step_op=Relax(), n_steps=n, history_stride=k)

    _close(eqx.filter_jit(cyc)(C0, S0), (c_ref, s_ref))
    c, s, (hc, hs) = eqx.filter_jit(cyc.run)(C0, S0)
    _close((c, s), (c_ref, s_ref))
    assert hc.shape[0] == len(ref.history) == -(-n // k)
    if ref.history:
        _close(hc, jnp.stack([h[0] for h in ref.history]))
        _close(hs, jnp.stack([h[1] for h in ref.history]))


def test_forward_model_step_matches_cycle():
    model = Oscillator(omega=jnp.asarray(2.0))
    state0 = {"x": jnp.asarray(1.0), "v": jnp.asarray(0.0)}
    ref, _ = Cycle(model, 50)(state0, None)
    out, _ = eqx.filter_jit(ScanCycle(step_op=model, n_steps=50))(state0, None)
    _close(out, ref)


def test_rejects_invalid_arguments():
    with pytest.raises(TypeError, match="StatefulOperator or a ForwardModel"):
        ScanCycle(step_op=object(), n_steps=1)
    with pytest.raises(ValueError, match="n_steps"):
        ScanCycle(step_op=Relax(), n_steps=-1)
    with pytest.raises(ValueError, match="history_stride"):
        ScanCycle(step_op=Relax(), n_steps=1, history_stride=0)
    with pytest.raises(ValueError, match="truncate"):
        ScanCycle(step_op=Relax(), n_steps=1, truncate=0)


def test_composes_in_a_stateful_sequential():
    pipe = Sequential([ScanCycle(step_op=Relax(), n_steps=4)])
    _close(pipe(C0, S0), Cycle(Relax(), 4)(C0, S0))


def test_get_config_mirrors_cycle_keys():
    cfg = ScanCycle(step_op=Relax(), n_steps=3).get_config()
    assert set(cfg) >= {"step_op", "n_steps", "history_stride"}
    assert cfg["step_op"]["class"] == "Relax"


# ---------------------------------------------------------------------
# Compile-time scaling: one scan regardless of n_steps
# ---------------------------------------------------------------------


def _jaxpr_size(fn, *args) -> int:
    return len(str(jax.make_jaxpr(fn)(*args)))


def test_traced_program_size_is_independent_of_n_steps():
    small = _jaxpr_size(ScanCycle(step_op=Relax(), n_steps=10), C0, S0)
    large = _jaxpr_size(ScanCycle(step_op=Relax(), n_steps=1000), C0, S0)
    assert abs(large - small) < 50  # only the literal `length=` changes
    # ... whereas the Python-loop Cycle unrolls.
    unrolled = _jaxpr_size(Cycle(Relax(), 100), C0, S0)
    assert unrolled > 10 * small


# ---------------------------------------------------------------------
# Pytree carriers
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "carrier",
    [
        {"u": jnp.ones(3), "v": jnp.zeros(2)},
        Spectral(hat=jnp.ones(4, dtype=jnp.complex64)),
    ],
    ids=["dict", "eqx-module"],
)
def test_history_stacks_arbitrary_pytree_carriers(carrier):
    step = Decay(a=0.5)
    cyc = ScanCycle(step_op=step, n_steps=6, history_stride=2)
    c, _, (hc, _) = eqx.filter_jit(cyc.run)(carrier, None)
    assert jax.tree.structure(hc) == jax.tree.structure(carrier)
    for leaf, final in zip(jax.tree.leaves(hc), jax.tree.leaves(c), strict=True):
        assert leaf.shape[0] == 3
        np.testing.assert_allclose(leaf[-1], final * 2)  # recorded after step 5


def test_rollout_handles_pytree_carriers():
    traj = Rollout(step=Damp(a=0.5), n_steps=4, save_every=2)(Spectral(hat=jnp.ones(2)))
    assert isinstance(traj, Spectral)
    np.testing.assert_allclose(traj.hat[:, 0], [1.0, 0.25, 0.0625])


# ---------------------------------------------------------------------
# Gradients: leaves, checkpointing, truncation
# ---------------------------------------------------------------------


def _loss(cyc):
    c, _ = cyc(C0 + 1.0, S0)
    return jnp.sum(c**2)


def test_filter_grad_reaches_step_leaves_and_matches_fd():
    cyc = ScanCycle(step_op=Decay(a=0.9), n_steps=20)
    g = eqx.filter_grad(_loss)(cyc).step_op.a
    eps = 1e-3

    def with_a(a):
        return _loss(ScanCycle(step_op=Decay(a=a), n_steps=20))

    fd = (with_a(0.9 + eps) - with_a(0.9 - eps)) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-3)


@pytest.mark.parametrize("n", [20, 23, 1])  # perfect square-ish, remainder, tiny
@pytest.mark.parametrize("truncate", [None, 5])
def test_checkpoint_changes_values_and_gradients_not_at_all(n, truncate):
    def make(ck):
        return ScanCycle(
            step_op=Decay(a=0.9), n_steps=n, checkpoint=ck, truncate=truncate
        )

    _close(make(False)(C0, S0), make(True)(C0, S0))
    _close(
        eqx.filter_grad(_loss)(make(False)).step_op.a,
        eqx.filter_grad(_loss)(make(True)).step_op.a,
    )


def test_checkpoint_in_run_and_rollout_keeps_values_and_gradients():
    def run_loss(ck):
        cyc = ScanCycle(
            step_op=Decay(a=0.9), n_steps=12, history_stride=4, checkpoint=ck
        )
        return lambda c0: jnp.sum(cyc.run(c0, S0)[2][0] ** 2)

    def roll_loss(ck):
        r = Rollout(step=Damp(a=0.9), n_steps=12, save_every=4, checkpoint=ck)
        return lambda c0: jnp.sum(r(c0) ** 2)

    for make in (run_loss, roll_loss):
        _close(make(False)(C0), make(True)(C0))
        _close(jax.grad(make(False))(C0), jax.grad(make(True))(C0))


def test_checkpoint_reduces_gradient_memory():
    u0 = jnp.ones((64, 64))

    def temp_mib(ck):
        def loss(op, u):
            return jnp.sum(
                ScanCycle(step_op=op, n_steps=400, checkpoint=ck)(u, None)[0] ** 2
            )

        grad = eqx.filter_grad(loss)
        compiled = jax.jit(lambda op, u: grad(op, u)).lower(Decay(a=0.9), u0).compile()
        analysis = compiled.memory_analysis()
        if analysis is None:
            pytest.skip("backend reports no memory analysis")
        return analysis.temp_size_in_bytes

    # 400 carries of 16 KiB vs ~2·√400 = 40: expect well over a 4x reduction.
    assert temp_mib(True) * 4 < temp_mib(False)


def test_truncate_cuts_gradients_through_early_carries():
    def grad_c0(truncate):
        cyc = ScanCycle(step_op=Decay(a=0.9), n_steps=10, truncate=truncate)
        return jax.grad(lambda c0: jnp.sum(cyc(c0, S0)[0]))(C0)

    full = jax.grad(
        lambda c0: jnp.sum(ScanCycle(step_op=Decay(a=0.9), n_steps=10)(c0, S0)[0])
    )(C0)
    np.testing.assert_allclose(grad_c0(3), 0.0)  # c0 only reaches the output via step 0
    _close(grad_c0(10), full)
    _close(grad_c0(None), full)


def test_truncate_matches_truncated_scan_on_parameter_gradients():
    k, n = 3, 12

    def via_cycle(a):
        c, _ = ScanCycle(step_op=Decay(a=a), n_steps=n, truncate=k)(C0 + 1.0, S0)
        return jnp.sum(c**2)

    def via_truncated_scan(a):
        c, _ = truncated_scan(lambda c, _: (a * c, None), C0 + 1.0, jnp.arange(n), k=k)
        return jnp.sum(c**2)

    np.testing.assert_allclose(
        jax.grad(via_cycle)(0.9), jax.grad(via_truncated_scan)(0.9), rtol=1e-6
    )


# ---------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------


def test_rollout_trajectory_matches_python_loop():
    step = Damp(a=0.5)
    traj = eqx.filter_jit(Rollout(step=step, n_steps=6, save_every=2))(C0)
    ref = [C0]
    c = C0
    for i in range(1, 7):
        c = step(c)
        if i % 2 == 0:
            ref.append(c)
    _close(traj, jnp.stack(ref))
    no_init = Rollout(step=step, n_steps=6, save_every=2, include_initial=False)(C0)
    _close(no_init, jnp.stack(ref[1:]))


def test_rollout_accepts_a_forward_model():
    model = Oscillator(omega=jnp.asarray(2.0))
    s0 = {"x": jnp.asarray(1.0), "v": jnp.asarray(0.0)}
    traj = Rollout(step=model, n_steps=50, save_every=50)(s0)
    ref, _ = Cycle(model, 50)(s0, None)
    _close(jax.tree.map(lambda t: t[-1], traj), ref)


def test_rollout_rejects_non_multiple_save_every():
    with pytest.raises(ValueError, match="multiple"):
        Rollout(step=Damp(a=0.5), n_steps=5, save_every=2)


def test_rollout_is_a_pipekit_operator_with_config():
    r = Rollout(step=Damp(a=0.5), n_steps=4, save_every=2)
    assert isinstance(r, Operator)
    assert r.get_config()["step"]["class"] == "Damp"


# ---------------------------------------------------------------------
# Review fixes (#69)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("stride", [1, 3])
def test_truncated_history_keeps_each_entrys_local_gradient(stride):
    # truncated_scan cuts the carry but leaves each output's own gradient:
    # a trajectory loss over run()'s history must match it.
    k, n = 2, 9
    c0 = C0 + 1.0

    def via_run(a):
        cyc = ScanCycle(
            step_op=Decay(a=a), n_steps=n, history_stride=stride, truncate=k
        )
        hist = cyc.run(c0, S0)[2][0]
        return jnp.sum(hist**2)

    def via_truncated_scan(a):
        _, ys = truncated_scan(lambda c, _: (a * c, a * c), c0, jnp.arange(n), k=k)
        return jnp.sum(ys[::stride] ** 2)

    np.testing.assert_allclose(via_run(0.9), via_truncated_scan(0.9), rtol=1e-6)
    np.testing.assert_allclose(
        jax.grad(via_run)(0.9), jax.grad(via_truncated_scan)(0.9), rtol=1e-6
    )


@pytest.mark.parametrize("value", [jnp.arange(3), jnp.array([True, False])])
def test_zero_step_histories_keep_the_carrier_dtype(value):
    class Same(StatefulOperator):
        def _apply(self, c, s):
            return c, s

    _, _, (hc, _) = ScanCycle(step_op=Same(), n_steps=0).run(value, None)
    assert hc.shape == (0, *value.shape) and hc.dtype == value.dtype

    traj = Rollout(step=Damp(a=1.0), n_steps=0)(value)
    assert traj.dtype == value.dtype and traj.shape == (1, *value.shape)
    empty = Rollout(step=Damp(a=1.0), n_steps=0, include_initial=False)(value)
    assert empty.dtype == value.dtype and empty.shape == (0, *value.shape)


def test_forward_model_wrappers_are_reported_as_pickle_hazards():
    from pipekit import check_pickleable

    model = Oscillator(omega=jnp.asarray(2.0))
    assert check_pickleable(ScanCycle(step_op=model, n_steps=2))
    assert check_pickleable(Rollout(step=model, n_steps=2))
    assert check_pickleable(Rollout(step=Damp(a=0.5), n_steps=2)) == []


def test_rollout_rejects_a_non_operator_step():
    with pytest.raises(TypeError, match="Operator or a ForwardModel"):
        Rollout(step=lambda c: c, n_steps=1)


def test_rollout_rejects_a_stateful_step():
    with pytest.raises(TypeError, match="ScanCycle"):
        Rollout(step=Relax(), n_steps=1)


def test_rollout_rejects_a_sequential_holding_a_stateful_step():
    """A composite is stateful when any step is; reject it at construction too."""
    with pytest.raises(TypeError, match="Sequential is stateful"):
        Rollout(step=Sequential([Relax()]), n_steps=1)
