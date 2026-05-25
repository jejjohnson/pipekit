"""Cross-package integration: pipekit-train ↔ pipekit-cycle handoff.

The headline differentiator vs. a backend-only trainer like eqx_trainer:
a model trained via `pipekit_train.TrainingLoop` over a
`SimulationDataset(forward_model=…)` drops straight into
`pipekit_cycle.NeuralForward(model_op=…)` and runs inside any
`pipekit_cycle.Cycle` / `DACycle`. The same operator that came out of
training is the operator that goes into inference — one swap, no
rewrite.

Tested end-to-end: train tiny emulator on a synthetic forward model,
wrap as NeuralForward, step a Cycle, verify the trajectory is sane.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


pytest.importorskip("equinox")
pytest.importorskip("jax")

import equinox as eqx
import jax
from pipekit import Operator
from pipekit_cycle import Cycle, NeuralForward
from pipekit_train import MSE, SimulationDataset, TrainingLoop
from pipekit_train.adapters.equinox import _EquinoxModelOp


# ---------------------------------------------------------------------
# Synthetic forward model — a 4-D state advected by a deterministic map
# ---------------------------------------------------------------------


class _DampedOscillator:
    """Tiny ForwardModel: 2-D harmonic oscillator with damping.

    State is ``[x, v]``; one step advances by Euler with dt=0.05:
    x' = x + dt*v;  v' = v - dt*(omega^2*x + gamma*v).
    Stable for the parameter ranges used in the test.
    """

    dt: float = 0.05
    state_signature: Any = None

    def __init__(self, omega: float = 2.0, gamma: float = 0.1) -> None:
        self.omega = omega
        self.gamma = gamma

    def step(self, state: Any, dt: float) -> Any:
        x, v = state[0], state[1]
        x_new = x + dt * v
        v_new = v - dt * (self.omega**2 * x + self.gamma * v)
        return np.stack([x_new, v_new])


class _GaussianPrior(Operator):
    """Sample initial states ``[x0, v0]`` from a Gaussian."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def _apply(self, seed: int | None = None) -> Any:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(2).astype(np.float32) * self.scale

    def __call__(self, *, seed: int | None = None) -> Any:
        return self._apply(seed=seed)


# ---------------------------------------------------------------------
# Test — train an emulator, drop it into a Cycle, verify shape + values
# ---------------------------------------------------------------------


def test_simulation_dataset_train_emulator_handoff_into_cycle():
    """End-to-end: SimulationDataset → TrainingLoop.run() → NeuralForward → Cycle."""
    # 1. Expensive forward model (mock — cheap, but represents the
    #    physics in a real workflow).
    expensive = _DampedOscillator(omega=2.0, gamma=0.1)
    prior = _GaussianPrior(scale=1.0)

    # 2. SimulationDataset yields (state_t, state_{t+dt}) pairs.
    train_ds = SimulationDataset(
        forward_model=expensive,
        prior=prior,
        n_samples=200,
        seed=42,
        split="train",
    )

    # 3. The neural emulator — a small MLP mapping R^2 → R^2.
    mlp = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=32,
        depth=2,
        key=jax.random.key(0),
    )
    emulator_op = _EquinoxModelOp(mlp)

    # 4. Train.
    loop = TrainingLoop(
        model_op=emulator_op,
        dataset=train_ds,
        loss=MSE(),
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=200,
        batch_size=16,
        backend="equinox",
        seed=0,
    )
    trained_op, artifact = loop.run()

    # 5. The trained model is a pipekit.Operator that satisfies the
    #    contract `NeuralForward` expects.
    assert isinstance(trained_op, Operator)

    # 6. Drop into NeuralForward. The wrapped emulator IS the new
    #    forward model for the cycle.
    emulator_forward = NeuralForward(model_op=trained_op, dt=expensive.dt)

    # 7. Step a Cycle for 10 timesteps from a fixed initial state.
    initial = np.array([1.0, 0.0], dtype=np.float32)
    cycle = Cycle(step_op=emulator_forward, n_steps=10, save_history=True)
    final_state, _ = cycle(initial, None)

    # 8. Sanity: the trajectory should be 2-D and finite.
    assert final_state.shape == (2,)
    assert np.all(np.isfinite(np.asarray(final_state)))
    assert len(cycle.history) == 10

    # 9. Sanity on the artifact — backend_info confirms the adapter
    #    ran end-to-end and reports training metrics.
    info = artifact.backend_info
    assert info["backend"] == "equinox"
    assert info["final_step"] == 200
    assert "mse" in info["final_metrics"]


def test_trained_emulator_round_trips_through_neural_forward_dt():
    """Cycle handoff respects the model's training cadence (dt).

    `NeuralForward.dt` is the cadence the emulator was trained for;
    passing a different dt to step() is accepted for protocol
    conformance but the emulator ignores it.
    """
    expensive = _DampedOscillator()
    train_ds = SimulationDataset(
        forward_model=expensive,
        prior=_GaussianPrior(),
        n_samples=64,
        seed=0,
    )
    emulator_op = _EquinoxModelOp(
        eqx.nn.MLP(2, 2, width_size=8, depth=2, key=jax.random.key(0))
    )
    loop = TrainingLoop(
        model_op=emulator_op,
        dataset=train_ds,
        loss=MSE(),
        max_steps=50,
        batch_size=8,
        backend="equinox",
        optimizer_config={"name": "adam", "lr": 1e-2},
        seed=0,
    )
    trained_op, _ = loop.run()

    fwd = NeuralForward(model_op=trained_op, dt=expensive.dt)
    assert fwd.dt == expensive.dt
    out1 = fwd.step(np.array([1.0, 0.0], dtype=np.float32), dt=expensive.dt)
    # Calling with a different dt yields the same output: the
    # neural emulator's cadence is fixed at construction time.
    out2 = fwd.step(np.array([1.0, 0.0], dtype=np.float32), dt=99.0)
    np.testing.assert_array_almost_equal(np.asarray(out1), np.asarray(out2))
