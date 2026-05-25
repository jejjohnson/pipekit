"""End-to-end smoke tests for the three worked examples in
``docs/design/examples/``.

The example markdowns are conceptual sketches that reference external
libraries (geocatalog, pyrox, plumax, somax). These tests exercise
the *same operator shapes* against synthetic stand-ins so the v0.1
release ships with concrete proof each example pattern works:

- ``examples/direct.md`` — Direct supervised. ``IterableDataset``
  feeding a model + ``MSE`` loss. (CatalogDataset is geocatalog-
  blocked and isn't in v0.1.)
- ``examples/emulator.md`` — Emulator training. ``SimulationDataset``
  + ``MSE`` + Equinox MLP; the trained model is wrapped as
  ``NeuralForward`` and demonstrated to step.
- ``examples/amortized.md`` — Amortized inference. Uses
  ``SimulationDataset(obs_op=…)`` so the network learns to invert
  *observations* rather than latent state. The conditional density
  estimator is a small MLP whose loss is provided by the user (a
  Gaussian log-likelihood head); pyrox isn't a workspace dep, so
  the smoke test stops at "the SBI data shape works under
  TrainingLoop without crashing".

Each test trains for a small number of steps and asserts the loss
decreased — the same convergence assertion the design doc claims.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


pytest.importorskip("equinox")
pytest.importorskip("jax")

import equinox as eqx
import jax
import jax.numpy as jnp
from pipekit import Operator
from pipekit_cycle import NeuralForward
from pipekit_train import (
    MSE,
    IterableDataset,
    SimulationDataset,
    TrainingLoop,
)
from pipekit_train.adapters.equinox import _EquinoxModelOp


# ---------------------------------------------------------------------
# Shared synthetic shapes
# ---------------------------------------------------------------------


class _ToyForward:
    """A 2-D toy forward model used by both emulator + amortized smoke."""

    dt: float = 0.1
    state_signature: Any = None

    def step(self, state: Any, dt: float) -> Any:
        x, y = state[0], state[1]
        return np.stack([0.95 * x + 0.1 * y, 0.95 * y - 0.1 * x])


class _GaussianPrior(Operator):
    def __init__(self, dim: int = 2) -> None:
        self.dim = dim

    def _apply(self, seed: int | None = None) -> Any:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)

    def __call__(self, *, seed: int | None = None) -> Any:
        return self._apply(seed=seed)


def _toy_mlp(in_size: int, out_size: int, key: int = 0) -> eqx.nn.MLP:
    return eqx.nn.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=16,
        depth=2,
        key=jax.random.key(key),
    )


def _first_and_last_loss(loop: TrainingLoop) -> tuple[float, float]:
    """Run loop with a capturing writer, return (first, last) MSE values."""
    written: list[dict[str, float]] = []

    class _Capture:
        def write(self, step: int, metrics: dict[str, float]) -> None:
            written.append(dict(metrics))

        def close(self) -> None:
            pass

    loop.metric_writer = _Capture()
    loop.run()
    assert written, "no metrics were captured — loop didn't log"
    return written[0]["mse"], written[-1]["mse"]


# ---------------------------------------------------------------------
# 1. Direct supervised — examples/direct.md
# ---------------------------------------------------------------------


def test_direct_supervised_smoke():
    """`IterableDataset` + `MSE` + `TrainingLoop.run()` converges."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1.0, 1.0, size=(128, 4)).astype(np.float32)
    # Linear ground truth: y = sum(x_i) + tiny noise
    ys = (xs.sum(axis=1, keepdims=True) + 0.01 * rng.standard_normal((128, 1))).astype(
        np.float32
    )
    dataset = IterableDataset(
        source=list(zip(xs, ys, strict=True)),
        content_hash="direct-smoke-128",
    )

    loop = TrainingLoop(
        model_op=_EquinoxModelOp(_toy_mlp(in_size=4, out_size=1)),
        dataset=dataset,
        loss=MSE(),
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=80,
        batch_size=16,
        log_every_n_steps=5,
        backend="equinox",
        seed=42,
    )
    first, last = _first_and_last_loss(loop)
    assert last < first * 0.5, (
        f"direct-supervised loss should drop ≥50%; first={first}, last={last}"
    )


# ---------------------------------------------------------------------
# 2. Emulator — examples/emulator.md
# ---------------------------------------------------------------------


def test_emulator_training_smoke():
    """`SimulationDataset` over a `ForwardModel` trains an emulator,
    and the trained operator drops into `NeuralForward` for a single
    cycle step. This is the design's headline differentiator."""
    sim_ds = SimulationDataset(
        forward_model=_ToyForward(),
        prior=_GaussianPrior(dim=2),
        n_samples=200,
        seed=0,
        split="train",
    )

    loop = TrainingLoop(
        model_op=_EquinoxModelOp(_toy_mlp(in_size=2, out_size=2)),
        dataset=sim_ds,
        loss=MSE(),
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=120,
        batch_size=16,
        log_every_n_steps=10,
        backend="equinox",
        seed=0,
    )
    first, last = _first_and_last_loss(loop)
    assert last < first * 0.5, (
        f"emulator MSE should drop ≥50%; first={first}, last={last}"
    )

    # Drop the trained emulator into NeuralForward.
    trained_op, _ = loop.run()
    fwd = NeuralForward(model_op=trained_op, dt=_ToyForward.dt)
    state0 = np.array([1.0, 0.0], dtype=np.float32)
    state1 = fwd.step(state0, dt=_ToyForward.dt)
    assert state1.shape == (2,)
    assert np.all(np.isfinite(np.asarray(state1)))


# ---------------------------------------------------------------------
# 3. Amortized inference — examples/amortized.md
# ---------------------------------------------------------------------


def test_amortized_inference_smoke():
    """SBI-shape dataset: `SimulationDataset(obs_op=…)` yields
    `(params, observed_concentrations)` pairs.

    We don't have pyrox in the workspace, so the test uses a small
    MLP that's trained to invert the obs → params mapping with MSE
    (rather than a full conditional density). What's exercised is
    the obs_op-applied SimulationDataset path + TrainingLoop with
    inverted target shape — the same data flow the SBI example
    promises.
    """

    class _Identity(Operator):
        """Obs operator: y = x (the observation is the state)."""

        def _apply(self, x: Any) -> Any:
            return x

    # The dataset yields (params, observed_state) pairs.
    sbi_ds = SimulationDataset(
        forward_model=_ToyForward(),
        prior=_GaussianPrior(dim=2),
        n_samples=200,
        obs_op=_Identity(),
        seed=0,
    )

    # The "inverse" network: takes observations, predicts params.
    # SimulationDataset yields (params, obs); for the inverse, we
    # need (obs, params). Use an adapter dataset that flips.
    class _Flipped(IterableDataset):
        def __init__(self, base: SimulationDataset) -> None:
            super().__init__(
                source=lambda: ((y, x) for x, y in base),
                content_hash=f"flipped-{base.content_hash()}",
            )

    inverse_ds = _Flipped(sbi_ds)

    loop = TrainingLoop(
        model_op=_EquinoxModelOp(_toy_mlp(in_size=2, out_size=2)),
        dataset=inverse_ds,
        loss=MSE(),
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=120,
        batch_size=16,
        log_every_n_steps=10,
        backend="equinox",
        seed=0,
    )
    first, last = _first_and_last_loss(loop)
    # SBI-shape inverse problem; assert improvement (not the strong
    # 50% drop — inverse problems are harder).
    assert last < first, (
        f"amortized inverse loss should improve; first={first}, last={last}"
    )

    # Pull a posterior sample: the trained inverse network maps
    # an observation to predicted params.
    trained_op, _ = loop.run()
    obs = jnp.array([1.0, 0.0], dtype=jnp.float32)
    params_pred = trained_op(obs)
    assert params_pred.shape == (2,)
    assert np.all(np.isfinite(np.asarray(params_pred)))
