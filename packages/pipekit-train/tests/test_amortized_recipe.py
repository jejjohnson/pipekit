"""The amortized inference recipe in ``docs/design/examples/amortized.md``.

The simulator adds two parameters with standard normal priors, so that each
parameter's marginal posterior given y = θ₁ + θ₂ is N(y / 2, 1 / 2). That closed
form is what the trained posterior is checked against.

Only the marginals: the diagonal head can't represent the true joint posterior,
in which θ₁ + θ₂ = y exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pipekit_ops import Operator


pytest.importorskip("equinox")
pytest.importorskip("optax")
pytest.importorskip("orbax.checkpoint")
pytest.importorskip("grain")

import equinox as eqx
import jax
import jax.numpy as jnp
from pipekit_train import NLL, SimulationDataset, TrainingLoop
from pipekit_train.adapters.equinox import EquinoxModelOp


class Prior(Operator):
    def _apply(self, seed: int | None = None) -> Any:
        return np.random.default_rng(seed).standard_normal(2).astype(np.float32)


class AddParameters:
    dt = 1.0
    state_signature = None

    def step(self, state: Any, dt: float) -> Any:
        return np.array([state.sum()], dtype=np.float32)

    def get_config(self) -> dict[str, Any]:
        return {}


class GaussianHead(eqx.Module):
    loc: jax.Array
    log_scale: jax.Array

    def log_prob(self, theta: jax.Array) -> jax.Array:
        z = (theta - self.loc) * jnp.exp(-self.log_scale)
        terms = -0.5 * z**2 - self.log_scale - 0.5 * jnp.log(2 * jnp.pi)
        return terms.sum(axis=-1)


class AmortizedPosterior(eqx.Module):
    net: eqx.nn.MLP

    def __call__(self, y: jax.Array) -> GaussianHead:
        out = self.net(y)
        return GaussianHead(out[:2], out[2:])


@pytest.fixture(scope="module")
def fitted() -> tuple[Any, SimulationDataset]:
    dataset = SimulationDataset(AddParameters(), Prior(), n_samples=4_000, invert=True)
    network = eqx.nn.MLP(1, 4, 32, 2, key=jax.random.key(0))
    trained, _ = TrainingLoop(
        model_op=EquinoxModelOp(AmortizedPosterior(network)),
        dataset=dataset,
        loss=NLL(),
        optimizer_config={"name": "adam", "lr": 3e-3},
        max_steps=1_500,
        batch_size=64,
        backend="equinox",
        seed=0,
    ).run()
    return trained, dataset


def _test_split(dataset: SimulationDataset) -> tuple[np.ndarray, np.ndarray]:
    pairs = list(dataset.with_split("test"))
    return np.stack([y for y, _ in pairs]), np.stack([theta for _, theta in pairs])


def test_posterior_matches_the_closed_form(fitted: Any) -> None:
    trained, dataset = fitted
    y, _ = _test_split(dataset)
    posterior = jax.vmap(trained.module)(jnp.asarray(y))

    # mean = y / 2 for both parameters
    mean_error = np.asarray(posterior.loc) - np.broadcast_to(y / 2, posterior.loc.shape)
    assert np.sqrt(np.mean(mean_error**2)) < 0.1
    # scale = √(1 / 2)
    scale = np.exp(np.asarray(posterior.log_scale))
    assert np.abs(scale - np.sqrt(0.5)).max() < 0.15


def test_posterior_is_calibrated(fitted: Any) -> None:
    trained, dataset = fitted
    y, theta = _test_split(dataset)
    posterior = jax.vmap(trained.module)(jnp.asarray(y))

    distance = np.abs(theta - np.asarray(posterior.loc))
    inside = distance <= 2 * np.exp(np.asarray(posterior.log_scale))
    assert 0.90 <= float(inside.mean()) <= 0.99
