"""Tests for the NumPyro backends — ``numpyro-svi`` and ``numpyro-mcmc``.

Gated by ``importorskip`` so they only run in the ``[numpyro]`` extra
environment. A Bayesian linear regression (``y = 2x + 1``) is fit by both
backends through the same `TrainingLoop`, asserting the posterior-predictive
operator recovers the line.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


pytest.importorskip("numpyro")
pytest.importorskip("optax")
pytest.importorskip("jax")

import numpyro
import numpyro.distributions as dist
from pipekit import Operator
from pipekit_train import MSE, IterableDataset, TrainingLoop
from pipekit_train.adapters.bayes import NumpyroPredictiveOp, NumpyroTask


# --- fixtures -------------------------------------------------------------


class _DummyModelOp(Operator):
    """A placeholder model_op (ignored by the numpyro backends)."""

    __config_mixin_auto__ = False

    def _apply(self, x: Any) -> Any:
        return x

    def get_config(self) -> dict[str, Any]:
        return {}


class _CaptureWriter:
    """Duck-typed MetricWriter that records writes."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, dict[str, float]]] = []

    def write(self, step: int, metrics: dict[str, float]) -> None:
        self.rows.append((step, dict(metrics)))

    def close(self) -> None:
        pass


def _linreg_dataset(n: int = 64, seed: int = 0) -> IterableDataset:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    ys = (2.0 * xs[:, 0] + 1.0 + 0.05 * rng.standard_normal(n)).astype(np.float32)
    return IterableDataset(
        source=list(zip(xs, ys, strict=True)), content_hash=f"linreg-{n}-{seed}"
    )


def _model(x: Any, y: Any = None) -> None:
    w = numpyro.sample("w", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    mu = w * x[..., 0] + b
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def _truth_mse(model_op: NumpyroPredictiveOp, dataset: IterableDataset) -> float:
    import jax.numpy as jnp

    xs = jnp.asarray(np.stack([x for x, _ in dataset]))
    y_hat = np.asarray(model_op(xs))
    truth = 2.0 * np.asarray(xs)[:, 0] + 1.0
    return float(np.mean((y_hat - truth) ** 2))


# --- task validation ------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpyro-svi", "numpyro-mcmc"])
def test_requires_a_task(backend: str) -> None:
    loop = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=_linreg_dataset(),
        loss=MSE(),
        backend=backend,
        max_steps=1,
        seed=0,
    )
    with pytest.raises(ValueError, match="task-first"):
        loop.run()


@pytest.mark.parametrize("backend", ["numpyro-svi", "numpyro-mcmc"])
def test_rejects_non_numpyro_task(backend: str) -> None:
    loop = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=_linreg_dataset(),
        task=object(),
        backend=backend,
        max_steps=1,
        seed=0,
    )
    with pytest.raises(TypeError, match="NumpyroTask"):
        loop.run()


class _RecordCallback:
    """Records the args each lifecycle hook receives."""

    def __init__(self) -> None:
        self.begin_states: list[Any] = []
        self.epoch_ends = 0
        self.step_ends = 0

    def on_train_begin(self, loop: Any, state: Any) -> None:
        self.begin_states.append(state)

    def on_step_end(self, loop: Any, state: Any, metrics: dict[str, float]) -> None:
        self.step_ends += 1

    def on_epoch_end(self, loop: Any, state: Any, metrics: dict[str, float]) -> None:
        self.epoch_ends += 1

    def on_train_end(self, loop: Any, state: Any) -> None:
        pass


def test_predictive_op_serialize_weights_roundtrips() -> None:
    import pickle

    posterior = {"w": np.array([2.0, 1.0], dtype=np.float32), "b": np.array(0.5)}
    op = NumpyroPredictiveOp(
        predictive=None, predictive_site="obs", posterior=posterior
    )
    blob = op.serialize_weights()
    assert isinstance(blob, bytes) and blob
    restored = pickle.loads(blob)
    assert set(restored) == {"w", "b"}
    np.testing.assert_allclose(restored["w"], posterior["w"])


def test_svi_dispatches_train_begin_state_and_epoch_end() -> None:
    # regression: on_train_begin must receive (loop, state); on_epoch_end
    # must fire when steps_per_epoch is set.
    cb = _RecordCallback()
    TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=_linreg_dataset(),
        task=NumpyroTask(_model, predictive_site="obs", predictive_samples=8),
        backend="numpyro-svi",
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=3,
        steps_per_epoch=1,
        callbacks=(cb,),
        seed=0,
    ).run()
    assert len(cb.begin_states) == 1
    assert cb.begin_states[0].step == 0  # a TrainerCarryState was passed
    assert cb.step_ends == 3
    assert cb.epoch_ends == 3


# --- numpyro-svi ----------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_svi_recovers_line_and_decreases_elbo() -> None:
    dataset = _linreg_dataset()
    writer = _CaptureWriter()
    loop = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=dataset,
        task=NumpyroTask(_model, predictive_site="obs", predictive_samples=200),
        backend="numpyro-svi",
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=3000,
        log_every_n_steps=100,
        metric_writer=writer,
        seed=0,
    )
    model_op, artifact = loop.run()
    info = artifact.backend_info

    assert isinstance(model_op, Operator)
    assert info["backend"] == "numpyro-svi"
    assert np.isfinite(info["final_elbo"])
    # the posterior-predictive operator recovers y = 2x + 1
    assert _truth_mse(model_op, dataset) < 0.2
    # ELBO improved over training (loss decreased)
    assert len(writer.rows) >= 2
    assert writer.rows[-1][1]["elbo"] > writer.rows[0][1]["elbo"]


# --- numpyro-mcmc ---------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_mcmc_recovers_line() -> None:
    dataset = _linreg_dataset()
    loop = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=dataset,
        task=NumpyroTask(
            _model,
            predictive_site="obs",
            predictive_samples=200,
            num_warmup=300,
            num_samples=300,
        ),
        backend="numpyro-mcmc",
        seed=0,
    )
    model_op, artifact = loop.run()
    info = artifact.backend_info

    assert isinstance(model_op, Operator)
    assert info["backend"] == "numpyro-mcmc"
    assert info["final_step"] == 300
    assert isinstance(info["num_divergences"], int)
    assert set(info["sites"]) >= {"w", "b", "sigma"}
    assert _truth_mse(model_op, dataset) < 0.2


@pytest.mark.slow
@pytest.mark.integration
def test_same_task_runs_under_both_backends() -> None:
    # one task object, swap the engine by flipping the backend string
    dataset = _linreg_dataset()
    task = NumpyroTask(
        _model,
        predictive_site="obs",
        predictive_samples=100,
        num_warmup=200,
        num_samples=200,
    )
    svi_op, _ = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=dataset,
        task=task,
        backend="numpyro-svi",
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=2000,
        seed=0,
    ).run()
    mcmc_op, _ = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=dataset,
        task=task,
        backend="numpyro-mcmc",
        seed=0,
    ).run()
    assert _truth_mse(svi_op, dataset) < 0.3
    assert _truth_mse(mcmc_op, dataset) < 0.3
