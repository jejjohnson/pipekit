"""Tests for the BlackJAX backend — ``backend="blackjax"``.

Gated by ``importorskip`` (needs the ``[blackjax]`` extra). The headline
path — BlackJAX NUTS on a NumPyro model via the shared ``_bayes`` bridge —
is exercised on a Bayesian linear regression and cross-checked against the
``numpyro-mcmc`` engine on the *same* task.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


pytest.importorskip("blackjax")
pytest.importorskip("numpyro")
pytest.importorskip("jax")

import numpyro
import numpyro.distributions as dist
from pipekit import Operator
from pipekit_train import IterableDataset, TrainingLoop
from pipekit_train.adapters._bayes import NumpyroTask
from pipekit_train.adapters.blackjax import BlackjaxPredictiveOp, BlackjaxTask


class _DummyModelOp(Operator):
    __config_mixin_auto__ = False

    def _apply(self, x: Any) -> Any:
        return x

    def get_config(self) -> dict[str, Any]:
        return {}


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
    numpyro.sample("obs", dist.Normal(w * x[..., 0] + b, sigma), obs=y)


def _truth_mse(model_op: Operator, dataset: IterableDataset) -> float:
    import jax.numpy as jnp

    xs = jnp.asarray(np.stack([x for x, _ in dataset]))
    y_hat = np.asarray(model_op(xs))
    truth = 2.0 * np.asarray(xs)[:, 0] + 1.0
    return float(np.mean((y_hat - truth) ** 2))


# --- task validation (fast) -----------------------------------------------


def test_requires_a_blackjax_task() -> None:
    loop = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=_linreg_dataset(),
        task=object(),
        backend="blackjax",
        max_steps=1,
        seed=0,
    )
    with pytest.raises(TypeError, match="BlackjaxTask"):
        loop.run()


def test_planned_sampler_raises() -> None:
    loop = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=_linreg_dataset(),
        task=BlackjaxTask(numpyro_task=NumpyroTask(_model), sampler="sgld"),
        backend="blackjax",
        max_steps=1,
        seed=0,
    )
    with pytest.raises(NotImplementedError, match="planned BlackJAX follow-on"):
        loop.run()


def test_raw_task_needs_target() -> None:
    loop = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=_linreg_dataset(),
        task=BlackjaxTask(sampler="nuts"),  # neither numpyro_task nor logdensity_fn
        backend="blackjax",
        max_steps=1,
        seed=0,
    )
    with pytest.raises(ValueError, match="numpyro_task"):
        loop.run()


def test_over_specified_task_raises() -> None:
    # supplying both a numpyro_task and a raw target is ambiguous → reject
    loop = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=_linreg_dataset(),
        task=BlackjaxTask(
            numpyro_task=NumpyroTask(_model),
            logdensity_fn=lambda p: 0.0,
            init_position={"w": np.array(0.0)},
            sampler="nuts",
        ),
        backend="blackjax",
        max_steps=1,
        seed=0,
    )
    with pytest.raises(ValueError, match="over-specified"):
        loop.run()


def test_num_chains_gt_one_raises() -> None:
    # v1 runs a single chain; num_chains>1 must raise rather than silently
    # return single-chain diagnostics.
    loop = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=_linreg_dataset(),
        task=BlackjaxTask(numpyro_task=NumpyroTask(_model), num_chains=2),
        backend="blackjax",
        max_steps=1,
        seed=0,
    )
    with pytest.raises(NotImplementedError, match="num_chains"):
        loop.run()


def test_raw_predictive_op_without_predict_fn_errors() -> None:
    op = BlackjaxPredictiveOp(None, {"w": np.array([1.0, 2.0])})
    with pytest.raises(RuntimeError, match="no predict_fn"):
        op(np.zeros((3, 1), dtype=np.float32))


# --- NUTS on a NumPyro model (slow) ---------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_nuts_on_numpyro_recovers_line() -> None:
    dataset = _linreg_dataset()
    model_op, artifact = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=dataset,
        task=BlackjaxTask(
            numpyro_task=NumpyroTask(_model),
            sampler="nuts",
            num_warmup=300,
            predictive_site="obs",
        ),
        backend="blackjax",
        max_steps=300,
        seed=0,
    ).run()
    info = artifact.backend_info

    assert isinstance(model_op, Operator)
    assert info["backend"] == "blackjax" and info["sampler"] == "nuts"
    assert isinstance(info["num_divergences"], int)
    assert _truth_mse(model_op, dataset) < 0.2


@pytest.mark.slow
@pytest.mark.integration
def test_raw_logdensity_path_skips_materialization() -> None:
    # the raw target carries its own logdensity, so it must run without
    # touching loop.dataset (here an empty placeholder) and must round-trip an
    # array-valued init_position through the predictive op as a pytree.
    import jax.numpy as jnp

    empty = IterableDataset(source=[], content_hash="empty")

    def logdensity_fn(theta: Any) -> Any:
        return -0.5 * jnp.sum((theta - 3.0) ** 2)

    model_op, artifact = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=empty,
        task=BlackjaxTask(
            logdensity_fn=logdensity_fn,
            init_position=jnp.zeros((1,)),
            predict_fn=lambda theta, x: theta,
            sampler="nuts",
            num_warmup=200,
        ),
        backend="blackjax",
        max_steps=300,
        seed=0,
    ).run()

    assert isinstance(model_op, BlackjaxPredictiveOp)
    assert artifact.backend_info["backend"] == "blackjax"
    # posterior mean of an N(3, 1) target recovered by the raw NUTS path
    assert abs(float(np.asarray(model_op(np.zeros((1, 1))))[0] - 3.0)) < 0.5


@pytest.mark.slow
@pytest.mark.integration
def test_same_model_swaps_numpyro_mcmc_and_blackjax() -> None:
    # the engine-swap headline: one NumPyro model, two MCMC engines
    dataset = _linreg_dataset()
    nt = NumpyroTask(_model, predictive_site="obs", num_warmup=300, num_samples=300)

    bj_op, _ = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=dataset,
        task=BlackjaxTask(numpyro_task=nt, sampler="nuts", num_warmup=300),
        backend="blackjax",
        max_steps=300,
        seed=0,
    ).run()
    np_op, _ = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=dataset,
        task=nt,
        backend="numpyro-mcmc",
        seed=0,
    ).run()

    assert _truth_mse(bj_op, dataset) < 0.2
    assert _truth_mse(np_op, dataset) < 0.2
