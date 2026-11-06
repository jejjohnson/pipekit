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
from pipekit_ops import Operator
from pipekit_train import MSE, IterableDataset, TrainingLoop
from pipekit_train.adapters.bayes import NumpyroPredictiveOp, NumpyroTask, materialize


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


def test_svi_traces_the_model_once_across_steps() -> None:
    # regression for #160: an uncompiled svi.update re-traced the model each step.
    traces = 0

    def counting_model(x: Any, y: Any = None) -> None:
        nonlocal traces
        traces += 1
        _model(x, y)

    def fit(max_steps: int) -> int:
        nonlocal traces
        traces = 0
        TrainingLoop(
            model_op=_DummyModelOp(),
            dataset=_linreg_dataset(),
            task=NumpyroTask(
                counting_model, predictive_site="obs", predictive_samples=4
            ),
            backend="numpyro-svi",
            optimizer_config={"name": "adam", "lr": 1e-2},
            max_steps=max_steps,
            seed=0,
        ).run()
        return traces

    assert fit(2) == fit(100)


def test_svi_scores_validation_on_the_final_state() -> None:
    _, artifact = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=_linreg_dataset(),
        val_dataset=_linreg_dataset(n=16, seed=1),
        task=NumpyroTask(_model, predictive_site="obs", predictive_samples=4),
        backend="numpyro-svi",
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=7,
        eval_every_n_steps=5,
        seed=0,
    ).run()
    assert set(artifact.backend_info["final_eval_metrics"]) == {"elbo"}


def test_svi_compiled_elbo_matches_the_uncompiled_path() -> None:
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    jax = pytest.importorskip("jax")
    optax = pytest.importorskip("optax")

    max_steps = 50
    x, y = materialize(_linreg_dataset())
    guide = AutoNormal(_model)
    # The adapter splits the seed once before init, so the same split is used here.
    _, init_key = jax.random.split(jax.random.key(0))
    svi = SVI(_model, guide, optax.adam(1e-2), loss=Trace_ELBO())
    state = svi.init(init_key, x, y)
    for _ in range(max_steps):
        state, loss = svi.update(state, x, y)

    _, artifact = TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=_linreg_dataset(),
        task=NumpyroTask(_model, predictive_site="obs", predictive_samples=4),
        backend="numpyro-svi",
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=max_steps,
        seed=0,
    ).run()

    # float32: compiled and eager rounding differ by about 2e-5 after 50 steps.
    assert artifact.backend_info["final_elbo"] == pytest.approx(-float(loss), rel=1e-4)


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


# --- Kernel settings, unsupervised fits and full posterior draws ------------


def _location_scale(data: Any = None, *, prior_scale: float = 10.0) -> None:
    mu = numpyro.sample("mu", dist.Normal(0.0, prior_scale))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    with numpyro.plate("n", 32 if data is None else data.shape[0]):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=data)


def _unpaired_dataset(n: int = 200) -> IterableDataset:
    draws = np.random.default_rng(0).normal(3.0, 0.5, size=n).astype(np.float32)
    return IterableDataset(source=list(draws), content_hash=f"unpaired-{n}")


def _unsupervised_task(**kwargs: Any) -> NumpyroTask:
    return NumpyroTask(
        _location_scale,
        supervised=False,
        model_kwargs={"prior_scale": 10.0},
        num_warmup=100,
        num_samples=100,
        predictive_samples=16,
        **kwargs,
    )


def _fit(task: Any, backend: str, **kwargs: Any) -> Any:
    return TrainingLoop(
        model_op=_DummyModelOp(),
        dataset=_unpaired_dataset(),
        task=task,
        backend=backend,
        seed=0,
        **kwargs,
    ).run()


def test_kernel_kwargs_reach_nuts(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpyro.infer as infer

    seen: dict[str, Any] = {}
    real_nuts = infer.NUTS

    def spy(model: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real_nuts(model, **kwargs)

    monkeypatch.setattr(infer, "NUTS", spy)
    settings = {"target_accept_prob": 0.9, "max_tree_depth": 5}
    _fit(_unsupervised_task(kernel_kwargs=settings), "numpyro-mcmc")
    assert seen == settings


def test_kernel_factory_replaces_nuts() -> None:
    from numpyro.infer import HMC

    built: list[Any] = []

    def factory(model: Any, **kwargs: Any) -> Any:
        built.append(kwargs)
        return HMC(model, **kwargs)

    task = _unsupervised_task(kernel_factory=factory, kernel_kwargs={"num_steps": 4})
    op, _ = _fit(task, "numpyro-mcmc")
    assert built == [{"num_steps": 4}]
    assert op.posterior_samples()["mu"].shape == (100,)


def test_unsupervised_mcmc_recovers_the_location() -> None:
    op, _ = _fit(_unsupervised_task(), "numpyro-mcmc")
    samples = op.posterior_samples()
    assert samples["mu"].shape == (100,)
    assert float(np.mean(samples["mu"])) == pytest.approx(3.0, abs=0.2)


def test_full_predictive_draws_have_a_sample_dim() -> None:
    op, _ = _fit(_unsupervised_task(), "numpyro-mcmc")
    draws = op.predictive_draws(None)
    assert draws["obs"].shape == (100, 32)
    # the operator surface stays the posterior-predictive mean
    assert op(None).shape == (32,)


def test_predictive_draws_take_a_key() -> None:
    import jax

    op, _ = _fit(_unsupervised_task(), "numpyro-mcmc")
    first = op.predictive_draws(None, key=jax.random.key(1))["obs"]
    again = op.predictive_draws(None, key=jax.random.key(1))["obs"]
    other = op.predictive_draws(None, key=jax.random.key(2))["obs"]
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_unsupervised_svi_draws_from_its_guide() -> None:
    op, _ = _fit(
        _unsupervised_task(),
        "numpyro-svi",
        optimizer_config={"name": "adam", "lr": 5e-2},
        max_steps=300,
    )
    samples = op.posterior_samples(num_samples=50)
    assert samples["mu"].shape == (50,)
    assert float(np.mean(samples["mu"])) == pytest.approx(3.0, abs=0.5)
    assert op.posterior_samples()["mu"].shape == (16,)


def test_svi_with_a_custom_guide_says_it_cant_sample() -> None:
    def guide(data: Any = None, *, prior_scale: float = 10.0) -> None:
        loc = numpyro.param("loc", 0.0)
        numpyro.sample("mu", dist.Delta(loc))
        numpyro.sample("sigma", dist.Delta(numpyro.param("s", 1.0)))

    op = NumpyroPredictiveOp(None, posterior={"loc": 0.0}, guide=guide)
    with pytest.raises(TypeError, match="sample_posterior"):
        op.posterior_samples(num_samples=2)


def test_blackjax_samples_an_unsupervised_numpyro_task() -> None:
    pytest.importorskip("blackjax")
    from pipekit_train.adapters.blackjax import BlackjaxTask

    op, _ = _fit(
        BlackjaxTask(numpyro_task=_unsupervised_task(), num_warmup=100),
        "blackjax",
        max_steps=100,
    )
    assert float(np.mean(op.posterior_samples()["mu"])) == pytest.approx(3.0, abs=0.2)
    assert op.predictive_draws(None)["obs"].shape == (100, 32)


def test_unsupervised_fit_round_trips_through_the_registry(tmp_path: Any) -> None:
    pytest.importorskip("pipekit_experiment")
    from pipekit_experiment import LocalModelRegistry
    from pipekit_train import Checkpoint

    registry = LocalModelRegistry(tmp_path / "models")
    op, artifact = _fit(
        _unsupervised_task(), "numpyro-mcmc", callbacks=(Checkpoint(registry=registry),)
    )

    reloaded = NumpyroPredictiveOp.from_registry(
        registry,
        artifact.trained_model_hash,
        model=_location_scale,
        model_kwargs={"prior_scale": 10.0},
    )
    np.testing.assert_array_equal(
        reloaded.posterior_samples()["mu"], op.posterior_samples()["mu"]
    )
    assert reloaded.predictive_draws(None)["obs"].shape == (100, 32)


def test_reloading_without_a_model_still_gives_the_samples(tmp_path: Any) -> None:
    pytest.importorskip("pipekit_experiment")
    from pipekit_experiment import LocalModelRegistry
    from pipekit_train import Checkpoint

    registry = LocalModelRegistry(tmp_path / "models")
    _, artifact = _fit(
        _unsupervised_task(), "numpyro-mcmc", callbacks=(Checkpoint(registry=registry),)
    )
    reloaded = NumpyroPredictiveOp.from_registry(registry, artifact.trained_model_hash)

    assert reloaded.posterior_samples()["mu"].shape == (100,)
    with pytest.raises(TypeError, match="no model"):
        reloaded.predictive_draws(None)


def test_reloading_a_hash_with_no_weights_raises() -> None:
    class Empty:
        def load_weights(self, ref: str) -> None:
            return None

    with pytest.raises(ValueError, match="no weights"):
        NumpyroPredictiveOp.from_registry(Empty(), "abc")


def test_kernel_without_divergence_diagnostics_still_fits() -> None:
    from numpyro.infer import BarkerMH

    op, artifact = _fit(_unsupervised_task(kernel_factory=BarkerMH), "numpyro-mcmc")
    assert op.posterior_samples()["mu"].shape == (100,)
    assert artifact.backend_info["num_divergences"] is None


def test_nuts_still_reports_divergences() -> None:
    _, artifact = _fit(_unsupervised_task(), "numpyro-mcmc")
    assert isinstance(artifact.backend_info["num_divergences"], int)


def test_guide_that_replays_the_model_gets_its_arguments() -> None:
    from numpyro.infer.autoguide import AutoDelta

    def with_deterministic(data: Any = None, *, prior_scale: float = 10.0) -> None:
        mu = numpyro.sample("mu", dist.Normal(0.0, prior_scale))
        numpyro.deterministic("doubled", 2.0 * mu)
        sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
        with numpyro.plate("n", 32 if data is None else data.shape[0]):
            numpyro.sample("obs", dist.Normal(mu, sigma), obs=data)

    data = np.asarray(list(_unpaired_dataset()))
    task = NumpyroTask(
        with_deterministic,
        guide=AutoDelta(with_deterministic),
        supervised=False,
        model_kwargs={"prior_scale": 10.0},
        predictive_samples=4,
    )
    op, _ = _fit(
        task,
        "numpyro-svi",
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=5,
    )
    samples = op.posterior_samples(data)
    np.testing.assert_allclose(samples["doubled"], 2.0 * samples["mu"])
