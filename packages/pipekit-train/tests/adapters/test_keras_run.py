"""The Keras backend's `run`: compile and fit under a `TrainingLoop` (ADR D18)."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest


if importlib.util.find_spec("keras") is None:
    pytest.skip("needs the [keras] extra", allow_module_level=True)

from pipekit_train import (
    MSE,
    EarlyStopping,
    HyperSweep,
    IterableDataset,
    ParameterGrid,
    TrainingDataset,
    TrainingLoop,
)
from pipekit_train.backends.keras import (
    KerasModelOp,
    KerasTask,
    build_keras_optimizer,
    import_keras,
)


keras = import_keras()


class _Pairs(TrainingDataset):
    """An indexable regression set, so batches come from Grain."""

    forbid_in_yaml = True
    __config_mixin_auto__ = False

    def __init__(self, n: int, seed: int) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(n, 3)).astype("float32")
        weights = np.array([[1.0], [-2.0], [0.5]], dtype="float32")
        self.y = (self.x @ weights + 0.3).astype("float32")
        self._hash = f"pairs-{n}-{seed}"

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def content_hash(self) -> str:
        return self._hash


def _model(seed: int = 0) -> Any:
    keras.utils.set_random_seed(seed)
    return keras.Sequential(
        [keras.Input((3,)), keras.layers.Dense(8, activation="tanh"), keras.layers.Dense(1)]
    )


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []
        self.step_metrics: list[dict[str, float]] = []
        self.eval_metrics: list[dict[str, float]] = []

    def on_train_begin(self, loop, state) -> None:
        self.events.append(("begin", state.step))

    def on_step_end(self, loop, state, metrics) -> None:
        self.step_metrics.append(dict(metrics))

    def on_epoch_end(self, loop, state, metrics) -> None:
        self.events.append(("epoch", state.step))

    def on_eval_end(self, loop, state, eval_metrics) -> None:
        self.events.append(("eval", state.step))
        self.eval_metrics.append(dict(eval_metrics))

    def on_train_end(self, loop, state) -> None:
        self.events.append(("end", state.step))


def _loop(model, **kwargs) -> TrainingLoop:
    defaults: dict[str, Any] = {
        "model_op": KerasModelOp(model),
        "dataset": _Pairs(128, 0),
        "loss": MSE(),
        "optimizer_config": {"name": "adam", "lr": 1e-2},
        "max_steps": 250,
        "eval_every_n_steps": 100,
        "batch_size": 16,
        "backend": "keras",
    }
    defaults.update(kwargs)
    return TrainingLoop(**defaults)


def test_stops_at_max_steps_and_validates_at_each_period_and_the_end() -> None:
    recorder = _Recorder()
    op, artifact = _loop(
        _model(),
        val_dataset=_Pairs(40, 1),
        callbacks=(recorder,),
        steps_per_epoch=50,
    ).run()
    info = artifact.backend_info
    assert info["backend"] == "keras"
    assert info["keras_backend"] == "jax"
    assert info["final_step"] == 250
    assert isinstance(op, KerasModelOp)
    assert [step for kind, step in recorder.events if kind == "eval"] == [100, 200, 250]
    assert [step for kind, step in recorder.events if kind == "epoch"] == [
        50,
        100,
        150,
        200,
        250,
    ]
    assert recorder.events[0] == ("begin", 0)
    assert recorder.events[-1] == ("end", 250)
    assert len(recorder.step_metrics) == 250
    assert set(info["final_eval_metrics"]) == {"loss", "mse"}
    assert info["final_metrics"] == recorder.step_metrics[-1]


def test_logged_losses_are_each_steps_own() -> None:
    """With metrics reset after each step, the log is not Keras's running mean."""
    recorder = _Recorder()
    model = _model()
    initial = model.get_weights()
    _loop(model, max_steps=6, eval_every_n_steps=3, callbacks=(recorder,)).run()
    logged = [m["loss"] for m in recorder.step_metrics]

    # the same batches, one train_on_batch each, from the same start
    from pipekit_train.backends._batches import BatchSource

    replay = _model()
    replay.set_weights(initial)
    replay.compile(optimizer=build_keras_optimizer({"name": "adam", "lr": 1e-2}), loss="mse")
    source = BatchSource(_Pairs(128, 0), 16, 0)
    expected = [float(replay.train_on_batch(*source.next_batch())) for _ in range(6)]
    np.testing.assert_allclose(logged, expected, rtol=1e-5)


def test_validation_is_the_per_sample_mean_over_a_ragged_set() -> None:
    model = _model()
    val = _Pairs(70, 2)
    _, artifact = _loop(model, val_dataset=val, max_steps=20, eval_every_n_steps=20, batch_size=32).run()
    eval_metrics = artifact.backend_info["final_eval_metrics"]
    predicted = np.asarray(model(val.x, training=False))
    exact = float(np.mean((predicted - val.y) ** 2))
    assert eval_metrics["loss"] == pytest.approx(exact, rel=1e-5)
    assert eval_metrics["mse"] == pytest.approx(exact, rel=1e-5)


def test_early_stopping_stops_at_the_end_of_a_period() -> None:
    stopper = EarlyStopping(metric="mse", patience=1, min_delta=1e9)
    _, artifact = _loop(
        _model(),
        val_dataset=_Pairs(40, 1),
        callbacks=(stopper,),
        max_steps=1_000,
        eval_every_n_steps=50,
    ).run()
    # the first evaluation sets the best; the second cannot improve by 1e9
    assert artifact.backend_info["final_step"] == 100


def test_metric_writer_writes_every_log_period(tmp_path) -> None:
    from pipekit_train import JSONLWriter

    path = tmp_path / "metrics.jsonl"
    _loop(
        _model(),
        max_steps=30,
        eval_every_n_steps=10,
        log_every_n_steps=10,
        metric_writer=JSONLWriter(path),
    ).run()
    lines = path.read_text().splitlines()
    assert len(lines) == 3


def test_keep_compile_keeps_the_models_own_loss() -> None:
    model = _model()
    model.compile(optimizer="sgd", loss="mae")
    with pytest.warns(UserWarning, match="optimizer_config is not used"):
        _, artifact = _loop(
            model, loss=None, task=KerasTask(keep_compile=True), max_steps=5
        ).run()
    assert model.loss == "mae"
    assert isinstance(model.optimizer, keras.optimizers.SGD)
    assert set(artifact.backend_info["final_metrics"]) == {"loss"}


def test_a_keras_loss_in_the_task_replaces_the_loop_loss() -> None:
    _, artifact = _loop(
        _model(), loss=None, task=KerasTask(loss="mae", metrics=("mse",)), max_steps=5
    ).run()
    assert set(artifact.backend_info["final_metrics"]) == {"loss", "mse"}


def test_streaming_datasets_train() -> None:
    pairs = list(zip(_Pairs(64, 3).x, _Pairs(64, 3).y, strict=True))
    dataset = IterableDataset(source=pairs, content_hash="stream-64")
    _, artifact = _loop(
        _model(), dataset=dataset, val_dataset=dataset, max_steps=12, eval_every_n_steps=6
    ).run()
    assert artifact.backend_info["final_step"] == 12


def test_errors_name_what_is_wrong() -> None:
    from pipekit_train.backends.equinox import EquinoxModelOp

    eqx = pytest.importorskip("equinox")
    jax = pytest.importorskip("jax")
    module = eqx.nn.Linear(3, 1, key=jax.random.key(0))
    with pytest.raises(TypeError, match="KerasModelOp"):
        _loop(None, model_op=EquinoxModelOp(module)).run()
    with pytest.raises(TypeError, match="KerasTask"):
        _loop(_model(), task=object()).run()
    with pytest.raises(ValueError, match="loop.loss or loop.task"):
        _loop(_model(), loss=None).run()
    with pytest.raises(ValueError, match="compiled model"):
        _loop(_model(), loss=None, task=KerasTask(keep_compile=True)).run()
    with pytest.raises(NotImplementedError, match="shard"):
        _loop(_model(), sharding=object()).run()


def test_hyper_sweep_ranks_keras_trials_on_validation() -> None:
    def build(params: dict[str, Any]) -> TrainingLoop:
        return _loop(
            _model(),
            val_dataset=_Pairs(40, 1),
            optimizer_config={"name": "adam", "lr": params["lr"]},
            max_steps=60,
            eval_every_n_steps=30,
        )

    results = HyperSweep(
        loop_template=build,
        search_space=ParameterGrid({"lr": [1e-5, 3e-2]}),
        select_metric="mse",
        select_mode="min",
    ).run()
    assert [r.params["lr"] for r in results] == [3e-2, 1e-5]
    assert all(isinstance(r.trained_model_op, KerasModelOp) for r in results)


def test_optimizer_config_trains_as_optax_does() -> None:
    """Adam from one config: Keras and optax agree to float32 rounding."""
    jax = pytest.importorskip("jax")
    optax = pytest.importorskip("optax")
    jnp = jax.numpy

    data = _Pairs(64, 4)
    model = _model()
    start = [np.array(w) for w in model.trainable_weights]
    config = {"name": "adam", "lr": 1e-2}
    model.compile(optimizer=build_keras_optimizer(config), loss="mse")

    optimizer = optax.adam(1e-2)
    params = [jnp.asarray(w) for w in start]
    state = optimizer.init(params)

    def loss(params, x, y):
        predicted, _ = model.stateless_call(params, [], x)
        return jnp.mean((predicted - y) ** 2)

    for k in range(50):
        rows = slice((k * 16) % 64, (k * 16) % 64 + 16)
        x, y = data.x[rows], data.y[rows]
        model.train_on_batch(x, y)
        grads = jax.grad(loss)(params, x, y)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)

    for keras_weight, optax_weight in zip(model.trainable_weights, params, strict=True):
        np.testing.assert_allclose(np.asarray(keras_weight), optax_weight, atol=1e-5)


def test_unknown_optimizer_names_are_refused() -> None:
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_keras_optimizer({"name": "lion"})
