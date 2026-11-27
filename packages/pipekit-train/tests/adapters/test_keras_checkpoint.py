"""Keras runs that resume from a checkpoint (`pipekit_train.backends.keras`)."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest


if importlib.util.find_spec("keras") is None:
    pytest.skip("needs the [keras] extra", allow_module_level=True)
pytest.importorskip("orbax.checkpoint")

from pipekit_train import MSE, Checkpoint, TrainingDataset, TrainingLoop
from pipekit_train.backends.keras import KerasModelOp, import_keras


keras = import_keras()


class _Pairs(TrainingDataset):
    forbid_in_yaml = True
    __config_mixin_auto__ = False

    def __init__(self, n: int = 96, seed: int = 0) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(n, 3)).astype("float32")
        weights = np.array([[1.0], [-2.0], [0.5]], dtype="float32")
        self.y = (self.x @ weights + 0.3).astype("float32")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def content_hash(self) -> str:
        return "pairs-ckpt"


def _model(seed: int = 0) -> Any:
    keras.utils.set_random_seed(seed)
    return keras.Sequential(
        [
            keras.Input((3,)),
            keras.layers.Dense(8, activation="tanh"),
            keras.layers.Dense(1),
        ]
    )


def _run(max_steps: int, period: int = 100, seed: int = 0, **kwargs: Any) -> Any:
    model = _model(seed)
    _, artifact = TrainingLoop(
        model_op=KerasModelOp(model),
        dataset=_Pairs(),
        val_dataset=_Pairs(40, 1),
        loss=MSE(),
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=max_steps,
        eval_every_n_steps=period,
        batch_size=16,
        backend="keras",
        **kwargs,
    ).run()
    return model, artifact


def _assert_same_weights(a: Any, b: Any) -> None:
    for x, y in zip(a.get_weights(), b.get_weights(), strict=True):
        np.testing.assert_array_equal(x, y)


def test_a_resumed_run_ends_with_the_weights_of_an_uninterrupted_one(tmp_path) -> None:
    uninterrupted, _ = _run(300)
    _run(200, checkpoint_dir=str(tmp_path))
    # a model built with another seed: the checkpoint sets its weights
    resumed, artifact = _run(300, seed=5, checkpoint_dir=str(tmp_path))
    assert artifact.backend_info["final_step"] == 300
    _assert_same_weights(resumed, uninterrupted)


def test_a_run_resumed_off_the_period_grid_catches_up(tmp_path) -> None:
    uninterrupted, _ = _run(400)
    _run(250, checkpoint_dir=str(tmp_path))
    resumed, artifact = _run(400, seed=5, checkpoint_dir=str(tmp_path))
    assert artifact.backend_info["final_step"] == 400
    _assert_same_weights(resumed, uninterrupted)


def test_one_step_fit_calls_keep_the_data_position(tmp_path) -> None:
    """Keras reads two batches ahead; the checkpoint keeps the one trained on."""
    uninterrupted, _ = _run(3, period=1)
    _run(1, period=1, checkpoint_dir=str(tmp_path))
    resumed, _ = _run(3, period=1, seed=5, checkpoint_dir=str(tmp_path))
    _assert_same_weights(resumed, uninterrupted)


def test_a_finished_run_does_not_train_again(tmp_path) -> None:
    first, _ = _run(200, checkpoint_dir=str(tmp_path))
    again, artifact = _run(200, seed=5, checkpoint_dir=str(tmp_path))
    assert artifact.backend_info["final_step"] == 200
    _assert_same_weights(again, first)


def test_keep_last_retains_that_many_checkpoints(tmp_path) -> None:
    import orbax.checkpoint as ocp

    _run(
        400, callbacks=(Checkpoint(every_n_steps=100, keep_last=2, save_dir=tmp_path),)
    )
    assert ocp.CheckpointManager(tmp_path).all_steps() == [300, 400]


def test_a_cadence_off_the_period_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="multiple of"):
        _run(200, callbacks=(Checkpoint(every_n_steps=150, save_dir=tmp_path),))


def test_an_equinox_model_resumes_too(tmp_path) -> None:
    eqx = pytest.importorskip("equinox")
    jax = pytest.importorskip("jax")
    from pipekit_train.backends.equinox import EquinoxModelOp

    def run(max_steps: int, directory: str | None) -> Any:
        module = eqx.nn.MLP(3, 1, width_size=8, depth=1, key=jax.random.key(0))
        trained, _ = TrainingLoop(
            model_op=EquinoxModelOp(module),
            dataset=_Pairs(),
            loss=MSE(),
            optimizer_config={"name": "adam", "lr": 1e-2},
            max_steps=max_steps,
            eval_every_n_steps=50,
            batch_size=16,
            backend="keras",
            checkpoint_dir=directory,
        ).run()
        return jax.tree.leaves(eqx.filter(trained.module, eqx.is_array))

    uninterrupted = run(150, None)
    run(100, str(tmp_path))
    resumed = run(150, str(tmp_path))
    for a, b in zip(uninterrupted, resumed, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
