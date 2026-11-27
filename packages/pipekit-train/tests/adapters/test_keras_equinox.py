"""Equinox models under the Keras backend, through `EquinoxLayer` (ADR D21)."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest


if importlib.util.find_spec("keras") is None:
    pytest.skip("needs the [keras] extra", allow_module_level=True)
eqx = pytest.importorskip("equinox")
jax = pytest.importorskip("jax")

import jax.numpy as jnp
from pipekit_train import MSE, TrainingDataset, TrainingLoop
from pipekit_train.backends.equinox import EquinoxModelOp
from pipekit_train.backends.keras import (
    EquinoxLayer,
    KerasModelOp,
    build_keras_optimizer,
    import_keras,
)


keras = import_keras()


class _Pairs(TrainingDataset):
    forbid_in_yaml = True
    __config_mixin_auto__ = False

    def __init__(self, n: int = 64, seed: int = 0) -> None:
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
        return "pairs-eqx"


def _mlp(seed: int = 0) -> Any:
    return eqx.nn.MLP(3, 1, width_size=8, depth=1, activation=jnp.tanh, key=jax.random.key(seed))


def _leaves(module: Any) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree.leaves(eqx.filter(module, eqx.is_array))]


def _loop(model_op: Any, backend: str = "keras", **kwargs: Any) -> TrainingLoop:
    defaults: dict[str, Any] = {
        "model_op": model_op,
        "dataset": _Pairs(),
        "loss": MSE(),
        "optimizer_config": {"name": "adam", "lr": 1e-2},
        "max_steps": 20,
        "eval_every_n_steps": 10,
        "batch_size": 16,
        "backend": backend,
    }
    defaults.update(kwargs)
    return TrainingLoop(**defaults)


def test_one_sgd_step_equals_the_hand_written_jax_step() -> None:
    data = _Pairs()
    module = _mlp()
    layer = EquinoxLayer(module)
    model = keras.Sequential([layer])
    model(data.x[:2])
    model.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")
    model.train_on_batch(data.x, data.y)

    params, static = eqx.partition(module, eqx.is_inexact_array)

    def loss(params):
        predicted = jax.vmap(eqx.combine(params, static))(data.x)
        return jnp.mean((predicted - data.y) ** 2)

    grads = jax.grad(loss)(params)
    stepped = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)
    for a, b in zip(_leaves(stepped), _leaves(layer.module), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_the_given_module_stays_usable_and_untrained() -> None:
    module = _mlp()
    before = _leaves(module)
    op = EquinoxModelOp(module)
    trained, _ = _loop(op).run()
    assert isinstance(trained, EquinoxModelOp)
    jax.vmap(module)(_Pairs().x)
    for a, b in zip(before, _leaves(op.module), strict=True):
        np.testing.assert_array_equal(a, b)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(before, _leaves(trained.module), strict=True)
    )


def test_a_jax_model_op_comes_back_and_round_trips_through_a_registry(tmp_path) -> None:
    pipekit_jax = pytest.importorskip("pipekit_jax")
    experiment = pytest.importorskip("pipekit_experiment")

    trained, artifact = _loop(pipekit_jax.JaxModelOp(_mlp())).run()
    assert type(trained) is pipekit_jax.JaxModelOp
    assert artifact.backend_info["final_step"] == 20

    registry = experiment.LocalModelRegistry(tmp_path)
    ref = registry.store(trained, weights=trained.serialize_weights())
    reloaded = pipekit_jax.JaxModelOp(_mlp(seed=9)).from_registry(registry, ref)
    x = _Pairs().x
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(reloaded)(x)), np.asarray(jax.vmap(trained)(x))
    )


def test_a_jax_operator_comes_back_as_itself() -> None:
    pipekit_jax = pytest.importorskip("pipekit_jax")

    class Affine(pipekit_jax.JaxOperator):
        weight: jax.Array = eqx.field(converter=jnp.asarray)
        bias: jax.Array = eqx.field(converter=jnp.asarray)

        def _apply(self, x):
            return x @ self.weight + self.bias

    op = Affine(weight=np.zeros((3, 1), np.float32), bias=np.zeros(1, np.float32))
    trained, _ = _loop(op, max_steps=200, eval_every_n_steps=100).run()
    assert type(trained) is Affine
    np.testing.assert_allclose(
        np.asarray(trained.weight)[:, 0], [1.0, -2.0, 0.5], atol=0.1
    )


def test_a_dropout_module_trains_and_predicts_deterministically() -> None:
    module = eqx.nn.Sequential(
        [
            eqx.nn.Linear(3, 16, key=jax.random.key(1)),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Dropout(0.5),
            eqx.nn.Linear(16, 1, key=jax.random.key(2)),
        ]
    )
    trained, artifact = _loop(EquinoxModelOp(module), max_steps=30).run()
    assert np.isfinite(artifact.backend_info["final_metrics"]["loss"])
    inference = eqx.nn.inference_mode(trained.module)
    x = _Pairs().x
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(inference)(x)), np.asarray(jax.vmap(inference)(x))
    )


def test_keras_and_equinox_backends_train_the_same_weights() -> None:
    """200 Adam steps, one model and one batch order: float32 rounding apart."""
    pytest.importorskip("optax")
    pytest.importorskip("orbax.checkpoint")
    by_keras, _ = _loop(
        EquinoxModelOp(_mlp()), max_steps=200, eval_every_n_steps=200
    ).run()
    by_equinox, _ = _loop(
        EquinoxModelOp(_mlp()),
        backend="equinox",
        max_steps=200,
        eval_every_n_steps=200,
    ).run()
    for a, b in zip(_leaves(by_keras.module), _leaves(by_equinox.module), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_a_keras_model_holding_an_equinox_layer_round_trips_its_blob() -> None:
    def build(seed: int) -> Any:
        model = keras.Sequential([keras.layers.Dense(3), EquinoxLayer(_mlp(seed))])
        model(np.zeros((1, 3), np.float32))
        return model

    trained = KerasModelOp(build(0))
    trained.model.compile(optimizer=build_keras_optimizer({"name": "adam"}), loss="mse")
    data = _Pairs()
    trained.model.train_on_batch(data.x, data.y)
    loaded = KerasModelOp(build(5)).with_weights(trained.serialize_weights())
    np.testing.assert_allclose(
        np.asarray(loaded(data.x)), np.asarray(trained(data.x)), atol=1e-6
    )


def test_errors_name_what_is_wrong() -> None:
    from pipekit_train.backends.keras import KerasTask

    with pytest.raises(TypeError, match="eqx.Module"):
        EquinoxLayer(object())
    with pytest.raises(ValueError, match="keep_compile"):
        _loop(EquinoxModelOp(_mlp()), loss=None, task=KerasTask(keep_compile=True)).run()
