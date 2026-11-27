"""`KerasModelOp` and its weight blob (`pipekit_train.backends.keras`, ADR D20)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest


if importlib.util.find_spec("keras") is None:
    pytest.skip("needs the [keras] extra", allow_module_level=True)

from pipekit_train.backends.keras import KerasModelOp, import_keras


keras = import_keras()


def _model(seed: int, width: int = 8) -> keras.Model:
    keras.utils.set_random_seed(seed)
    return keras.Sequential(
        [
            keras.Input((3,)),
            keras.layers.Dense(width, activation="tanh"),
            keras.layers.Dense(1),
        ]
    )


def _inputs() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(16, 3)).astype("float32")


def test_calls_the_model_in_inference_mode() -> None:
    model = _model(0)
    op = KerasModelOp(model)
    np.testing.assert_array_equal(
        np.asarray(op(_inputs())), np.asarray(model(_inputs(), training=False))
    )


def test_rejects_what_is_not_a_keras_model() -> None:
    with pytest.raises(TypeError, match=r"keras\.Model"):
        KerasModelOp(object())


def test_blob_is_byte_stable() -> None:
    op = KerasModelOp(_model(0))
    assert op.serialize_weights() == op.serialize_weights()
    assert op.serialize_weights().startswith(b"PKKRW1\n")


def test_blob_loads_into_a_skeleton_built_with_another_seed() -> None:
    trained = KerasModelOp(_model(0))
    skeleton = KerasModelOp(_model(123))
    loaded = skeleton.with_weights(trained.serialize_weights())
    np.testing.assert_array_equal(
        np.asarray(loaded(_inputs())), np.asarray(trained(_inputs()))
    )
    # the skeleton itself is untouched
    assert not np.allclose(
        np.asarray(skeleton(_inputs())), np.asarray(trained(_inputs()))
    )
    assert loaded.serialize_weights() == trained.serialize_weights()


def test_a_scalar_weight_keeps_its_shape() -> None:
    class Scale(keras.layers.Layer):
        def build(self, input_shape):
            self.scale = self.add_weight(shape=(), initializer="ones", name="scale")

        def call(self, x):
            return x * self.scale

    def build(seed: int) -> keras.Model:
        keras.utils.set_random_seed(seed)
        model = keras.Sequential([keras.Input((3,)), Scale(), keras.layers.Dense(1)])
        model.layers[0].scale.assign(2.0 + seed)
        return model

    trained = KerasModelOp(build(0))
    blob = trained.serialize_weights()
    loaded = KerasModelOp(build(1)).with_weights(blob, copy=False)
    assert float(loaded.model.layers[0].scale.numpy()) == 2.0
    assert trained.with_weights(blob, copy=False).serialize_weights() == blob


def test_copy_false_loads_into_the_skeleton_itself() -> None:
    trained = KerasModelOp(_model(0))
    skeleton = KerasModelOp(_model(123))
    loaded = skeleton.with_weights(trained.serialize_weights(), copy=False)
    assert loaded.model is skeleton.model
    np.testing.assert_array_equal(
        np.asarray(skeleton(_inputs())), np.asarray(trained(_inputs()))
    )


def test_a_skeleton_of_another_shape_is_refused() -> None:
    blob = KerasModelOp(_model(0, width=8)).serialize_weights()
    with pytest.raises(ValueError, match="does not match"):
        KerasModelOp(_model(1, width=4)).with_weights(blob)
    with pytest.raises(ValueError, match="not a KerasModelOp weight blob"):
        KerasModelOp(_model(1)).with_weights(b"not a blob")
    with pytest.raises(TypeError, match="bytes"):
        KerasModelOp(_model(1)).with_weights("text")  # type: ignore[arg-type]


def test_round_trip_through_a_local_registry(tmp_path) -> None:
    pytest.importorskip("pipekit_experiment")
    from pipekit_experiment import LocalModelRegistry

    registry = LocalModelRegistry(tmp_path)
    trained = KerasModelOp(_model(0))
    ref = registry.store(trained, weights=trained.serialize_weights())
    reloaded = KerasModelOp(_model(7)).from_registry(registry, ref)
    np.testing.assert_array_equal(
        np.asarray(reloaded(_inputs())), np.asarray(trained(_inputs()))
    )
    # the same weights store under the same hash
    assert registry.store(trained, weights=trained.serialize_weights()) == ref


def test_importing_the_area_imports_no_keras() -> None:
    code = (
        "import sys; import pipekit_train.backends.keras; print('keras' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_keras_on_another_backend_is_refused() -> None:
    code = (
        "import os; os.environ['KERAS_BACKEND'] = 'numpy'; import keras; "
        "from pipekit_train.backends.keras import import_keras\n"
        "try:\n    import_keras()\nexcept RuntimeError as e:\n    print('refused', e)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.startswith("refused")
    assert "JAX only" in out.stdout


def test_first_use_chooses_the_jax_backend() -> None:
    code = (
        "import os; os.environ.pop('KERAS_BACKEND', None)\n"
        "from pipekit_train.backends.keras import import_keras\n"
        "print(import_keras().config.backend())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "jax"
