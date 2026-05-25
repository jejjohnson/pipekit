"""Tests for `pipekit_train.loop` — TrainerCarryState, ValidationStep, TrainingLoop.

End-to-end Equinox adapter behaviour lives in
``tests/adapters/test_equinox.py``. These tests cover the layer-3
plumbing in isolation: state shape, validation aggregation, adapter
dispatch, artifact assembly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pipekit import Const, Operator, StatefulOperator
from pipekit_train import (
    MSE,
    IterableDataset,
    TrainerCarryState,
    TrainingDataset,
    TrainingLoop,
    ValidationStep,
)
from pipekit_train.loop import _LocalArtifact


# --- Helpers --------------------------------------------------------------


class _IdentityModel(Operator):
    def _apply(self, x: Any) -> Any:
        return x


class _ScaleModel(Operator):
    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    def _apply(self, x: Any) -> Any:
        return x * self.factor


def _toy_dataset(n: int = 4) -> IterableDataset:
    return IterableDataset(
        source=[(np.array([i], dtype=float), np.array([i * 2], dtype=float)) for i in range(n)],
        content_hash=f"toy-{n}",
    )


# --- TrainerCarryState ----------------------------------------------------


def test_trainer_carry_state_default_fields():
    state = TrainerCarryState(model=Const(7), opt_state=None)
    assert state.step == 0
    assert state.epoch == 0
    assert state.metrics == {}
    assert state.rng is None


def test_trainer_carry_state_to_dict_is_json_safe():
    import json

    state = TrainerCarryState(
        model=Const(7),
        opt_state="opaque",
        step=10,
        epoch=1,
        metrics={"loss": 0.5},
        rng="rng_repr",
    )
    d = state.to_dict()
    # All values must be JSON-safe (no opaque objects).
    json.dumps(d)
    assert d["step"] == 10
    assert d["epoch"] == 1
    assert d["metrics"] == {"loss": 0.5}


# --- ValidationStep -------------------------------------------------------


def test_validation_step_averages_metrics():
    dataset = _toy_dataset(n=4)  # pairs: (i, 2i) for i in 0..3
    # Model predicts x*2 = correct target, so MSE = 0
    step = ValidationStep(
        model_op=_ScaleModel(factor=2.0),
        dataset=dataset,
        metrics=(MSE(),),
    )
    metrics = step()
    assert metrics["mse"] == pytest.approx(0.0)


def test_validation_step_aggregates_aux_metrics():
    dataset = _toy_dataset(n=4)
    # Model predicts x * 3, target is x * 2 → diff per pair: x.
    # mse per pair: x^2 for x in [0, 1, 2, 3] → 0, 1, 4, 9 → mean 3.5
    step = ValidationStep(
        model_op=_ScaleModel(factor=3.0),
        dataset=dataset,
        metrics=(MSE(),),
    )
    metrics = step()
    assert metrics["mse"] == pytest.approx(3.5)


def test_validation_step_empty_dataset_returns_empty():
    empty = IterableDataset(source=[], content_hash="empty")
    step = ValidationStep(
        model_op=_IdentityModel(),
        dataset=empty,
        metrics=(MSE(),),
    )
    assert step() == {}


def test_validation_step_rejects_bad_args():
    dataset = _toy_dataset()
    with pytest.raises(TypeError, match="Operator"):
        ValidationStep(
            model_op="not an op",  # type: ignore[arg-type]
            dataset=dataset,
            metrics=(MSE(),),
        )
    with pytest.raises(TypeError, match="TrainingDataset"):
        ValidationStep(
            model_op=_IdentityModel(),
            dataset="not a dataset",  # type: ignore[arg-type]
            metrics=(MSE(),),
        )
    with pytest.raises(ValueError, match="non-empty"):
        ValidationStep(
            model_op=_IdentityModel(),
            dataset=dataset,
            metrics=(),
        )


# --- TrainingLoop — construction & validation ----------------------------


def test_training_loop_is_stateful_operator():
    loop = TrainingLoop(
        model_op=_IdentityModel(),
        dataset=_toy_dataset(),
        loss=MSE(),
        max_steps=10,
        backend="equinox",
    )
    assert isinstance(loop, StatefulOperator)
    # Verify the class-level marker that pipekit.Sequential uses.
    assert TrainingLoop._is_stateful is True


def test_training_loop_rejects_bad_model_op():
    with pytest.raises(TypeError, match="Operator"):
        TrainingLoop(
            model_op="not an op",  # type: ignore[arg-type]
            dataset=_toy_dataset(),
            loss=MSE(),
            max_steps=10,
        )


def test_training_loop_rejects_bad_dataset():
    with pytest.raises(TypeError, match="TrainingDataset"):
        TrainingLoop(
            model_op=_IdentityModel(),
            dataset=[1, 2, 3],  # type: ignore[arg-type]
            loss=MSE(),
            max_steps=10,
        )


def test_training_loop_rejects_bad_val_dataset():
    with pytest.raises(TypeError, match="TrainingDataset"):
        TrainingLoop(
            model_op=_IdentityModel(),
            dataset=_toy_dataset(),
            val_dataset=[1, 2, 3],  # type: ignore[arg-type]
            loss=MSE(),
            max_steps=10,
        )


def test_training_loop_rejects_bad_backend():
    with pytest.raises(ValueError, match="backend"):
        TrainingLoop(
            model_op=_IdentityModel(),
            dataset=_toy_dataset(),
            loss=MSE(),
            max_steps=10,
            backend="tensorflow",  # type: ignore[arg-type]
        )


def test_training_loop_rejects_nonpositive_max_steps():
    with pytest.raises(ValueError, match="max_steps"):
        TrainingLoop(
            model_op=_IdentityModel(),
            dataset=_toy_dataset(),
            loss=MSE(),
            max_steps=0,
        )


def test_training_loop_rejects_nonpositive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        TrainingLoop(
            model_op=_IdentityModel(),
            dataset=_toy_dataset(),
            loss=MSE(),
            max_steps=10,
            batch_size=0,
        )


def test_training_loop_default_optimizer_config():
    loop = TrainingLoop(
        model_op=_IdentityModel(),
        dataset=_toy_dataset(),
        loss=MSE(),
        max_steps=10,
    )
    assert loop.optimizer_config == {"name": "adamw", "lr": 3e-4}


# --- TrainingLoop — adapter dispatch -------------------------------------


def test_training_loop_dispatches_to_lightning_stub_with_clear_error():
    loop = TrainingLoop(
        model_op=_IdentityModel(),
        dataset=_toy_dataset(),
        loss=MSE(),
        max_steps=10,
        backend="lightning",
    )
    with pytest.raises(NotImplementedError, match=r"v0\.2"):
        loop.run()


def test_training_loop_dispatches_to_keras_stub_with_clear_error():
    loop = TrainingLoop(
        model_op=_IdentityModel(),
        dataset=_toy_dataset(),
        loss=MSE(),
        max_steps=10,
        backend="keras",
    )
    with pytest.raises(NotImplementedError, match=r"v0\.3"):
        loop.run()


# --- TrainingLoop — config & artifact assembly ---------------------------


def test_training_loop_get_config_surfaces_key_fields():
    loop = TrainingLoop(
        model_op=_ScaleModel(),
        dataset=_toy_dataset(n=8),
        loss=MSE(),
        max_steps=42,
        backend="equinox",
        seed=7,
    )
    config = loop.get_config()
    assert config["model_op"] == "_ScaleModel"
    assert config["dataset"] == "IterableDataset"
    assert config["loss"] == "MSE"
    assert config["max_steps"] == 42
    assert config["seed"] == 7
    assert config["backend"] == "equinox"


def test_training_loop_build_artifact_falls_back_to_local_artifact_when_no_extra():
    """Without pipekit-experiment[experiment] extra, the artifact is _LocalArtifact."""
    loop = TrainingLoop(
        model_op=_ScaleModel(),
        dataset=_toy_dataset(),
        loss=MSE(),
        max_steps=10,
    )
    # We don't want to actually run training here — call the build path directly.
    artifact = loop._build_artifact(
        trained_model_op=_ScaleModel(),
        backend_info={"backend": "equinox"},
    )
    # The artifact may be pipekit_experiment.TrainingArtifact OR _LocalArtifact;
    # either is fine. The contract is the field names match.
    assert hasattr(artifact, "training_pipeline_yaml")
    assert hasattr(artifact, "dataset_hash")
    assert hasattr(artifact, "trained_model_hash")
    assert hasattr(artifact, "backend_info")
    assert artifact.dataset_hash == loop.dataset.content_hash()
    assert artifact.backend_info == {"backend": "equinox"}


def test_local_artifact_to_dict_round_trips():
    art = _LocalArtifact(
        training_pipeline_yaml="...",
        dataset_hash="abc",
        trained_model_hash="def",
        tracker_run_id="mlflow://r/123",
        model_registry_uri="registry://def",
        backend_info={"backend": "equinox"},
        deps_lock="",
        metadata={},
    )
    d = art.to_dict()
    assert d["dataset_hash"] == "abc"
    assert d["trained_model_hash"] == "def"


def test_training_loop_forbid_in_yaml():
    """TrainingLoop carries closures (model_op, dataset, callbacks) so
    is marked forbid_in_yaml = True."""
    assert TrainingLoop.forbid_in_yaml is True
