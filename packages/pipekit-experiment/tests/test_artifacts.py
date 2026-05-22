"""Tests for `pipekit_experiment.artifacts`."""

from __future__ import annotations

from pipekit import Const
from pipekit_experiment import (
    InferenceArtifact,
    LocalModelRegistry,
    TrainingArtifact,
)


def _make_training_artifact() -> TrainingArtifact:
    return TrainingArtifact(
        training_pipeline_yaml="loop: stuff",
        dataset_hash="ab12",
        trained_model_hash="cd34",
        tracker_run_id="mlflow://runs/xyz",
        model_registry_uri="s3://bucket/",
        backend_info={"backend": "lightning", "version": "2.4.0"},
        deps_lock="uv.lock contents",
        metadata={"author": "ej"},
    )


def test_training_artifact_save_load_round_trip(tmp_path):
    art = _make_training_artifact()
    path = tmp_path / "art.json"
    art.save(path)
    loaded = TrainingArtifact.load(path)
    assert loaded == art


def test_training_artifact_reload_model(tmp_path):
    reg = LocalModelRegistry(tmp_path / "registry")
    h = reg.store(Const(7))
    art = TrainingArtifact(
        training_pipeline_yaml="",
        dataset_hash="d",
        trained_model_hash=h,
        tracker_run_id=None,
        model_registry_uri=str(tmp_path / "registry"),
    )
    op = art.reload_model(reg)
    assert op() == 7


def test_inference_artifact_save_load_round_trip(tmp_path):
    art = InferenceArtifact(
        pipeline_yaml="pipe: stuff",
        pinned_model_hashes={"step1.model": "ab12", "step2.model": "cd34"},
        metadata={"author": "ej"},
    )
    path = tmp_path / "inf.json"
    art.save(path)
    loaded = InferenceArtifact.load(path)
    assert loaded == art


def test_inference_artifact_reload_models(tmp_path):
    reg = LocalModelRegistry(tmp_path / "registry")
    h1 = reg.store(Const(1))
    h2 = reg.store(Const(2))
    art = InferenceArtifact(
        pipeline_yaml="",
        pinned_model_hashes={"a": h1, "b": h2},
    )
    models = art.reload_models(reg)
    assert models["a"]() == 1
    assert models["b"]() == 2
