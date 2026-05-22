"""Reproducibility artifacts — training and inference snapshots.

- `TrainingArtifact` — captures everything needed to rebuild a
  trained model: the training pipeline YAML, dataset hash, trained
  model hash, tracker run id, registry URI, backend info, dep lock.
- `InferenceArtifact` — pins the trained-model hashes referenced by
  the inference pipeline. Subclass of `TrainingArtifact` is overkill;
  it's a separate dataclass.

Both serialise as JSON via `save` / `load`. They're plain data carriers
— `rerun` and `reload_model` are convenience helpers that delegate
to a supplied registry / loader.

See master plan Report 12, section 2.5.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pipekit import Operator


@dataclass
class TrainingArtifact:
    """Reproducibility snapshot for a training run.

    Attributes:
        training_pipeline_yaml: Serialised training pipeline (the
            pipekit-train `TrainingLoop` config, typically as YAML
            or JSON text).
        dataset_hash: Content hash of the training dataset.
        trained_model_hash: Content hash of the trained model
            (registry identifier).
        tracker_run_id: External-tracker run reference (e.g.
            ``"mlflow://runs/abc123"``). ``None`` if no tracker was
            attached.
        model_registry_uri: URI under which the model was stored.
        backend_info: Free-form dict describing the training stack
            (lightning version, JAX version, GPUs, …).
        deps_lock: Contents of the uv / poetry lockfile.
        metadata: Optional arbitrary metadata (author, date, notes).
    """

    training_pipeline_yaml: str
    dataset_hash: str
    trained_model_hash: str
    tracker_run_id: str | None
    model_registry_uri: str
    backend_info: dict[str, Any] = field(default_factory=dict)
    deps_lock: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Write the artifact to ``path`` as JSON."""
        Path(path).write_text(json.dumps(asdict(self), sort_keys=True, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> TrainingArtifact:
        """Read an artifact previously written by `save`."""
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def reload_model(self, registry: Any) -> Operator:
        """Resolve the stored model via ``registry.load``.

        Args:
            registry: Anything satisfying the `ModelRegistry` protocol
                (typically a `LocalModelRegistry` or `S3ModelRegistry`).
        """
        return registry.load(self.trained_model_hash)


@dataclass
class InferenceArtifact:
    """Reproducibility snapshot for an inference pipeline.

    Pins the trained-model hashes referenced by the pipeline so a
    pipekit YAML round-trip resolves to the exact same weights.

    Attributes:
        pipeline_yaml: Serialised inference pipeline.
        pinned_model_hashes: ``operator-path → hash`` for every
            registry-resolved operator in the pipeline.
        backend_info: Backend-stack info.
        deps_lock: Lockfile contents.
        metadata: Arbitrary metadata.
    """

    pipeline_yaml: str
    pinned_model_hashes: dict[str, str] = field(default_factory=dict)
    backend_info: dict[str, Any] = field(default_factory=dict)
    deps_lock: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), sort_keys=True, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> InferenceArtifact:
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def reload_models(self, registry: Any) -> dict[str, Operator]:
        """Resolve every pinned model via ``registry.load``.

        Returns a ``{operator-path: Operator}`` dict for callers to
        substitute back into the pipeline.
        """
        return {path: registry.load(h) for path, h in self.pinned_model_hashes.items()}
