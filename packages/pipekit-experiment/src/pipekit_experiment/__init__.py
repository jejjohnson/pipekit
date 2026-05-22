"""pipekit-experiment — experiment tracking and content-addressed model registry.

The boundary between pipekit pipelines and external orchestration
tools (DVC, Hydra, Metaflow, …). Two pieces:

- **Protocols** (`protocols`): `ExperimentTracker`, `ModelRegistry`
  — runtime-checkable seams that adapters satisfy structurally.
- **Run records** (`run`): `Run`, `RunMetrics`, `RunArtifacts`.
- **Model registry** (`registry`): `LocalModelRegistry` (always
  available); `S3ModelRegistry` (behind ``[s3]`` extra, fsspec-backed).
- **Artifacts** (`artifacts`): `TrainingArtifact`, `InferenceArtifact`.

Adapters live under `pipekit_experiment.adapters.<tool>` and are
gated behind their tool's optional extra:

- `adapters.dvc.DVCDatasetVersioning` — ``[dvc]``.
- `adapters.hydra.HydraConfigLoader` — ``[hydra]``.
- `adapters.metaflow.MetaflowStepAdapter` — ``[metaflow]``.

See master plan Report 12.
"""

from pipekit_experiment.artifacts import InferenceArtifact, TrainingArtifact
from pipekit_experiment.protocols import ExperimentTracker, ModelRegistry
from pipekit_experiment.registry import LocalModelRegistry, S3ModelRegistry
from pipekit_experiment.run import Run, RunArtifacts, RunMetrics


__all__ = [
    "ExperimentTracker",
    "InferenceArtifact",
    "LocalModelRegistry",
    "ModelRegistry",
    "Run",
    "RunArtifacts",
    "RunMetrics",
    "S3ModelRegistry",
    "TrainingArtifact",
]

__version__ = "0.1.0"  # x-release-please-version
