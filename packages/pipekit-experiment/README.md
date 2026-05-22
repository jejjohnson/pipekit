# pipekit-experiment

The boundary between pipekit pipelines and external orchestration
tools. Ships a content-addressed model registry, two
runtime-checkable protocols, JSON-roundtrippable reproducibility
artifacts, and per-tool adapters gated behind optional extras.

## Install

Core (registry + protocols + artifacts):

```bash
uv add pipekit-experiment
```

Per-tool adapters (install only what you use):

```bash
uv add 'pipekit-experiment[dvc]'        # dataset versioning via the DVC CLI
uv add 'pipekit-experiment[hydra]'      # hydra-core + hydra-zen config bridge
uv add 'pipekit-experiment[metaflow]'   # wrap operators as Metaflow @step
uv add 'pipekit-experiment[s3]'         # fsspec-backed S3ModelRegistry
```

The pure-pipekit surface remains importable even when no extras are
installed; each adapter raises a clean `ImportError` with the install
hint on first use.

## What's shipped (v0.0.1)

| Module                | Symbols                                                            |
|-----------------------|--------------------------------------------------------------------|
| `protocols`           | `ExperimentTracker`, `ModelRegistry` (runtime-checkable)           |
| `run`                 | `Run`, `RunMetrics`, `RunArtifacts`                                |
| `registry`            | `LocalModelRegistry`, `S3ModelRegistry`                            |
| `artifacts`           | `TrainingArtifact`, `InferenceArtifact`                            |
| `adapters.dvc`        | `DVCDatasetVersioning`                                             |
| `adapters.hydra`      | `HydraConfigLoader`                                                |
| `adapters.metaflow`   | `MetaflowStepAdapter`                                              |

MLflow and W&B adapters are planned for v0.0.2.

## Quickstart — model registry

```python
import pipekit_experiment as pe

registry = pe.LocalModelRegistry("/tmp/models")

# Store; the hash is the canonical identifier
h = registry.store(trained_op, name="methane_emulator_v3",
                   tags={"family": "methane"})

# Load by hash, or by tag name
op = registry.load(h)
op = registry.load("methane_emulator_v3")

# Atomic tag promotion
registry.tag(h, "production", force=True)
```

## Quickstart — Hydra round-trip

```python
from pipekit_experiment.adapters.hydra import HydraConfigLoader

yaml_text = HydraConfigLoader.to_yaml(my_op)        # via hydra-zen.builds
restored  = HydraConfigLoader.from_yaml(yaml_text)  # via hydra.utils.instantiate
```

## Quickstart — Metaflow step

```python
from metaflow import FlowSpec
from pipekit_experiment.adapters.metaflow import MetaflowStepAdapter


class MyFlow(FlowSpec):
    preprocess = MetaflowStepAdapter.as_step(
        my_preprocess_op, name="preprocess", inputs=["raw"]
    )
    train = MetaflowStepAdapter.as_step(
        my_training_loop, name="train", inputs=["preprocess"]
    )
```

## References

Master plan: [Report 12](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_10_pipekit_experiment.md).
API reference: [pipekit-experiment](https://github.com/jejjohnson/pipekit/blob/main/docs/api/pipekit-experiment.md).
