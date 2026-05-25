---
status: draft
version: 0.1.0
---

# API Overview — Core Abstractions

The public surface of `pipekit-train`. Six concepts; each maps to one
detail file below.

## Concepts at a glance

| Concept            | Layer | Module               | Form                       |
|--------------------|-------|----------------------|----------------------------|
| `TrainingDataset`  | 0     | `dataset`            | `Operator` base class      |
| `CatalogDataset`   | 0     | `dataset`            | `Operator` subclass        |
| `SimulationDataset`| 0     | `dataset`            | `Operator` subclass        |
| `CachedDataset`    | 0     | `dataset`            | `Operator` subclass        |
| `Loss`             | 1     | `loss`               | `Protocol`                 |
| `MSE`/`NLL`/`KL`   | 1     | `loss`               | `Operator` subclasses      |
| `Composite`        | 1     | `loss`               | `Operator` subclass        |
| `Callback`         | 2     | `callbacks`          | `Protocol`                 |
| `Checkpoint`       | 2     | `callbacks`          | `Operator` (Callback impl) |
| `EarlyStopping`    | 2     | `callbacks`          | `Operator` (Callback impl) |
| `LogToExperiment`  | 2     | `callbacks`          | `Operator` (Callback impl) |
| `MetricWriter`     | 2     | `writer`             | `Protocol`                 |
| `JSONLWriter`      | 2     | `writer`             | dataclass (Writer impl)    |
| `TrainerCarryState`| 3     | `loop`               | `CarryState` subclass      |
| `ValidationStep`   | 3     | `loop`               | `Operator`                 |
| `TrainingLoop`     | 3     | `loop`               | `StatefulOperator`         |
| `adapters.equinox` | 4     | `adapters/equinox`   | `run(loop) -> (op, info)`  |
| `adapters.lightning`| 4    | `adapters/lightning` | scaffold (v0.2)            |
| `adapters.keras`   | 4     | `adapters/keras`     | scaffold (v0.3)            |

## Import conventions

```python
# Layer 0 — datasets
from pipekit_train import (
    TrainingDataset,
    CatalogDataset,
    SimulationDataset,
    CachedDataset,
)

# Layer 1 — losses
from pipekit_train import Loss, MSE, NLL, KL, Composite

# Layer 2 — callbacks and writers
from pipekit_train import (
    Callback,
    Checkpoint,
    EarlyStopping,
    LogToExperiment,
    MetricWriter,
    JSONLWriter,
)

# Layer 3 — the training loop
from pipekit_train import TrainingLoop, ValidationStep, TrainerCarryState

# Layer 4 — backend adapters (extras-gated)
from pipekit_train.adapters import equinox as eqx_adapter
# eqx_adapter.run(loop)
```

The `pipekit_train` namespace re-exports everything from Layers 0–3.
Layer 4 adapters are explicit submodule imports — calling
`from pipekit_train import adapters` does not eagerly import torch /
jax / tensorflow.

## Detail files

| File                          | Covers                                              |
|-------------------------------|-----------------------------------------------------|
| [datasets.md](datasets.md)    | Layer 0 — Dataset operators + content hashing      |
| [primitives.md](primitives.md)| Layer 1 — Loss surface + per-adapter primitives    |
| [components.md](components.md)| Layer 2 — Callbacks, MetricWriter, ValidationStep  |
| [models.md](models.md)        | Layer 3 — TrainingLoop, TrainerCarryState          |
| [adapters.md](adapters.md)    | Layer 4 — Equinox reference + Lightning/Keras stubs|

For usage patterns, see [../examples/](../examples/).
