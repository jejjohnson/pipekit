---
status: draft
version: 0.1.0
---

# Example — Cross-package integration

How pipekit-train composes with the rest of the GeoStack — the
patterns that emerge once the package is in the workspace.

## Train → Cycle (DA)

The strongest integration. Train an emulator with pipekit-train,
drop it into a `DACycle` from pipekit-cycle, run a 30-day
operational forecast cycle with daily assimilation.

```python
import pipekit_train as pt
import pipekit_cycle as pc

emulator_op, _ = pt.TrainingLoop(...).run()
emulator_forward = pc.NeuralForward(model_op=emulator_op, dt=86400.0)

# Same DACycle that would have used the expensive forward model.
da_cycle = pc.DACycle(
    forward_model=emulator_forward,
    obs_op=tropomi_obs_op,
    analysis_step=enkf_analysis,
    obs_err_cov=R,
)
trajectory = da_cycle.run(initial_state, observations_30days, n_cycles=30)
```

## Train → Experiment registry → Inference YAML

A "regulatory replay" workflow. The trained model is content-addressed;
the inference pipeline pins its hash; the whole pipeline is then a
YAML round-trip.

```python
import pipekit as pk
import pipekit_train as pt
import pipekit_experiment as pe

# Train.
registry = pe.LocalModelRegistry(root="./models")
trained_op, artifact = pt.TrainingLoop(
    ...,
    callbacks=(pt.Checkpoint(..., registry=registry),),
).run()
artifact.save("./artifacts/train_v3.json")

# Build the inference pipeline.
inference = pk.Sequential([
    preprocess,
    trained_op,
    postprocess,
])

# Pin the model hash into an InferenceArtifact for byte-identical replay.
inf_artifact = pe.InferenceArtifact(
    pipeline_yaml=pk.dumps(inference),
    pinned_model_hashes={"step[1]": artifact.trained_model_hash},
)
inf_artifact.save("./artifacts/inference_v3.json")

# Months later, on a fresh machine:
loaded = pe.InferenceArtifact.load("./artifacts/inference_v3.json")
models = loaded.reload_models(registry)
# `models["step[1]"]` is byte-identical to `trained_op` above.
```

## Train with VarDA (vardax)

Variational data assimilation networks (VarDANet from `vardax`)
trained as a `TrainingLoop`. The Equinox adapter's `task=` argument
lets the user bypass the synthesised loss and supply a domain-specific
`TrainTask` directly — `vardax.training.train_loss` is exactly the
right shape.

```python
import pipekit_train as pt
from pipekit_train.adapters.equinox import TrainTask
from vardax.model import VarDANet2D
from vardax.training import train_loss

class VarDATrainTask:
    """Adapts vardax.train_loss to the Equinox TrainTask Protocol."""
    def loss_fn(self, model, batch, key):
        loss = train_loss(model, batch)
        return loss, {"reconstruction_mse": loss}

assert isinstance(VarDATrainTask(), TrainTask)   # structural check

loop = pt.TrainingLoop(
    model_op=pj.JaxModelOp(VarDANet2D(...)),
    dataset=...,
    task=VarDATrainTask(),       # ← overrides the auto-synthesised path
    optimizer_config={"name": "adam", "lr": 1e-3},
    max_steps=50_000,
    backend="equinox",
    ...
)
trained_vardanet, _ = loop.run()
```

This pattern works for any external Equinox model whose loss has the
shape `loss_fn(model, batch, key) → (loss, aux)` — somax,
vardax, pyrox-nn feature extractors, custom equinox models. The
adapter sees a `TrainTask`; pipekit-train sees an `Operator`.

## Train a somax ocean model

Learn ocean model parameters by differentiating through the solver.
Same pattern as VarDA — supply a `TrainTask` that closes over the
somax solver call.

```python
import pipekit_train as pt
from pipekit_train.adapters.equinox import TrainTask
import somax

class OceanParamTask:
    """Learn ocean model parameters by differentiating through the solver."""
    def loss_fn(self, model, batch, key):
        trajectory = model.integrate(batch["state0"], t0=0, t1=T, dt=dt)
        loss = jnp.mean((trajectory.ys - batch["observations"])**2)
        return loss, {"mse": loss}

loop = pt.TrainingLoop(
    model_op=pj.JaxModelOp(somax.SomaxModel(...)),
    dataset=...,
    task=OceanParamTask(),
    optimizer_config={"name": "adam", "lr": 1e-3},
    max_steps=5_000,
    backend="equinox",
)
trained_somax, _ = loop.run()
```

## Custom `MetricWriter` (W&B)

The `MetricWriter` Protocol is ten lines for any backend tracker:

```python
import wandb

class WandbWriter:
    """MetricWriter implementation for W&B."""

    def __init__(self, project: str, name: str) -> None:
        wandb.init(project=project, name=name)

    def write(self, step: int, metrics: dict[str, float]) -> None:
        wandb.log(metrics, step=step)

    def close(self) -> None:
        wandb.finish()

loop = pt.TrainingLoop(..., metric_writer=WandbWriter("plumes", "v3"))
```

When `pipekit-experiment` ships a `WandbTracker` adapter (planned in
its master plan §3.4), the same flow goes via `LogToExperiment`
instead and the W&B artifact URI lands in the `TrainingArtifact`.

## Composition pattern summary

| Pattern                       | Components                                      | Use case                            |
|-------------------------------|-------------------------------------------------|-------------------------------------|
| Direct supervised             | `CatalogDataset + TrainingLoop`                 | Cloud mask, super-resolution        |
| Emulator training             | `SimulationDataset(cycle=…) + TrainingLoop`     | Forward-model surrogates            |
| Amortized inference           | `SimulationDataset(obs_op=…) + TrainingLoop`    | SBI, neural posterior, NRE          |
| Pretrain → fine-tune          | `Sequential([TrainingLoop, TrainingLoop])`      | Self-supervised → supervised        |
| Train → Cycle handoff         | `TrainingLoop → NeuralForward → Cycle`          | Fast operational forecast           |
| Train → DA handoff            | `TrainingLoop → NeuralForward → DACycle`        | ML-augmented data assimilation      |
| Train → Registry → Inference  | `TrainingLoop → ModelRegistry → Sequential`     | Regulatory replay, audit trail      |
| Domain-specific loss          | `TrainingLoop(task=CustomTrainTask())`          | VarDA, somax, pyrox-nn              |
| Hyperparameter sweep (v0.2)   | `HyperSweep + LogToExperiment + Optuna`         | Architecture search                 |
