---
status: draft
version: 0.1.0
---

# Architecture

## 1. Where pipekit-train sits in the stack

```
   Domain libraries          geotoolz │ xr_toolz
                                  ▲
                                  │
   Infrastructure       ┌─ pipekit-cycle ─┐
                        │  pipekit-train  │ ← this design
                        │  pipekit-experiment │
                        └─ statecatalog ──┘
                                  ▲
                                  │
   Framework                   pipekit ◄── pipekit-array
                                  ▲
                                  │
   Substrate              georeader │ xarray ecosystem
```

Sibling of `pipekit-cycle` and `pipekit-experiment`. Depends only on
`pipekit` directly; algorithm libraries (`filterx`, `vardax`,
`pyrox`, `gaussflowx`, `somax`) plug in via their own
`[pipekit_train]` extras — the same pattern as everywhere else.

## 2. Layered design

Five layers, each composable with what's above and below. Each layer
maps to one file in `api/`.

```
+--------------------------------------------------------------+
| Layer 4 — Adapters         pipekit_train.adapters.equinox    |
|                            pipekit_train.adapters.lightning  |
|                            pipekit_train.adapters.keras      |
+--------------------------------------------------------------+
| Layer 3 — TrainingLoop     loop.py — StatefulOperator        |
|                            outer loop, validation, artifact  |
+--------------------------------------------------------------+
| Layer 2 — Components       callbacks.py, sweep.py            |
|                            EarlyStopping, Checkpoint,        |
|                            LogToExperiment, MetricWriter     |
+--------------------------------------------------------------+
| Layer 1 — Primitives       loss.py — Loss Protocol + MSE/NLL |
|                            (per-adapter: train_step,         |
|                             save_state, restore_state)       |
+--------------------------------------------------------------+
| Layer 0 — Datasets         dataset.py                        |
|                            TrainingDataset, CatalogDataset,  |
|                            SimulationDataset, CachedDataset  |
+--------------------------------------------------------------+
```

Each layer can be used standalone. A user who wants only the
carrier-agnostic dataset abstractions can use Layer 0 and pass the
loaders to their own training loop. A user who wants the full
pipekit experience uses Layer 3 with a Layer 4 adapter.

## 3. Source layout

```
packages/pipekit-train/
├── pyproject.toml             # extras: [equinox], [lightning], [keras],
│                              #         [catalog], [cycle], [experiment]
├── README.md
├── CHANGELOG.md
├── src/pipekit_train/
│   ├── __init__.py            # public re-exports
│   ├── dataset.py             # Layer 0
│   ├── loss.py                # Layer 1 — carrier-agnostic Loss surface
│   ├── callbacks.py           # Layer 2 — Callback / Checkpoint /
│   │                          #           EarlyStopping / LogToExperiment
│   ├── writer.py              # Layer 2 — MetricWriter Protocol +
│   │                          #           JSONL impl
│   ├── loop.py                # Layer 3 — TrainingLoop (StatefulOperator),
│   │                          #           ValidationStep, TrainerCarryState
│   ├── sweep.py               # Layer 2-3 — HyperSweep (deferred to v0.2)
│   └── adapters/
│       ├── __init__.py
│       ├── equinox.py         # v0.1 reference adapter
│       ├── lightning.py       # scaffold — raises NotImplementedError
│       └── keras.py           # scaffold — raises NotImplementedError
├── tests/
└── docs/
    └── design/                # this design doc
```

## 4. Integration seams

Three seams matter most.

### 4.1 Seam to pipekit core — `StatefulOperator`

`TrainingLoop` is a `pipekit.StatefulOperator` (Group M). Its
`_apply(carrier, state) -> (carrier, state)` signature lets it
participate in any `Sequential` that threads state through. The
carry-state is a `TrainerCarryState(CarryState)` carrying:

- `model` — the (possibly partially-trained) model operator
- `opt_state` — backend-specific optimizer state
- `step` — int, global step counter
- `epoch` — int, derived from step / steps_per_epoch
- `metrics` — latest per-step metric snapshot
- `rng` — backend-agnostic PRNG state (a seed + counter)

That gives `TrainingLoop.run()` the same JSON round-trip discipline
as every other stateful pipekit operator. Checkpointing is then just
"serialise the carry-state plus the model weights".

### 4.2 Seam to pipekit-cycle — `ForwardModel` as dataset source

`SimulationDataset` accepts any object satisfying the
`pipekit_cycle.ForwardModel` Protocol. The dataset's content hash
includes the forward model's `state_signature` and `dt` so two
datasets generated from structurally identical forward models hash to
the same value. When the dataset's optional `cycle` argument is set,
it uses a `pipekit_cycle.Cycle` to roll out trajectories rather than
single forward steps.

The reverse seam — the trained emulator dropping back into
`pipekit-cycle` — is via `pipekit_cycle.NeuralForward`. The trained
`ModelOp` returned by `loop.run()` is passed to `NeuralForward(model_op,
dt=...)` and immediately substitutes for the expensive forward model
in any existing `Cycle` / `DACycle`.

### 4.3 Seam to pipekit-experiment — `TrainingArtifact` + registry

`pipekit-experiment` already ships the artifact contract
(`pipekit_experiment.artifacts.TrainingArtifact`). `TrainingLoop.run()`
returns a tuple `(trained_model_op, training_artifact)`. The artifact
contains:

- `training_pipeline_yaml` — the serialised `TrainingLoop` (via
  `pipekit.dumps`)
- `dataset_hash` — `dataset.content_hash()`
- `trained_model_hash` — content hash of the model operator after
  training (via `ModelRegistry.store`)
- `tracker_run_id` — set by the `LogToExperiment` callback if attached
- `model_registry_uri` — set by the `Checkpoint` callback if it stores
  to a `ModelRegistry`
- `backend_info` — backend name + version, hardware, duration
- `deps_lock` — contents of `uv.lock` (best-effort capture)

The callback layer is where these fields get filled in.
`LogToExperiment` wraps a `pipekit_experiment.ExperimentTracker`;
`Checkpoint` wraps a `pipekit_experiment.ModelRegistry`. Neither
package imports the other at top level — the `[experiment]` extra
makes the integration explicit.

## 5. Backend adapter contract

Each backend adapter is a single module under
`pipekit_train.adapters.<backend>` exporting **one** function:

```python
def run(loop: TrainingLoop) -> tuple[Operator, dict[str, Any]]:
    """Run the training loop with this backend.

    Args:
        loop: The TrainingLoop operator.

    Returns:
        A tuple of (trained_model_op, backend_info_dict). The
        trained_model_op is a pipekit.Operator wrapping the trained
        weights. The backend_info_dict is merged into the
        TrainingArtifact under the `backend_info` field.
    """
```

The adapter is responsible for:

1. Translating `loop.dataset` into the backend's loader idiom
   (Equinox: Grain DataLoader; Lightning: `LightningDataModule`;
   Keras: `tf.data.Dataset` or generator).
2. Translating `loop.loss` and `loop.task` into the backend's
   loss / step idiom (Equinox: `TrainTask` Protocol with `loss_fn` +
   optional `eval_fn`; Lightning: `LightningModule.training_step`;
   Keras: `Model.compile(loss=...)`).
3. Threading `loop.callbacks` into the backend's callback hooks.
4. Running the training loop.
5. Wrapping the trained weights in a `pipekit.Operator` (typically
   `pipekit-array.ModelOp` or `pipekit-jax.JaxModelOp`).

The adapter does **not** reimplement gradients, optimiser updates, or
checkpoint serialisation — that machinery is the backend tool's job.
The adapter is glue, not logic.

## 6. The Equinox adapter (v0.1 reference)

The full eqx_trainer design lives inside this one module. Specifically:

- `TrainState(eqx.Module)` — carries `model`, `opt_state`, `step`.
  Built by `TrainState.create(model, optimizer)`.
- `TrainTask(Protocol)` — the user-supplied loss/eval interface.
  Single method `loss_fn(model, batch, key) -> (scalar_loss,
  aux_metrics_dict)`; optional `eval_fn(model, batch, key) ->
  metrics_dict`.
- `train_step(state, batch, key, task, optimizer)` — the
  compiled inner loop, `eqx.filter_jit` + `eqx.filter_grad`. Pure
  function; identical to the eqx_trainer primitive.
- `save_state` / `restore_state` — Orbax `CheckpointManager` +
  `eqx.partition` bridge, identical to the eqx_trainer primitive.
- Grain `DataLoader` + `PyGrainCheckpointHandler` for the data
  iterator; checkpoint pulls both `TrainState` and data-iter
  position so preemption recovers exactly.

The Equinox adapter takes a generic `pipekit_train.Loss` (Layer 1)
and adapts it into a `TrainTask` automatically when the user doesn't
supply one. Users who want full control supply their own `TrainTask`
and the `Loss` argument is ignored. See
[api/adapters.md](api/adapters.md) for the full API.

## 7. Data flow

```
                  ┌────────────────────────────────────┐
                  │   TrainingDataset (Layer 0)        │
                  │                                    │
   catalog  ──►  CatalogDataset    SimulationDataset  ◄──  ForwardModel
                  │                                    │     (pipekit-cycle)
                  └──────────────────┬─────────────────┘
                                     │ (yields (x, y))
                                     ▼
                  ┌────────────────────────────────────┐
                  │   TrainingLoop (Layer 3)           │
                  │   StatefulOperator                 │
                  │                                    │
                  │   carry = (model, opt_state,       │
                  │            step, epoch, metrics,   │
                  │            rng)                    │
                  │                                    │
                  │   ┌── adapter.run(loop) ────────┐  │
                  │   │  backend training loop      │  │
                  │   │  (Equinox / Lightning / …)  │  │
                  │   └─────────────┬───────────────┘  │
                  └─────────────────┼──────────────────┘
                                    │
                                    ▼
                         (trained_model_op, TrainingArtifact)
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                                            ▼
   pipekit-cycle.NeuralForward             pipekit-experiment
   (drops into Cycle / DACycle)            (ModelRegistry.store,
                                            ExperimentTracker.end_run)
```

The arrows are deliberately one-directional: data flows from sources
into the loop and trained artifacts flow out. There is no feedback
arrow from inference back into training in v0.1 (online learning is
out of scope, per [boundaries.md](boundaries.md)).

## 8. Dependency matrix

| Layer    | Module          | Hard deps     | Optional deps                      |
|----------|-----------------|---------------|------------------------------------|
| 0        | dataset.py      | pipekit       | geocatalog, geopatcher (catalog);  |
|          |                 |               | pipekit-cycle (simulation)         |
| 1        | loss.py         | pipekit       | —                                  |
| 2        | callbacks.py    | pipekit       | pipekit-experiment                 |
| 2        | writer.py       | (stdlib)      | —                                  |
| 3        | loop.py         | pipekit       | pipekit-experiment                 |
| 4 (eqx)  | adapters/eqx    | pipekit       | equinox, optax, jax, grain,        |
|          |                 |               | orbax-checkpoint, jaxtyping        |
| 4 (lit)  | adapters/lit    | pipekit       | lightning, torch                   |
| 4 (kr)   | adapters/keras  | pipekit       | keras                              |

Hard deps stay minimal. Backend extras pin only what that backend
needs. The `[experiment]`, `[catalog]`, `[cycle]` extras give the
integration points without forcing them.

## 9. Test strategy

Three rings of tests, mirroring the pipekit-cycle / pipekit-experiment
convention:

1. **Unit** — `tests/test_dataset.py`, `test_loss.py`,
   `test_loop.py`, `test_callbacks.py`. Each tests one module in
   isolation; no backend required for Layers 0–3.
2. **Adapter** — `tests/adapters/test_equinox.py`. Only runs if the
   `[equinox]` extra is installed (gated via `pytest.importorskip`).
   Trains a toy MLP on a 100-sample synthetic dataset end-to-end;
   verifies checkpoint round-trip and artifact contents.
3. **Integration** — `tests/test_cycle_handoff.py`,
   `test_experiment_handoff.py`. Cross-package: trains a toy
   emulator, wraps it as a `NeuralForward`, runs one `Cycle` step;
   trains a toy model and stores it via `LocalModelRegistry`,
   reloads, verifies hash.

CI runs ring 1 by default, ring 2 in the `[equinox]` matrix job, and
ring 3 in the `[experiment, cycle]` matrix job. Lightning and Keras
adapter tests don't exist in v0.1 because the adapters don't.
