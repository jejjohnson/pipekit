---
status: draft
version: 0.1.0
---

# Design Decisions

Eleven ADRs total. Decisions D1–D7 are about the carrier-agnostic
surface. Decisions D8–D11 are about the Equinox adapter
specifically (they port and extend the eqx_trainer D1–D6).

When the design lands, each accepted ADR should also be reflected in
a one-paragraph entry in the package CHANGELOG and (where it
materially constrains future work) in `AGENTS.md`.

---

## D1: Thin orchestration over existing training tools

**Status:** accepted

**Context:** PyTorch Lightning, Equinox+Optax, and Keras 3 each ship
complete, battle-tested training loops with distributed support,
mixed precision, profiling, and checkpointing. A fourth
implementation cannot improve on any of them.

**Decision:** pipekit-train wraps these tools through an
adapter pattern. It does not reimplement gradient descent,
optimiser updates, distributed primitives, or checkpoint
serialisation. Each adapter is ~150 LOC of glue.

**Consequences:** Users debugging a training failure may need to read
both pipekit-train and the underlying backend. Mitigation: error
messages name the responsible layer; per-backend debugging guides
in the docs.

---

## D2: TrainingDataset is a `pipekit.Operator`

**Status:** accepted

**Context:** Could be a generator function, an iterator class, or an
operator. Generator and iterator forms lose YAML round-trip, content
hashing, and composability with the rest of pipekit. Master plan
proposes the Operator form.

**Decision:** `TrainingDataset` extends `pipekit.Operator`. It
defines `__iter__` returning `(x, y)` pairs and a `content_hash()`
returning a stable identifier. Subclasses (`CatalogDataset`,
`SimulationDataset`, `CachedDataset`) inherit operator semantics:
they serialise via `get_config` / `dumps`, compose with
`Sequential`, and integrate with the registry.

**Consequences:** Iteration semantics need to be defined precisely
for the operator surface — see `api/datasets.md` for the
"`__iter__` is the side-effect entrypoint; `_apply` is undefined"
rule. Worth it for the composability gain.

---

## D3: `TrainingLoop` is a `StatefulOperator`

**Status:** accepted

**Context:** Master plan and `eqx_trainer` both treat the training
loop as the headline composable. The pipekit-core `StatefulOperator`
abstraction (Group M) already exists for cycles and iterative
processes. The training loop's natural shape is exactly
"(carrier_in, state_in) → (carrier_out, state_out)" with state
being `(model, optimizer_state, step, epoch, metrics, rng)`.

**Decision:** `TrainingLoop` extends `pipekit.StatefulOperator`. Its
carry-state is a `TrainerCarryState` (subclass of
`pipekit.CarryState`) carrying the six fields above.
`TrainingLoop._apply(carrier, state)` delegates to the configured
adapter and returns `(trained_model_op, new_state)`.

**Consequences:** Checkpointing reduces to "serialise the
carry-state". Resume-from-checkpoint is a single
`from_dict`. Multi-stage training (pretrain → fine-tune) is a
`Sequential` of two `TrainingLoop`s threading carry-state through.

---

## D4: Loss is a Protocol; common Losses are Operators

**Status:** accepted

**Context:** `eqx_trainer` (D4) makes `TrainTask` a Protocol and
deliberately avoids shipping common losses. The master plan ships
`MSE`, `NLL`, `KL`, `Composite` as Operators. Both can coexist.

**Decision:** Ship two surfaces. (1) A runtime-checkable `Loss`
Protocol (signature: `(predicted, target) -> float`) for users who
want to plug in their own loss. (2) Common `Loss` Operators (`MSE`,
`NLL`, `KL`, `Composite`) that satisfy the Protocol and additionally
provide YAML round-trip + composition. Per-backend, the adapter
either calls the `Loss` directly or wraps it in the backend's
expected idiom (Equinox: bundles it into a synthesised `TrainTask`;
Lightning: uses it inside the generated `LightningModule.training_step`).

**Consequences:** Two ways to write a loss is an honest cost. The
Protocol is the escape hatch; the common Losses are the default.

---

## D5: Adapter pattern, extras-gated

**Status:** accepted

**Context:** Each backend has a different idiom (Lightning's
LightningModule, Equinox's filter\_jit, Keras's compile/fit). Heavy
backend dependencies (torch, jax, tensorflow) cannot all be hard
requirements.

**Decision:** Each backend lives in one module under
`pipekit_train.adapters.<backend>`. Each is gated behind a
`[<backend>]` extra in `pyproject.toml`. Importing the adapter
module without its extra raises a clean `ImportError` on first use,
not at import time — so a CLI can branch on what's installed.
Equinox is the v0.1 reference. Lightning and Keras adapters are
scaffolds that raise `NotImplementedError` in v0.1; targeted v0.2 /
v0.3.

**Consequences:** Pipekit-train ships with no backend by default.
Users must `pip install pipekit-train[equinox]` (or another
backend). This matches the `pipekit-array` and `pipekit-experiment`
extras model.

---

## D6: Training artifacts integrate with `pipekit-experiment`

**Status:** accepted

**Context:** Reproducibility for training runs requires capturing the
training pipeline, dataset content hash, trained model hash, tracker
run id, and registry URI. `pipekit-experiment` already ships
`TrainingArtifact` for exactly this purpose.

**Decision:** `TrainingLoop.run()` returns
`(trained_model_op, TrainingArtifact)`. The artifact is filled in by
the callback layer: `LogToExperiment` fills `tracker_run_id`;
`Checkpoint` (when configured against a `ModelRegistry`) fills
`model_registry_uri` and `trained_model_hash`; the loop itself fills
`training_pipeline_yaml`, `dataset_hash`, `backend_info`, and
`deps_lock`. The `pipekit-experiment` dependency is gated behind the
`[experiment]` extra.

**Consequences:** No new reproducibility primitive is invented in
this package. Users without `pipekit-experiment` get a "naked"
`(trained_model_op, dict)` return value — the dict is the same
shape as `TrainingArtifact.__dict__` but not typed.

---

## D7: Trained models are pipekit Operators

**Status:** accepted

**Context:** A trained model needs to compose with inference
pipelines without rewrite. `pipekit-array.ModelOp` and (planned)
`pipekit-jax.JaxModelOp` already exist as the operator wrappers.

**Decision:** The Layer 4 adapter's `run()` returns a
`pipekit.Operator`. For Equinox, this is a `JaxModelOp` (or a
domain-specific subclass). For Lightning / Keras, it's a `ModelOp`.
The wrapper round-trips its weights through a backend-defined
weight blob (`bytes`) that `ModelRegistry.store(..., weights=...)`
persists alongside the operator config.

**Consequences:** A trained emulator drops into
`pipekit_cycle.NeuralForward(model_op, dt=...)` without any
conversion. A trained classifier drops into any `Sequential`
preprocessing pipeline. This is the core "train→serve loop is one
operator swap" claim.

---

## D8: Equinox adapter uses Grain + Orbax + Optax

**Status:** accepted (ports eqx_trainer D2, D3)

**Context:** The eqx_trainer design picks Grain for data, Orbax for
checkpointing, and Optax for optimisation. These are the modern
JAX-ecosystem choices: deterministic iterators, async multi-device
checkpoints, structured gradient transformations.

**Decision:** The Equinox adapter uses these libraries directly.
`grain.DataLoader` for the iterator (with
`PyGrainCheckpointHandler` so preemption recovery restores both
weights *and* iterator position); `orbax-checkpoint.CheckpointManager`
with `eqx.partition` bridge for serialisation; `optax` for the
optimiser.

**Consequences:** The `[equinox]` extra pins six libraries
(equinox, optax, jax, grain, orbax-checkpoint, jaxtyping). All are
single-binary-friendly on CPU+GPU. No TensorFlow dependency leaks
through.

---

## D9: TrainTask Protocol per backend; Callback hooks are optional

**Status:** accepted (ports eqx_trainer D4)

**Context:** Two related issues. (1) `eqx_trainer` (D4) makes the
user-supplied training target a `TrainTask` Protocol (`loss_fn` +
optional `eval_fn`). This is the cleanest interface for JAX where
the loss function returns auxiliary metrics from inside the same
`eqx.filter_grad`. Other backends have their own idioms
(Lightning's `LightningModule.training_step`). (2) The `Callback`
surface defines five hooks but most callbacks only care about one
or two; the adapter must tolerate partial implementations.

**Decision:**

1. Each adapter defines its own `TrainTask` Protocol or equivalent.
   For the Equinox adapter, this is the eqx_trainer `TrainTask`
   Protocol verbatim. For the Lightning adapter (v0.2), the
   equivalent is a `LightningModule` factory function. For Keras
   (v0.3), it's a `(loss, metrics)` tuple. The carrier-agnostic
   `Loss` from D4 is the *default* — if the user doesn't supply a
   backend `TrainTask`, the adapter synthesises one from the `Loss`
   Operator.
2. The `Callback` Protocol is **not** `@runtime_checkable`.
   `runtime_checkable` would require every callback to implement
   all five hooks (otherwise `isinstance(cb, Callback)` returns
   False), which contradicts the partial-implementation model.
   The Protocol stays as the static-typing contract for the *full*
   hook surface; runtime dispatch in adapters uses `getattr(cb,
   hook, None)`. `Loss` and `MetricWriter` *are* runtime-checkable
   because every implementation has the same single-method surface.

**Consequences:** Power users see backend idioms; default users
only see `Loss`. The synthesised path is what the worked examples
in `examples/` use. Partial `Callback` implementations (e.g. a
metric logger that only cares about `on_step_end`) are valid by
design, and the test suite encodes this contract structurally
rather than via `isinstance`.

---

## D10: Per-step is the unit of training, not per-epoch

**Status:** accepted

**Context:** Master plan uses `n_epochs`. eqx_trainer uses
`num_steps`. Grain iterators are typically infinite (`num_epochs=None`).
Per-epoch semantics break down when the dataset is a streaming
simulator or when steps_per_epoch isn't well-defined.

**Decision:** The training loop's unit is the step. `TrainingLoop`
takes `max_steps` (required); `steps_per_epoch` is optional and
purely cosmetic (controls when `on_epoch_end` callbacks fire).
Eval and checkpoint cadences are `every_n_steps`. Datasets are
expected to be infinite iterators; finite datasets are wrapped to
repeat.

**Consequences:** Existing-user mental model of "10 epochs" needs
translation to "10 × steps_per_epoch steps". Documented prominently
in the examples.

---

## D11: TrainerConfig is the operator's config dict, not a separate dataclass

**Status:** accepted (refines eqx_trainer D5)

**Context:** `eqx_trainer` (D5) ships a `TrainerConfig` dataclass and
explicitly avoids YAML / ml_collections. The pipekit world already
has a YAML round-trip via `pipekit.serial.dumps`. A separate config
dataclass would duplicate machinery.

**Decision:** `TrainingLoop` is constructed with the same flat-config
shape that the eqx_trainer `TrainerConfig` proposes (`max_steps`,
`seed`, `checkpoint_every_n_steps`, `max_checkpoints_to_keep`,
`eval_every_n_steps`, `log_every_n_steps`, …), but those fields live
directly on the operator and are surfaced through its
`get_config()`. The YAML round-trip is free. Users who prefer the
dataclass-only style can ignore the YAML side and treat
`TrainingLoop` exactly like the eqx_trainer's `Trainer` +
`TrainerConfig` pair.

**Consequences:** Slight tension with eqx_trainer D5's "config files
are the caller's responsibility". Resolved by saying: pipekit-train
*provides* the YAML facility (via the operator surface) but does not
*require* it. Callers using a dataclass-style API see no YAML.

---

## D12: Sharding is an optional `ShardingSpec`, not a new code path

**Status:** accepted (resolves boundaries.md Q1, issue #15)

**Context:** The v0.1 Equinox adapter was single-host, single-device.
Multi-GPU (model parallelism, larger batches) and multi-host (training
across nodes) are the most-requested extension. The risk was forking the
adapter into a parallel "distributed" code path.

**Decision:** Add one optional, frozen `ShardingSpec(mesh, model_pspec,
data_pspec)` and thread it through the *existing* path rather than
branching:

- `TrainState.create(..., sharding=spec)` places model + optimiser array
  leaves on `spec.model_sharding`; the step counter is replicated.
- `train_step` is unchanged — `eqx.filter_jit` compiles sharding-aware
  automatically once its inputs are sharded (replicated model + a batch
  sharded along `data_pspec` gives data-parallel SPMD with XLA inserting
  the collectives). No explicit `pjit` wrapper is needed for the
  data-parallel case.
- The data iterator places each batch on `spec.data_sharding`; for
  multi-host it first reads a disjoint **process-shard** of the `grain`
  `MapDataset` (`map_ds[process_index::process_count]`, the modern
  equivalent of `grain.ShardByJaxProcess`) and assembles the global array
  with `jax.make_array_from_process_local_data`.
- Orbax restore is sharding-aware *for free*: the restore template is the
  already-sharded `TrainState`, so `to_shape_dtype_struct` carries each
  leaf's sharding into `StandardRestore`.
- `TrainingLoop.sharding` is typed `Any | None` so the carrier-agnostic
  core never imports JAX (the spec lives in `adapters.equinox`).

**Consequences:** `sharding=None` is byte-for-byte the old single-device
path. The default (data-parallel) spec is `model_pspec=PartitionSpec()`
(replicated) + `data_pspec=PartitionSpec("data")`; uniform model-parallel
specs work when leaf ranks are compatible. Verified on a CPU-simulated
2-device mesh and a scripted 2-process `jax.distributed` smoke. The
Lightning adapter keeps its own native distributed support (D5).

---

## Resolved Questions

| Question                            | Resolution                                         |
|-------------------------------------|----------------------------------------------------|
| Backend strategy                    | Adapter pattern, extras-gated (D5)                 |
| Carrier-agnostic surface            | Datasets + Losses as Operators (D2, D4)            |
| Loop primitive                      | StatefulOperator (D3)                              |
| Reproducibility                     | pipekit-experiment.TrainingArtifact (D6)           |
| Train→serve composition             | Trained models are Operators (D7)                  |
| JAX stack                           | Grain + Orbax + Optax (D8)                         |
| User loss interface                 | Per-backend TrainTask Protocol (D9)                |
| Training unit                       | Per-step (D10)                                     |
| Config representation               | Operator.get_config() — YAML for free (D11)        |
| Distributed training                | Delegated to backend (Lightning native; JAX        |
|                                     | sharding for Equinox) — see boundaries.md          |
| Hyperparameter search               | Deferred to v0.2 sweep.py — see boundaries.md      |

## Open Questions (see boundaries.md)

- Q1: Multi-device sharding for the Equinox adapter.
- Q2: Gradient accumulation (Optax MultiSteps vs trainer-level).
- Q3: Mixed precision (per-backend, no carrier-agnostic API).
- Q4: Stateful layers (`eqx.nn.State` for BatchNorm-like layers).
- Q5: Streaming / online updates as a v0.3 follow-up.
