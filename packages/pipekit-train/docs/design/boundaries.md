---
status: draft
version: 0.1.0
---

# Boundaries

## 11. Open Questions & Future Work

### Q1: Multi-device / sharding (Equinox adapter)

The v0.1 Equinox adapter assumes single-host, single-device for
simplicity. Supporting `jax.sharding.Mesh` and multi-host Grain
sharding (`grain.ShardByJaxProcess`) requires threading shardings
through `TrainState` and adjusting the Orbax restore path. This is
the most likely first extension. The Lightning adapter inherits its
distributed support from Lightning itself (`Trainer(devices=N)`).

### Q2: Gradient accumulation

Should the trainer support micro-batching natively, or should this be
delegated to the backend? For Equinox, `optax.MultiSteps` is the
clean answer — it composes with the optimizer transformation without
any trainer-level changes. For Lightning, `accumulate_grad_batches`
is a `Trainer` argument that the adapter forwards. Decision:
**delegate to the backend**; pipekit-train does not invent its own
accumulation API.

### Q3: Mixed precision

Equinox doesn't have a built-in mixed-precision API. The cleanest
approach is a `jmp`-based callback or an `eqx.filter_jit` wrapper
that casts arrays. Lightning has native `precision=` support; Keras
has `keras.mixed_precision.set_global_policy`. **Per-backend, no
carrier-agnostic API.** Documented in each adapter's docs.

### Q4: Stateful layers (`eqx.nn.State`)

BatchNorm and similar layers that carry running statistics need
`eqx.nn.State`. The Equinox adapter's `TrainState` should probably
carry an optional `eqx.nn.State` alongside the model. Straightforward
extension; deferred to the first concrete user need. Tracked as a
v0.2 issue.

### Q5: Hyperparameter sweeps (`sweep.py`)

Master plan section 3.1 lists `sweep.py` (HyperSweep, ParameterGrid).
v0.1 ships only the inner training loop; sweeps are a v0.2 feature
that builds on top. The integration target is Optuna or Ray Tune
through the `LogToExperiment` callback (each sweep trial logs to the
same `ExperimentTracker`; the sweep orchestrator reads metrics back
via the tracker's API).

### Q6: Streaming / online updates

Pipekit-train v0.1 assumes batch training. Online updates (a single
gradient step per inference call) are conceptually a different
`StatefulOperator` and could ship as `OnlineTrainer` in v0.3. Out of
scope for now.

### Q7: Async logging

Currently `MetricWriter.write` is synchronous; high-throughput
training may benefit from `jax.experimental.io_callback` style async
logging to prevent host-device sync stalls. Deferred to v0.2.

### Q8: Foundation-model fine-tuning at scale

LoRA / PEFT / QLoRA plug in naturally on the Lightning side (e.g.
through `peft`'s `LightningModule` wrappers) but pipekit-train adds
no machinery for them. The training loop sees a `model_op` with the
right shape; the user is responsible for constructing it with the
right adapter modules attached. Not a gap in pipekit-train; a
documentation item.

## 12. Non-Goals

These were explicitly considered and rejected for the v0.1 scope.

- **Distributed orchestration.** Pipekit-train does not launch
  processes or manage SLURM / K8s jobs. Use your existing infra.
- **Hyperparameter search.** Use Optuna, Ray Tune, etc. The trainer
  is a single-run loop. See Q5.
- **Dataset preparation.** Grain / Lightning DataModule / Keras
  `tf.data` handle this. We accept their loaders directly.
- **Model definition DSL.** Your models are `Operator`s wrapping
  Equinox / PyTorch / Keras modules. We don't wrap them again.
- **Reinforcement learning.** No environment-step / reward loop.
- **Online / streaming training.** Batch only in v0.1. See Q6.
- **Foundation-model orchestration.** Beyond what the backend already
  provides. See Q8.

## 13. v0.1 Definition of Done

All boxes ticked as of the v0.1 polish PR.

- [x] Layers 0–3 implemented and tested (Dataset, Loss, Components,
  TrainingLoop).
- [x] Equinox adapter (Layer 4) implemented and tested end-to-end
  on a toy MLP regression task.
- [x] Lightning and Keras adapter modules exist as scaffolds and
  raise `NotImplementedError` from `run()` with a clear "this
  adapter ships in v0.2 / v0.3" message.
- [x] `TrainingArtifact` round-trip works against
  `LocalModelRegistry` — `tests/test_experiment_handoff.py`. The
  v0.1 contract is "the hash + URI land in the artifact and the
  operator config persists in the registry"; round-tripping the
  trained `eqx.Module` weights through `from_state` alone is a
  `pipekit-jax.JaxModelOp` v0.2 feature. `S3ModelRegistry` works
  identically (fsspec-backed) under `[experiment, s3]` — not
  exercised in CI to avoid the cloud credential, but the
  `pipekit-experiment` test suite covers it.
- [x] `SimulationDataset → Cycle → NeuralForward` round-trip works
  on a toy oscillator forward model — `tests/test_cycle_handoff.py`.
- [x] Smoke tests for the three worked examples in `examples/`
  (direct / emulator / amortized) run end-to-end on synthetic data
  under `[equinox, cycle]` — `tests/test_examples_smoke.py`.
- [x] Documentation published to the mkdocs site as
  `docs/api/pipekit-train.md` next to the existing package docs,
  plus two example notebooks under `docs/notebooks/`
  (`pipekit_train_quickstart.ipynb`,
  `pipekit_train_cycle_handoff.ipynb`).

## 13.1 Small post-design divergences

A handful of implementation details landed slightly differently from
the design as written; logged here for honesty.

- **`Callback` Protocol is not `@runtime_checkable`** — needed so a
  partial callback (e.g. one that only implements `on_step_end`)
  still satisfies the contract. The Protocol exists for static
  typing of the full hook surface; adapters dispatch via
  `getattr(cb, hook, None)`. See ADR D9.
- **`CachedDataset` is local-filesystem only in v0.1.** Fsspec URIs
  (`s3://`, `gcs://`) remain documented as the target, but the
  v0.1 implementation rejects them — fsspec-backed zarr stores
  land in v0.2 alongside the rest of the cache work.
- **Indexable iterator constraint** — `Equinox.run()` fails fast
  with a clear `ValueError` when `len(dataset) < batch_size` for
  the indexable path. With a fixed batch_size the JIT'd
  `train_step` doesn't recompile per epoch boundary, but a dataset
  smaller than one batch would yield nothing and hang the loop.
  Streaming `IterableDataset` doesn't have this constraint.
- **`optimizer_config` accepts both `lr` and `learning_rate`** —
  the design used `lr` (the common shorthand); optax expects
  `learning_rate`. The Equinox adapter's `_build_optimizer`
  normalises `lr` → `learning_rate` at the boundary.
- **`_EquinoxModelOp` ships in-package** as the v0.1 trained-model
  wrapper. It's the drop-in replacement for the future
  `pipekit-jax.JaxModelOp` — same constructor signature, same
  `_apply` shape. Documented in `api/adapters.md`.

## 14. Versioning targets

- **v0.1** — Layers 0–3 + Equinox adapter (this design).
- **v0.2** — Lightning adapter; `sweep.py`; multi-device for Equinox
  (Q1); gradient accumulation polish (Q2).
- **v0.3** — Keras 3 adapter; async logging (Q7); stateful layers
  (Q4); online trainer (Q6).

The cadence matches the rest of the pipekit workspace: roughly one
minor bump per quarter, gated on the v0.1 scope landing first.
