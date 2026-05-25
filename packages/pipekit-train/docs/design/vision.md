---
status: draft
version: 0.1.0
---

# Vision

## 1. Motivation

Three training tools already do the heavy work. PyTorch Lightning,
Equinox+Optax, and Keras 3 each have battle-tested training loops,
distributed primitives, mixed precision, and profiling. Reimplementing
that machinery is wrong.

What's *missing* in geophysical ML is the **glue**: a way to compose
training pipelines from the same operator-graph machinery that
`pipekit` already provides for inference, with first-class support
for two non-standard data shapes:

- **Simulator-generated training pairs.** Emulators train on the
  output of a `pipekit-cycle.ForwardModel`. The training loop closes
  with the inference loop — the trained emulator drops into the same
  `Cycle` as a `NeuralForward`.
- **Amortized inverse problems.** Simulation-based inference / neural
  posterior estimation requires `(parameters, simulated_obs)` pairs
  and a conditional density estimator. Same simulator-as-dataset
  abstraction; different loss and head.

There is also a smaller but persistent need: **reproducibility**. A
trained model should be recoverable from `(training_pipeline_yaml,
dataset_content_hash, seed)` — the same content-hash discipline that
`pipekit-experiment` already provides for inference pipelines.
Pipekit-train must extend that discipline to the training side.

## 2. What pipekit-train is

A thin orchestration layer that wraps existing training tools and
exposes them as composable `pipekit.Operator`s. Five conceptual
pieces, all on top of `pipekit` core:

- **Datasets** as `Operator`s. `CatalogDataset`,
  `SimulationDataset`, `CachedDataset`. Each yields `(input, target)`
  pairs and provides a content hash.
- **Losses** as `Operator`s, satisfying a `Loss` Protocol. `MSE`,
  `NLL`, `KL`, `Composite`. Composable with the rest of the pipeline.
- **The training loop** as a `StatefulOperator` (carries model,
  optimizer state, step, epoch, metrics). The headline composable.
- **Callbacks and MetricWriters** as runtime-checkable Protocols.
  Includes a `LogToExperiment` callback that bridges to any
  `pipekit-experiment.ExperimentTracker`.
- **Backend adapters** under `pipekit_train.adapters.*`, one per
  training tool, extras-gated. The Equinox adapter is the v0.1
  reference; Lightning and Keras are scaffolded.

The trained model is itself a `pipekit.Operator`. After training, the
output is a `pipekit-array.ModelOp` (PyTorch / Keras) or a
`pipekit-jax.JaxModelOp` (Equinox) wrapping the trained weights, and
its content hash flows into the `pipekit-experiment` model registry.

## 3. Design principles

These are the principles that govern the rest of the design. They
echo and extend the principles from `eqx_trainer`'s vision: the same
"composable, not a framework" stance, scaled up to a carrier-agnostic
operator-graph context.

- **Thin orchestration, not a fourth training-loop implementation.**
  Wrap Lightning / Equinox+Optax / Keras 3; don't compete with them.
- **Datasets and losses are pipekit Operators.** Same YAML
  round-trip discipline, same content-hashing, same composition
  surface as everything else in pipekit.
- **The training loop is a StatefulOperator.** Optimizer state, step
  count, epoch count, metrics are the carry-state. Built on
  `pipekit.state.StatefulOperator` from Group M.
- **Adapter pattern, extras-gated.** Each backend is one module
  under `pipekit_train.adapters.*`. Backend extras pin only the
  libraries that backend needs. No backend is required to import the
  core surface.
- **Reproducibility via pipekit-experiment.**
  `pipekit-experiment.TrainingArtifact` already exists — it expects
  the YAML config, dataset hash, trained-model hash, tracker run id,
  and registry URI. `TrainingLoop.run()` returns exactly this
  artifact. No new reproducibility primitive is invented here.
- **Trained models are inference operators.** The output of
  `loop.run()` is a `pipekit.Operator` that composes into any
  `Sequential` / `Graph` / `Cycle`. Train→serve is one operator
  swap, not a rewrite.

## 4. Why this is *not* the same as eqx_trainer

`eqx_trainer` (the design doc in `jej_vc_snippets/`) is an excellent
Equinox-native trainer. It is also exactly what should live inside
`pipekit_train.adapters.equinox` — Grain DataLoader, Orbax
CheckpointManager, `eqx.partition` bridge, `TrainTask` Protocol, JSON
lines `MetricWriter`. The eqx_trainer ADRs (D1–D6) become the
adapter-level decisions (see [decisions.md](decisions.md), Section
"Equinox adapter decisions").

What `pipekit-train` adds on top is the **carrier-agnostic surface**:
dataset abstractions that work for catalogs and simulators alike,
loss and callback Protocols that don't depend on JAX, an artifact
contract that integrates with `pipekit-experiment`, and a
`TrainingLoop` operator that serializes via the same machinery as
every other pipekit operator. The Equinox adapter then *uses* the
eqx_trainer design internally to do the actual training.

The two designs do not compete; they layer.

## 5. Out of scope

- Distributed orchestration (SLURM, K8s, multi-node) — that's the
  orchestrator's job. Single-node multi-GPU is delegated to the
  backend (Lightning does it natively; Equinox via `jax.sharding`).
- Hyperparameter search — see `pipekit_train.sweep` as deferred work
  (Section 11.5 in [boundaries.md](boundaries.md)). Optuna / Ray Tune
  plug in through `LogToExperiment`.
- Online / streaming training — pipekit-train assumes batch training.
- Reinforcement learning — different abstraction entirely.
- Dataset preparation — Grain / Lightning DataModule / Keras
  `tf.data` handle that. We accept their loaders directly.
