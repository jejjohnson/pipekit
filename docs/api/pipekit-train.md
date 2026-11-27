# pipekit-train

Training pipelines for emulators and amortized inference. A thin
orchestration layer over existing training tools (Equinox+Optax
ships as the v0.1 reference; a Keras adapter module is
scaffolded for v0.3). Trained models are first-class
`pipekit_ops.Operator`s — they drop into inference pipelines (e.g.
`pipekit_cycle.NeuralForward` for emulator handoff). Master plan
reference: Report 11.

The full multi-file design lives in the package:
[`packages/pipekit-train/docs/design/`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train/docs/design).

The package is five areas, each with one job, and an area imports only from the ones before it ([D17](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/decisions.md)).
Each backend is an area of its own under `pipekit_train.backends`, behind its extra.

## Datasets

::: pipekit_train.data

## Losses

::: pipekit_train.losses

## Training loop

The loop, its callbacks and its metric writers.

::: pipekit_train.loop

## Hyperparameter sweeps

::: pipekit_train.sweep

## Backends

The Equinox, Optax and Orbax backend is the v0.1 reference.
The Keras backend trains with Keras 3's `compile` and `fit` on JAX; see [`keras_adapter.md`](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/keras_adapter.md) for its design.

### Equinox

::: pipekit_train.backends.equinox
