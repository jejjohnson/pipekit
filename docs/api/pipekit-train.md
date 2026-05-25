# pipekit-train

Training pipelines for emulators and amortized inference. A thin
orchestration layer over existing training tools (Equinox+Optax
ships as the v0.1 reference; Lightning and Keras adapter modules
are scaffolded for v0.2 / v0.3). Trained models are first-class
`pipekit.Operator`s — they drop into inference pipelines (e.g.
`pipekit_cycle.NeuralForward` for emulator handoff). Master plan
reference: Report 11.

The full multi-file design lives in the package:
[`packages/pipekit-train/docs/design/`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train/docs/design).

## Datasets

::: pipekit_train.dataset

## Losses

::: pipekit_train.loss

## Callbacks

::: pipekit_train.callbacks

## Metric writers

::: pipekit_train.writer

## Training loop

::: pipekit_train.loop

## Hyperparameter sweeps

::: pipekit_train.sweep

## Backend adapters

The Equinox+Optax+Orbax adapter is the v0.1 reference. Lightning and
Keras adapter modules ship as scaffolds that raise `NotImplementedError`
on `run()` until v0.2 / v0.3 land — see
[`adapters.md`](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/api/adapters.md)
for the planned designs.

### Equinox (v0.1 reference)

::: pipekit_train.adapters.equinox
