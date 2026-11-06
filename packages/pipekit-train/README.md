<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-train/logo-dark.svg">
    <img alt="pipekit-train" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-train/logo-light.svg" width="373">
  </picture>
</p>

# pipekit-train

> **Status — v0.0 / planning only.** This package is scaffolded for
> future work; the source directory currently contains no operators.
> The design lives under [`docs/design/`](docs/design/).

Planned: training pipelines for emulators and amortized inference. A
thin orchestration layer over existing training tools (Equinox+Optax,
Keras 3), not a fourth training-loop
implementation. Trained models are first-class `pipekit_ops.Operator`s
that drop into inference pipelines.

Three training shapes, one framework:

- **Direct supervised** — `CatalogDataset` over labeled scenes (cloud
  masks, super-resolution, retrieval networks).
- **Emulator training** — `SimulationDataset` over a
  `pipekit-cycle.ForwardModel` (neural surrogates for radiative
  transfer, chemistry, plume dispersion).
- **Amortized inference** — `SimulationDataset` again, but the
  network learns the inverse mapping (neural posterior estimation,
  neural likelihood, neural ratio).

See:

- [Master plan — Report 11 (`pipekit-train`)](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_9_pipekit_train.md).
- [Local design doc](docs/design/) — vision, architecture, ADRs, API
  sketch, worked examples.
