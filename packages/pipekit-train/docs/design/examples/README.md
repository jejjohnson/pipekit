---
status: draft
version: 0.1.0
---

# Examples

Four sketches that cover the three training shapes from the master
plan plus ecosystem composition. Each is a code-first walkthrough of
how the API in `api/` flows together; they are intentionally short
(under one screen each) and use the v0.1 Equinox adapter.

## Index

| File                          | Shape                          | Master plan ref |
|-------------------------------|--------------------------------|-----------------|
| [direct.md](direct.md)        | Direct supervised — cloud mask | §2.1            |
| [emulator.md](emulator.md)    | Emulator — chemistry surrogate | §2.2            |
| [amortized.md](amortized.md)  | Amortized inference — plume SBI| §2.3            |
| [integration.md](integration.md) | Cross-package composition   | §4              |

## Reading order

If you're new to the framework: **direct → emulator → amortized →
integration**. The complexity grows linearly:

- *direct.md* introduces `CatalogDataset`, `Loss`, `TrainingLoop`,
  `Checkpoint`, `LogToExperiment`.
- *emulator.md* adds `SimulationDataset` and the
  `pipekit-cycle.NeuralForward` handoff.
- *amortized.md* adds `ConditionalNormalizingFlow` from `pyrox`,
  `obs_op`, and the SBI loss path.
- *integration.md* composes pipekit-train with pipekit-cycle,
  pipekit-experiment, geocatalog, geopatcher, somax, vardax,
  pyrox-nn — the full vertical slice.
