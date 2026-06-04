---
status: draft
version: 0.1.0
---

# pipekit-train Design Doc

**A thin orchestration layer over existing training tools, exposed as
`pipekit.Operator`s. Trained models drop into inference pipelines
without rewrite.**

## Structure

```
docs/design/
├── README.md              # This file — index and reading order.
├── vision.md              # Motivation, scope, three training shapes.
├── architecture.md        # The layered design + integration with
│                          # pipekit, pipekit-cycle, pipekit-experiment.
├── boundaries.md          # Non-goals, open questions, future work.
├── decisions.md           # ADRs D1–D14 (carrier-agnostic + Equinox +
│                          # sharding + NumPyro + BlackJAX).
├── numpyro_adapter.md     # Full design — the NumPyro inference backend
│                          # (SVI + NUTS). Companion to api/adapters.md.
├── blackjax_adapter.md    # Full design — the BlackJAX sampler backend
│                          # (NUTS / SG-MCMC). Sibling to numpyro_adapter.md.
├── api/
│   ├── README.md          # Core abstractions (TrainingDataset,
│   │                      # Loss, TrainingLoop, Callback, MetricWriter).
│   ├── datasets.md        # Layer 0 — TrainingDataset, CatalogDataset,
│   │                      # SimulationDataset, CachedDataset.
│   ├── primitives.md      # Layer 1 — Loss protocol, common Loss ops,
│   │                      # train_step (Equinox), save/restore (Orbax).
│   ├── components.md      # Layer 2 — Callback, MetricWriter,
│   │                      # ValidationStep, EarlyStopping, Checkpoint,
│   │                      # LogToExperiment.
│   ├── models.md          # Layer 3 — TrainingLoop (StatefulOperator),
│   │                      # the outer loop, the TrainingArtifact handshake.
│   └── adapters.md        # Layer 4 — Equinox (v0.1 reference),
│                          # Lightning (v0.2 scaffold), Keras (v0.3 scaffold).
└── examples/
    ├── README.md          # Index and reading order.
    ├── direct.md          # Cloud-mask classifier (CatalogDataset).
    ├── emulator.md        # Chemistry surrogate (SimulationDataset).
    ├── amortized.md       # Neural posterior for plume source (SBI).
    └── integration.md     # With pipekit-cycle, pipekit-experiment,
                           # geocatalog, geopatcher, somax, vardax, pyrox.
```

## Reading Order

1. **[vision.md](vision.md)** — what gap this fills, what it isn't.
2. **[architecture.md](architecture.md)** — the layered design and
   where each piece sits in the stack.
3. **[decisions.md](decisions.md)** — the ADRs that lock the shape in.
4. **[api/README.md](api/README.md)** → drill into
   **[datasets](api/datasets.md)** → **[primitives](api/primitives.md)**
   → **[components](api/components.md)** → **[models](api/models.md)**
   → **[adapters](api/adapters.md)**.
5. **[examples/](examples/)** — see the three training shapes in code.
6. **[boundaries.md](boundaries.md)** — what's deferred and why.

## Sources

The design synthesises two upstream sources:

- The master plan: [`research_journal_v2/notes/geotoolz/master_plan/toolz_9_pipekit_train.md`](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_9_pipekit_train.md)
  (Report 11). Vision, three training shapes, package scope,
  reproducibility story.
- The `eqx_trainer` design doc in
  `jej_vc_snippets/design_docs/eqx_trainer/`. Concrete JAX-native
  stack (Equinox + Optax + Grain + Orbax) that becomes the v0.1
  reference adapter inside `pipekit_train.adapters.equinox`.

Where the two diverge, this doc resolves the tension explicitly —
see [decisions.md](decisions.md) for the ADRs.
