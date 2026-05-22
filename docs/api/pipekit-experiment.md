# pipekit-experiment

The boundary between pipekit pipelines and external orchestration
tools — runtime-checkable protocols, a content-addressed model
registry, JSON-roundtrippable artifacts, and thin per-tool adapters.
Master plan reference: Report 12.

## Protocols

::: pipekit_experiment.protocols

## Run records

::: pipekit_experiment.run

## Model registry

::: pipekit_experiment.registry

## Artifacts

::: pipekit_experiment.artifacts

## Adapters

Each adapter is gated behind its tool's optional extra and raises a
clean `ImportError` on first use if the underlying tool isn't
installed. The pure-pipekit surface remains importable either way.

### DVC

Requires `pipekit-experiment[dvc]`.

::: pipekit_experiment.adapters.dvc

### Hydra + hydra-zen

Requires `pipekit-experiment[hydra]`.

::: pipekit_experiment.adapters.hydra

### Metaflow

Requires `pipekit-experiment[metaflow]`.

::: pipekit_experiment.adapters.metaflow
