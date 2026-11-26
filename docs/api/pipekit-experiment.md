# pipekit-experiment

The boundary between pipekit pipelines and external orchestration
tools — runtime-checkable protocols, a content-addressed model
registry, JSON-roundtrippable artifacts, and thin per-tool bridges.
Master plan reference: Report 12.

## Protocols

::: pipekit_experiment.protocols

## Run records

::: pipekit_experiment.run

## Model registry

::: pipekit_experiment.registry

## Artifacts

::: pipekit_experiment.artifacts

## Bridges

`pipekit_experiment.bridges` has one module per tool, as in pipekit-array ([X1](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-experiment/docs/design/decisions.md)).
Each bridge is gated behind its tool's optional extra and raises a
clean `ImportError` on first use if the underlying tool isn't
installed. The pure-pipekit surface remains importable either way.

### DVC

Requires `pipekit-experiment[dvc]`.

::: pipekit_experiment.bridges.dvc

### Hydra + hydra-zen

Requires `pipekit-experiment[hydra]`.

::: pipekit_experiment.bridges.hydra

### Metaflow

Requires `pipekit-experiment[metaflow]`.

::: pipekit_experiment.bridges.metaflow
