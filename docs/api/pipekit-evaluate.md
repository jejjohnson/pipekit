# pipekit-evaluate

Multidimensional evaluation framework.
Design doc: `packages/pipekit-evaluate/docs/design/benchmark_ladder.md`.

The package is three areas, each with one job, and an area imports only from the ones before it ([E1](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/decisions.md)).

## Scoring

`pipekit_evaluate.scoring` carries the `Unit x Lens x Stage` tags, `MetricScorer`, which turns any metric operator into a tagged scorer, the `Estimate` and `Scorer` protocols, and `ReferenceSource`, which says where a score's ground truth comes from.

::: pipekit_evaluate.scoring

## Reports

`pipekit_evaluate.reports` collects a run's tagged scores into a content-hashed `EvaluationReport`.

::: pipekit_evaluate.reports

## Benchmark

`pipekit_evaluate.benchmark` is the orchestration layer of the benchmark ladder: the `level x rung x complexity` cube, the `Task`, `Estimator` and `Oracle` protocols, the "simpler thing is ground truth" reference rule, and `Benchmark`, which runs it.

::: pipekit_evaluate.benchmark
