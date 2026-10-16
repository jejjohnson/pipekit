# pipekit-evaluate

Multidimensional evaluation framework. The `benchmark` subpackage ships
the orchestration layer of the benchmark ladder — the
`level x rung x complexity` cube, the carrier-agnostic protocols, and
the "simpler thing is ground truth" reference rule. The `taxonomy`
module carries the `Unit x Lens x Stage` tags, `MetricScorer` turns
any metric operator into a tagged scorer, and `evaluate` collects a run's
tagged scores into a content-hashed `EvaluationReport`. Design doc:
`packages/pipekit-evaluate/docs/design/benchmark_ladder.md`.

## Taxonomy

::: pipekit_evaluate.taxonomy

## Scorers

::: pipekit_evaluate.scorers

## Evaluation report

::: pipekit_evaluate.report

## Protocols

::: pipekit_evaluate.benchmark.protocols

## The cube

::: pipekit_evaluate.benchmark.cube

## Reference rule

::: pipekit_evaluate.benchmark.reference

## Benchmark runner

::: pipekit_evaluate.benchmark.run
