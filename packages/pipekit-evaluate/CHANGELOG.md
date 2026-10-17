# Changelog

## [0.1.1](https://github.com/jejjohnson/pipekit/compare/pipekit-evaluate-v0.1.0...pipekit-evaluate-v0.1.1) (2026-10-17)


### Bug Fixes

* **build:** ship a LICENSE in each package so it builds ([#90](https://github.com/jejjohnson/pipekit/issues/90)) ([5671c46](https://github.com/jejjohnson/pipekit/commit/5671c46e933d7770bd930b93591c621b135cc8ff))

## [0.1.0](https://github.com/jejjohnson/pipekit/compare/pipekit-evaluate-v0.0.2...pipekit-evaluate-v0.1.0) (2026-10-17)


### ⚠ BREAKING CHANGES

* **evaluate:** BenchmarkReport.to_dict() returns {"task", "domain", "axis", "cells"}, where each cell maps to its rung_name, primary reference, scores by reference source and note, instead of {cell label: {metric: value}}.

### Features

* **evaluate:** add CovarianceEstimate for second-moment scorers ([#63](https://github.com/jejjohnson/pipekit/issues/63)) ([020f528](https://github.com/jejjohnson/pipekit/commit/020f528788e69d0ab9509d2f6e3940f04421faa9))
* **evaluate:** add EvaluationReport and evaluate() ([#59](https://github.com/jejjohnson/pipekit/issues/59)) ([811a7ac](https://github.com/jejjohnson/pipekit/commit/811a7ac80fd04e50ad102ce4f7879fd82a3ee6dc))
* **evaluate:** convert a benchmark run into an EvaluationReport ([#62](https://github.com/jejjohnson/pipekit/issues/62)) ([6289867](https://github.com/jejjohnson/pipekit/commit/62898671a4f31e0b9288206c46c5ad1727d43704))
* **evaluate:** score a ladder cell against several references ([#61](https://github.com/jejjohnson/pipekit/issues/61)) ([db64d3a](https://github.com/jejjohnson/pipekit/commit/db64d3af1e30b762b699722d7a163456258c707b))
* **evaluate:** tag scorers with unit, lens and stage; add MetricScorer ([#58](https://github.com/jejjohnson/pipekit/issues/58)) ([a0204e7](https://github.com/jejjohnson/pipekit/commit/a0204e74c50d2ce05f8b08c635162da4216e939a))

## [0.0.2](https://github.com/jejjohnson/pipekit/compare/pipekit-evaluate-v0.0.1...pipekit-evaluate-v0.0.2) (2026-07-11)


### Features

* bootstrap pipekit monorepo with core, cycle, and experiment packages ([#6](https://github.com/jejjohnson/pipekit/issues/6)) ([9b992e6](https://github.com/jejjohnson/pipekit/commit/9b992e625cb076b9951329555420597429fc161e))
* **evaluate:** implement the benchmark ladder orchestration layer ([#31](https://github.com/jejjohnson/pipekit/issues/31)) ([e0c739e](https://github.com/jejjohnson/pipekit/commit/e0c739e252db25b55e0fa5d810cec098e8fe96e5))


### Bug Fixes

* workspace cleanup — registry hardening, correctness fixes, API consistency, docs, tests ([#49](https://github.com/jejjohnson/pipekit/issues/49)) ([9e8daac](https://github.com/jejjohnson/pipekit/commit/9e8daac1ced6d8428693ff0c981f41401b950955))

## 0.0.1 (2026-05-22)


### Features

* bootstrap pipekit monorepo with core, cycle, and experiment packages ([#6](https://github.com/jejjohnson/pipekit/issues/6)) ([9b992e6](https://github.com/jejjohnson/pipekit/commit/9b992e625cb076b9951329555420597429fc161e))
