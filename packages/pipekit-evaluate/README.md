<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-evaluate/logo-dark.svg">
  <img alt="pipekit-evaluate" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-evaluate/logo-light.svg" width="457">
</picture>

**Tagged scores, content-hashed evaluation reports and a benchmark ladder that scores each inference rung against the simpler one below it, for any carrier, on pipekit-ops and the protocols of pipekit-cycle.**

[![Tests](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pipekit/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pipekit/)
[![codecov](https://codecov.io/gh/jejjohnson/pipekit/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pipekit)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

📖 [Documentation](https://jejjohnson.github.io/pipekit/) ·
🚀 [Quickstart](https://jejjohnson.github.io/pipekit/quickstart-evaluate/) ·
📚 [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-evaluate/) ·
🧭 [Design](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-evaluate/docs/design)

</div>

---

Consider a methane plume retrieval, and then an emission-rate estimate on top of it. We have an exact but slow posterior for the source rate (MCMC on a plume model) and a network that gives an answer in milliseconds. An RMSE against the truth tells us very little here: a field can match pixel for pixel and get the plume's shape wrong, a posterior can have the right mean and the wrong spread, and the fast network should first be judged against the slow posterior it replaces, and only then against the truth. So pipekit-evaluate treats an evaluation as a set of tagged values, one for every value every scorer returns,

```math
\mathcal{R}(\hat{\mathbf{y}}, \mathbf{y}_\text{ref}) = \left\{ \left( k, m, s_{k,m}(\hat{\mathbf{y}}, \mathbf{y}_\text{ref}), \tau_k \right) : k \in \mathcal{K}, \ m \in \mathcal{M}_k \right\}
```

where $`\hat{\mathbf{y}}`$ is the prediction and $`\mathbf{y}_\text{ref}`$ the reference, in whatever carrier the scorers take (an array, an xarray Dataset, an estimate with a posterior), $`\mathcal{K}`$ is the set of scorers, $`\mathcal{M}_k`$ the names of the values scorer $`k`$ returns, $`s_{k,m}(\hat{\mathbf{y}}, \mathbf{y}_\text{ref}) \in \mathbb{R}`$ one of those values, and $`\tau_k = (u_k, \ell_k, \mathcal{S}_k)`$ the scorer's tags: the unit $`u_k`$ it scores, the lens $`\ell_k`$ it looks through and the set of stages $`\mathcal{S}_k`$ at which it may run. $`\mathcal{R}`$ is the `EvaluationReport`, which we can filter on any tag, diff against an earlier run and identify by a content hash. On top of it sits the benchmark ladder, which decides for every cell of a level × rung × complexity cube which reference $`\mathbf{y}_\text{ref}`$ is.

```python
# Shapes: H × W = 32 × 32 pixels; N = 25 per-pass estimates of one source rate
# (rmse, pattern_corr, retrieved and truth as in the quickstart below; the ladder's
# scorer, task and estimators as in the docs quickstart)

# today (pipekit-evaluate)
# s_k with τ_k; an untagged scorer is a point-wise field score
scorers = [pev.MetricScorer(rmse), pev.MetricScorer(pattern_corr, lens="structural")]
# today (pipekit-evaluate)
# ℛ(ŷ, y_ref), one tagged entry per value, under one content hash
report = pev.evaluate(retrieved, truth, scorers)
# today (pipekit-evaluate)
# each cell scored against the reference its rung names, as the same kind of report
ladder = pev.Benchmark(task, estimators, [scorer]).run().to_report()
```

None of it is a new metrics library. The metrics stay in the carrier libraries ([`xrops.metrics`](https://github.com/jejjohnson/geotoolz/tree/main/packages/xrtoolz-ops) for xarray, [`geotoolz`](https://github.com/jejjohnson/geotoolz) for GeoTensors), which never import this package, and `MetricScorer` wraps any of them, or any other `(prediction, reference) -> result` callable, with its tags. The content hash is [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops)'s (`pipekit_ops.saving`), so a report is identified the way a pipeline is. And a benchmark `Task` returns [pipekit-cycle](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle)'s `ForwardModel` and `ObservationOperator` as its simulator and observation operator, so a benchmark is built from the same parts as a `DACycle`; this package names them for type checking only, so pipekit-cycle is an extra.

## 🌟 Highlights

- 🏷️ **Three tags on every scorer**: `Unit`, `Lens` and `Stage` are read off any scorer as `ScorerTags` by `tags_of`, so a scorer with only a `name` is a point-wise field score for validation and final evaluation.
- 🔌 **Any metric becomes a scorer**: `MetricScorer` wraps a `(prediction, reference) -> result` callable and flattens a Dataset, a mapping or a scalar into named floats, and it rejects a value that still has a dimension.
- 🎲 **No uncertainty dropped silently**: handed an `Estimate`, a probabilistic `MetricScorer` refuses to score its point unless `from_estimate=` says what to take (`lambda e: e.sample(100)`).
- 🧾 **Reports to filter, diff and hash**: `EvaluationReport.filter` cuts along unit, lens, stage, scorer, cell or reference, `diff` returns what changed against a baseline, and `content_hash` ignores entry order and the timestamp.
- 🪜 **The benchmark ladder as data**: `Benchmark` fills the cells of a level × rung × complexity cube, and the tables of `ReferenceRule` say what each cell is scored against, so a domain overrides a table and not a code branch.
- 🎯 **Several references per cell**: the amortized rung is scored against the oracle's posterior and, by the scorers that ask for it in `references`, against the known truth in the same run.
- 🧩 **Protocols, not inheritance**: `Task`, `Estimator`, `Estimate`, `CovarianceEstimate`, `Prior`, `Oracle` and `Scorer` are runtime-checkable, so a domain package satisfies them without importing pipekit-evaluate.

## 📐 Three axes on the scorer, three on the cell

A score is placed twice. The scorer that made it carries three tags, and a benchmark run puts it in one cell of a cube with three more axes.

| Axis | Values | Says | Name |
|---|---|---|---|
| Unit | field, statistic, trajectory, event, budget | what is scored | `Unit` |
| Lens | point-wise, probabilistic, spectral, structural, detection, physical | what failure it can see | `Lens` |
| Stage | training, validation, final, monitoring | when it may run | `Stage` |
| Level | `L0` to `L4` | how processed the data are | `Cell.level` |
| Rung | 1 to 6 | which inference stage | `Cell.rung` |
| Complexity | 0, 1, 2, … | how rich the method on its rung | `Cell.complexity` |

Because the stage bounds how cheap and differentiable a scorer must be, an untagged scorer is not assumed fit for training or monitoring, and runs at validation and final evaluation only (`DEFAULT_STAGES`). The levels run from unstructured observations (`L0`) to reanalysis and forecast (`L4`), and the rungs from a simple model (1) through model-based inference (2), an emulator (3), emulator-based inference (4) and an amortized predictor (5) to improving any of them (6).

In a benchmark run each value is

```math
v_{c,\rho,k,m} = s_{k,m}(q_c, \mathbf{y}_\rho)
```

where $`c = (L, r, j)`$ is a cell with level $`L`$, rung $`r`$ and complexity step $`j`$, $`q_c`$ is the estimate the estimator in that cell returned (a point, a posterior or both), $`\rho`$ is a reference source and $`\mathbf{y}_\rho`$ what it resolves to. The rule is that the simpler thing is the ground truth. Along the staged axis, rung 2 is scored against the task's known parameters $`\boldsymbol{\theta}`$, rung 3 against the simulator's output, rungs 4 and 5 against the posterior of the rung-2 oracle, and rung 6 against the best validated rung below it, while rung 1 is generative and not scored; along the complexity axis, a richer step on rungs 1 to 4 is scored against the simpler step on its rung. The docs quickstart has both tables and the further references.

## 📥 Install

pipekit-evaluate is not on PyPI, and neither is pipekit-ops, which it needs, so install both from GitHub. It brings in pipekit-ops, NumPy and SciPy (no module of the package imports NumPy or SciPy yet), and needs Python 3.12 or newer.

```bash
pip install "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops" "pipekit-evaluate @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-evaluate"
```

Three extras add the rest: `cycle` brings in pipekit-cycle, to build or type-check a `Task` against the concrete `ForwardModel` and `ObservationOperator`; `xskill` brings in xskillscore and `proper` brings in properscoring, for scorers written around their metrics, since this package imports nothing from either. `EvaluationReport.to_pandas` needs pandas, which no extra installs. A project that depends on pipekit-evaluate tells uv where each pipekit package lives, because none of them is on PyPI, and pins them all to one release tag (a tag names one commit of the whole workspace, so one tag serves every package):

```toml
[project]
dependencies = ["pipekit-evaluate[cycle]"]

[tool.uv.sources]
pipekit-ops = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-ops", tag = "pipekit-array-v0.3.1" }
pipekit-cycle = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-cycle", tag = "pipekit-array-v0.3.1" }
pipekit-evaluate = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-evaluate", tag = "pipekit-array-v0.3.1" }
```

## 🛰️ Quickstart: scoring a retrieved plume

A retrieved plume of XCH₄ enhancement $`\hat{\alpha}(\mathbf{s})`$ against the true one $`\alpha(\mathbf{s})`$, both in ppm·m on a $`32 \times 32`$ scene, where $`\mathbf{s} = (\text{row}, \text{col})`$ is a pixel. The truth is a Gaussian plume of amplitude $`A = 800`$ ppm·m, and the retrieval has the amplitude 760 ppm·m plus noise $`\varepsilon(\mathbf{s}) \sim \mathcal{N}(0, 40^2)`$ in each pixel, so $`\hat{\mathbf{y}} = \hat{\alpha}`$ and $`\mathbf{y}_\text{ref} = \alpha`$. Two scorers look at it: a point-wise RMSE, and the pattern correlation $`r`$, which sees the plume's shape, computed from the departures $`\hat{\alpha}'`$ and $`\alpha'`$ of each field from its scene mean.

```python
import numpy as np
import pipekit_evaluate as pev


# Shapes: H × W = 32 × 32 pixels of one scene

rng = np.random.default_rng(0)
# s = (row, col), the pixel coordinates
# → (H, W) each
rows, cols = np.mgrid[0:32, 0:32]


# user
# α(s) = A exp(−(col − 12)² / 40 − (row − 16)² / 8), XCH₄ enhancement in ppm·m
def plume(amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-((cols - 12) ** 2) / 40 - (rows - 16) ** 2 / 8)


# α, the true plume, with A = 800 ppm·m
# → (H, W)
truth = plume(800.0)
# α̂ = α(s; A = 760) + ε(s), with ε(s) ~ 𝒩(0, 40²) in each pixel
# → (H, W)
retrieved = plume(760.0) + rng.normal(0.0, 40.0, size=(32, 32))


# user
# RMSE = √(mean_s (α̂(s) − α(s))²)
def rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - reference) ** 2)))


# user
# r = Σ_s α̂′(s) α′(s) / √(Σ_s α̂′(s)² Σ_s α′(s)²), with ′ the departure from the mean
def pattern_corr(prediction: np.ndarray, reference: np.ndarray) -> float:
    prediction_anomaly = prediction - np.mean(prediction)
    reference_anomaly = reference - np.mean(reference)
    covariance = np.sum(prediction_anomaly * reference_anomaly)
    variances = np.sum(prediction_anomaly**2) * np.sum(reference_anomaly**2)
    return float(covariance / np.sqrt(variances))


# today (pipekit-evaluate)
# RMSE untagged, so a point-wise field score for validation and final evaluation;
# r tagged structural, for the final evaluation only
scorers = [
    pev.MetricScorer(rmse),
    pev.MetricScorer(pattern_corr, lens="structural", stages="final"),
]
# today (pipekit-evaluate)
# ℛ(α̂, α), one tagged entry per value
report = pev.evaluate(retrieved, truth, scorers, model_ref="plume-net@3f2a")
# today (pipekit-evaluate)
# the entries whose scorer sees structural failures
structural = report.filter(lens="structural")
# today (pipekit-evaluate)
# a validation pass runs only the scorers tagged for validation
validation = pev.evaluate(retrieved, truth, scorers, stage="validation")
# today (pipekit-evaluate)
# the same scores of the same model hash equal, whenever they were made
again = pev.evaluate(retrieved, truth, scorers, model_ref="plume-net@3f2a")

print(
    {key: round(value, 3) for key, value in report.values.items()},
    list(structural.values),
    list(validation.values),
    again.content_hash == report.content_hash,
)
```

This prints `{'rmse.value': 39.624, 'pattern_corr.value': 0.95} ['pattern_corr.value'] ['rmse.value'] True`: an RMSE of 39.6 ppm·m, about the noise we added, and a pattern correlation of 0.95, so the retrieval has the plume's shape; then the one structural entry, the one scorer a validation pass runs, and two reports of the same scores with the same hash, though they were made at different times. Each entry is keyed `"<scorer>.<metric>"` and carries its scorer's tags. Nothing here is specific to plumes, so swapping the two functions for `xrops.metrics` operators on a DataArray, or the arrays for estimates with a posterior, gives the same report.

The [quickstart](https://jejjohnson.github.io/pipekit/quickstart-evaluate/) takes this further: the benchmark ladder for the source rate behind this plume, with an exact conjugate posterior as the rung-2 oracle and an amortized head on rung 5 scored against it and against the truth, the reference rule in full, and how scorers, reports and estimates behave at their edges.

## 🔬 What else is in the box

Each public name sits on one rung of the GeoStack role ladder, by the job it does: the scorers are strategies (②), the estimates and estimators models (③), `evaluate` an algorithm (④) and `Benchmark` a recipe (⑥), while the tags, the cells and the reports are the structures (⓪) the rest act on. The table lists them by area.

| Area | Symbols | Notes |
|---|---|---|
| `pipekit_evaluate.scoring` | `Unit`, `Lens`, `Stage`, `ScorerTags`, `tags_of`, `DEFAULT_STAGES`, `MetricScorer`, `Scorer`, `Estimate`, `CovarianceEstimate`, `ReferenceSource`, `as_sources` | The tags read off any scorer, the scorer of any metric, the protocols of what a method returns and what grades it, and where a score's ground truth comes from |
| `pipekit_evaluate.reports` | `evaluate`, `EvaluationReport`, `ReportEntry` | One run of every scorer, collected under a content hash, with `filter`, `diff`, `to_dict`, `to_records` and `to_pandas` |
| `pipekit_evaluate.benchmark` | `Cell`, `LEVEL_NAMES`, `RUNG_NAMES`, `Task`, `Estimator`, `Prior`, `Oracle`, `Axis`, `ReferenceRule`, `DEFAULT_STAGED`, `DEFAULT_COMPLEXITY`, `DEFAULT_ALSO`, `Benchmark`, `BenchmarkReport`, `CellResult`, `scores_against` | The cube, the contracts a domain satisfies, the reference rule as tables, and the runner, whose result becomes an `EvaluationReport` |

An area imports only from the ones before it, in the order of the table ([E1](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/decisions.md)). `pipekit_evaluate` re-exports every name above except `as_sources`, `LEVEL_NAMES`, `RUNG_NAMES`, the three `DEFAULT_*` tables of the reference rule and `scores_against`, which are imported from their area.

## 📖 Documentation

- **[Quickstart](https://jejjohnson.github.io/pipekit/quickstart-evaluate/)**: the benchmark ladder worked through, the reference rule in full, and the calling conventions of the scorers, the reports and the runner.
- **[API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-evaluate/)**: every public name, generated from the docstrings.
- **[Protocols](https://jejjohnson.github.io/pipekit/protocols/)**: the members of every benchmark protocol, beside the other protocols of the workspace.
- **[The benchmark ladder](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/benchmark_ladder.md)**: the domain-independent design, with its build order and its open questions.
- **The four pilots**, each working the ladder for one domain: [plume](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/plume_pilot.md) (methane retrieval through emission inference to persistency forecasts, the recommended first pilot), [ocean](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/ocean_pilot.md) (SSH, SST, SSS and ocean colour), [land extremes](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/land_extremes_pilot.md) (the Event unit, the Detection lens and tail calibration) and [fusion](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/fusion_pilot.md) (GEO, LEO and in-situ observations merged into one L3 analysis).
- **[Decisions](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/decisions.md)**: why the package is three areas, with the shared protocols at the bottom.
- **[Report 14 of the GeoStack](https://github.com/jejjohnson/geostack/blob/main/docs/reports/14_pipekit_evaluate.md)**: why the metrics stay with their carriers while this package orchestrates them, and the tags and the report of its workstream A.

## 🧩 Ecosystem

pipekit-evaluate is a package of the pipekit [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), in Core tier 0 of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the tags, the report and the ladder, and composes the rest in:

| Concern | Owner |
|---|---|
| The content hash (`sha256_hex`, `stable_json`) | [`pipekit-ops`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) |
| The simulator and observation operator of a `Task` (the `cycle` extra) | [`pipekit-cycle`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) |
| The metrics a scorer wraps | [`xrops.metrics`](https://github.com/jejjohnson/geotoolz/tree/main/packages/xrtoolz-ops) and [`geotoolz`](https://github.com/jejjohnson/geotoolz) |
| Gridded and ensemble verification (the `xskill` extra) | [`xskillscore`](https://github.com/xarray-contrib/xskillscore) |
| Proper scores such as CRPS (the `proper` extra) | [`properscoring`](https://github.com/properscoring/properscoring) |

Dependencies point one way: pipekit-evaluate imports pipekit-ops, and pipekit-cycle only for type checking, and the carrier libraries whose metrics it wraps do not import it. No package of the stack imports pipekit-evaluate on its `main` today; the GeoStack's report notebooks 14, 15 and 17 do, for the benchmark and scoring areas.

## 🌳 Repository layout

```
packages/pipekit-evaluate/
├── src/pipekit_evaluate/
│   ├── __init__.py   ← re-exports the names of the three areas
│   ├── scoring/      ← the tags, MetricScorer, Estimate, Scorer, ReferenceSource
│   ├── reports/      ← evaluate, EvaluationReport, ReportEntry
│   └── benchmark/    ← the cube, Task, Estimator, Oracle, ReferenceRule, Benchmark
├── tests/            ← one file per area, and the import rule between areas
└── docs/design/      ← the benchmark ladder, the four pilots and the decisions
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install                                    # uv sync --all-groups except train-extras, + pre-commit hooks
uv run pytest packages/pipekit-evaluate/tests   # this package's tests
make lint                                       # ruff check on the whole repo
make typecheck                                  # ty on every package
make docs-serve                                 # build the docs site and serve it locally
```

`make help` lists every task. [`AGENTS.md`](https://github.com/jejjohnson/pipekit/blob/main/AGENTS.md) holds the rules for contributors and coding agents.

## 📄 License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/LICENSE).
