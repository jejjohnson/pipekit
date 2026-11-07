<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-evaluate/logo-dark.svg">
    <img alt="pipekit-evaluate" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-evaluate/logo-light.svg" width="457">
  </picture>
</p>

**Tagged scores, content-hashed evaluation reports and a benchmark ladder that scores each inference rung against the simpler one below it.**

<p align="center">
  <a href="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Version 0.2.0" src="https://img.shields.io/badge/version-0.2.0-blue">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="https://jejjohnson.github.io/pipekit/"><b>Docs</b></a> ·
  <a href="https://jejjohnson.github.io/pipekit/api/pipekit-evaluate/"><b>API</b></a> ·
  <a href="https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/benchmark_ladder.md"><b>Benchmark ladder</b></a>
</p>

## Why it exists

Consider a methane plume retrieval, then an emission-rate estimate on top of it.
We have an exact but slow posterior for the source rate (MCMC on a plume model), and a network that gives an answer in milliseconds.
An RMSE against the truth tells us very little here: a field can match pixel for pixel and get the plume's shape wrong, a posterior can have the right mean and the wrong spread, and the fast network should first be judged against the slow posterior it replaces, then against the truth.

pipekit-evaluate gives every score three tags (what it scores, what failure it can see, where in a model's life it may run) and collects a run's scores into one report you can filter, diff and hash.
On top of that sits the benchmark ladder: a cube of data level × inference rung × complexity step, where each cell is scored against the "simpler thing" below it, as data rather than as five separate scripts.
The metrics themselves stay in the carrier libraries (`xrops.metrics`, `geotoolz`), which never import this package.

## Where it sits

pipekit-evaluate is built on pipekit-ops; it does not use pipekit-array.
Both are libraries in their own right, which you can build on directly.

```
   domain packages (plumax, somax, xtremax, ...): a Task, Estimators
   carrier libraries (xrops.metrics, geotoolz): the metrics
                     │
          ┌──────────┴──────────┐
          │  pipekit-evaluate   │  ◀ this package
          │ taxonomy, scorers,  │
          │ reports, benchmark  │
          └──────────┬──────────┘
            ┌────────┴────────┬──────────────────┐
            │                 │                  │
       pipekit-ops       pipekit-array      pipekit-cycle
       (dependency:      (not used)         ([cycle] extra:
        hashing)                             ForwardModel and
                                             ObservationOperator
                                             for a Task)
```

- **pipekit-ops** is a required dependency, beside NumPy and SciPy.
  The report hash uses `pipekit_ops.hashing`, and a metric can be any callable, a pipekit operator included.
- **pipekit-array** is not a dependency and is not used.
- **pipekit-cycle** is optional: a benchmark `Task` returns its `ForwardModel` and `ObservationOperator`, which this package names only for type checking.

## Install

```bash
uv add pipekit-evaluate
uv add 'pipekit-evaluate[cycle]'    # plus pipekit-cycle
```

| Extra | What it adds | When you need it |
| --- | --- | --- |
| *(none)* | `pipekit-ops`, `numpy`, `scipy` | the taxonomy, scorers, reports and benchmark ladder |
| `cycle` | `pipekit-cycle` | building or type-checking a `Task` against the concrete `ForwardModel` and `ObservationOperator` |
| `xskill` | `xskillscore` | your scorers wrap xskillscore metrics; this package imports nothing from it |
| `proper` | `properscoring` | your scorers wrap proper scores (CRPS); this package imports nothing from it |

## Quickstart

Score a retrieved plume of XCH₄ enhancement $\hat{\alpha}(\mathbf{s})$ against the true one $\alpha(\mathbf{s})$ (both in ppm·m on a 32 × 32 scene) with a point-wise RMSE and a structural pattern correlation.

```python
import numpy as np
import pipekit_evaluate as pev

# Shapes: H, W = 32 × 32 pixels of one scene
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:32, 0:32]                                # (H, W) each


# user
# a Gaussian plume of XCH₄ enhancement α(s) [ppm·m], and a retrieval of it
def plume(amplitude):
    return amplitude * np.exp(-((xx - 12) ** 2 / 40 + (yy - 16) ** 2 / 8))  # (H, W)


truth = plume(800.0)                                         # (H, W)
retrieved = plume(760.0) + rng.normal(0.0, 40.0, (32, 32))  # (H, W)


# user
# RMSE = √(mean (α̂ − α)²)
def rmse(pred, ref):
    return float(np.sqrt(np.mean((pred - ref) ** 2)))       # → ()


# user
# ρ = corr(α̂, α) over the pixels, the plume's shape
def pattern_corr(pred, ref):
    return float(np.corrcoef(pred.ravel(), ref.ravel())[0, 1])  # → ()


# today (pipekit-evaluate)
scorers = [
    pev.MetricScorer(rmse),
    pev.MetricScorer(pattern_corr, lens="structural", stages="final"),
]
report = pev.evaluate(retrieved, truth, scorers, model_ref="plume-net@3f2a")
print(sorted(report.values))          # ['pattern_corr.value', 'rmse.value']
print(list(report.filter(lens="structural").values))  # ['pattern_corr.value']
print(list(pev.evaluate(retrieved, truth, scorers, stage="validation").values))
# ['rmse.value']
again = pev.evaluate(retrieved, truth, scorers, model_ref="plume-net@3f2a")
print(again.content_hash == report.content_hash, again.diff(report))  # True {}
```

Now the ladder, for the source rate $\theta = Q$ [kg/h] behind that plume.
We have $N = 25$ per-pass estimates $y_n = \theta + \varepsilon_n$ with $\varepsilon_n \sim \mathcal{N}(0, \sigma^2)$, $\sigma = 20$ kg/h, and a prior $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$ with $\mu_0 = 100$ and $\sigma_0 = 50$ kg/h.
Rung 2 is the exact conjugate posterior; rung 5 is a cheap amortized head.
The ladder scores rung 2 against the known truth and rung 5 against rung 2's posterior, and, because the scorer asks for it, against the truth as well.
This block uses the `[cycle]` extra for the task's simulator.

```python
import pipekit_cycle as pc

# Shapes: N = 25 per-pass estimates of one source rate θ = Q [kg/h]
theta_true = 120.0
obs = theta_true + rng.normal(0.0, 20.0, 25)                 # (N,)


class PlumeTask:  # user: satisfies pev.Task
    name, domain = "single-source", "plume"

    def prior(self):
        return None  # user: a Prior with sample / log_prob

    def simulator(self):
        return pc.CallableForward(lambda q, dt: q, dt=1.0)

    def observation_operator(self):
        return pc.IdentityObs()

    def datasets(self):
        return {"obs": obs, "theta": theta_true}


class Gaussian:  # user: satisfies pev.CovarianceEstimate
    def __init__(self, mean, var):
        self.mean, self.var = mean, var

    def point(self):
        return self.mean

    def sample(self, n, *, key=None):
        return np.random.default_rng(key).normal(self.mean, np.sqrt(self.var), n)

    def log_prob(self, theta):
        return -0.5 * (theta - self.mean) ** 2 / self.var

    def covariance(self):
        return np.array([[self.var]])


class ConjugateOracle:  # user: rung 2, the exact posterior
    rung, complexity = 2, 0

    def fit(self, task):
        return self

    def __call__(self, y):
        # θ ~ 𝒩(μ₀, σ₀²), yₙ ~ 𝒩(θ, σ²):  σ_post⁻² = σ₀⁻² + N σ⁻²
        prec = 1 / 50.0**2 + len(y) / 20.0**2                 # → ()
        # μ_post = σ_post² (μ₀ σ₀⁻² + ∑ₙ yₙ σ⁻²)
        mean = (100.0 / 50.0**2 + y.sum() / 20.0**2) / prec   # → ()
        return Gaussian(mean, 1 / prec)


class MedianHead:  # user: rung 5, a cheap amortized point estimate
    rung, complexity = 5, 0

    def fit(self, task):
        return self

    def __call__(self, y):
        return Gaussian(float(np.median(y)), 4.0)


# user
# |θ̂ − θ_ref|
def abs_error(pred, ref):
    return abs(pred - ref)                                    # → ()


# today (pipekit-evaluate)
bench = pev.Benchmark(
    task=PlumeTask(),
    estimators=[ConjugateOracle(), MedianHead()],
    scorers=[pev.MetricScorer(abs_error, references={"oracle_posterior", "known_theta"})],
)
result = bench.run()
for label, cell in result.to_dict()["cells"].items():
    print(label, cell["reference"], sorted(cell["scores"]))
# L4/r2/c0 known_theta ['known_theta']
# L4/r5/c0 oracle_posterior ['known_theta', 'oracle_posterior']
truth_view = result.to_report().filter(reference="known_theta")
print(sorted(truth_view.values))
# ['L4/r2/c0:known_theta:abs_error.value', 'L4/r5/c0:known_theta:abs_error.value']
```

## What ships

Version 0.2.0.
Every name below is exported from `pipekit_evaluate`, except where the table names `pipekit_evaluate.benchmark`.

| Module | Public names | Purpose |
| --- | --- | --- |
| `taxonomy` | `Unit`, `Lens`, `Stage`, `ScorerTags`, `tags_of`, `DEFAULT_STAGES` | The Unit × Lens × Stage tags, read off any scorer with defaults |
| `scorers` | `MetricScorer` | Turn any `(prediction, reference) -> result` metric into a tagged scorer |
| `report` | `evaluate`, `EvaluationReport`, `ReportEntry` | Run scorers once; filter, diff and hash the result |
| `benchmark.cube` | `Cell` (`LEVEL_NAMES`, `RUNG_NAMES` in `pipekit_evaluate.benchmark`) | One level × rung × complexity cell |
| `benchmark.protocols` | `Task`, `Estimator`, `Estimate`, `CovarianceEstimate`, `Prior`, `Oracle`, `Scorer` | Runtime-checkable contracts that domain packages satisfy |
| `benchmark.reference` | `ReferenceRule`, `ReferenceSource`, `Axis` (`DEFAULT_STAGED`, `DEFAULT_COMPLEXITY`, `DEFAULT_ALSO`, `as_sources` in `pipekit_evaluate.benchmark`) | The "simpler thing is ground truth" rule, as tables |
| `benchmark.run` | `Benchmark`, `BenchmarkReport`, `CellResult` (`scores_against` in `pipekit_evaluate.benchmark`) | Fit every estimator, resolve each cell's references, score |

### Not built yet

The build order is §7 of the [benchmark ladder design](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/benchmark_ladder.md); step 1 has landed, with the level × rung × complexity report of step 6.

- **Probabilistic scorers**: simulation-based calibration ranks, coverage, CRPS.
- **Posterior-distance scorers** to the oracle: C2ST, MMD, sliced Wasserstein.
- **Physical-constraint scorers** (mass and flux budgets, balance laws); these belong with the domain operators.
- **A pluggable oracle.** The `Oracle` protocol exists but `Benchmark` does not consume it yet; it takes `reference=` or the richest rung-2 estimator.
- **Per-cell runs.** `Benchmark` does not write a `pipekit_experiment.Run` per cell yet.
- **The pilot tasks** in the design docs; a domain joins by shipping its own `Task`, outside this package.

## How it works

### Unit × Lens × Stage

A scorer places itself on three axes.
**Unit** is what is scored: field, statistic, trajectory, event or budget.
**Lens** is what kind of failure it can see: point-wise, probabilistic, spectral, structural, detection or physical constraint.
**Stage** is where it may run: training, validation, final or monitoring, which bounds how cheap and differentiable it has to be.
The tags are optional attributes on the scorer (`unit`, `lens`, `stages`, `differentiable`), not on the metric, and `tags_of(scorer)` fills in the defaults, so a scorer with only a `name` is a point-wise field score for validation and final evaluation.

### `MetricScorer`

`MetricScorer(metric, lens=..., stages=...)` wraps any callable and flattens its result into named floats: one per data variable of a Dataset, one per key of a mapping, or a single value.
Every value must be a scalar with no dimensions, so a metric with a lead-time or region dimension is reduced first.
Handed an `Estimate`, it passes the metric the estimate's point by default.
A probabilistic scorer refuses that, because a point would drop the uncertainty it is meant to score, so you say what to take with `from_estimate=` (for example `lambda e: e.sample(100)`).
Because the values become Python floats, a `MetricScorer` is never differentiable; a scorer meant for use as a loss implements `Scorer` itself.

### `EvaluationReport`

`evaluate(prediction, reference, scorers)` returns one `ReportEntry` per value, keyed `"<scorer>.<metric>"`, each carrying its tags.
`filter(unit=, lens=, stage=, scorer=, cell=, reference=)` cuts the report along any axis, and `diff(baseline)` returns the keys whose value changed, as `(before, after)`.
`content_hash` is a sha256 over the entries, `model_ref`, `dataset_ref` and the config; it ignores entry order and the timestamp, so two runs that score the same values hash equal.
The config is snapshotted as JSON when the report is made, so changing your dict afterwards changes neither the report nor its hash.

### The benchmark ladder

A `Cell` is a data level (`L0` unstructured observations to `L4` reanalysis and forecast), an inference rung (1 simple model, 2 model-based inference, 3 emulator, 4 emulator-based inference, 5 amortized predictor, 6 improve) and a complexity step on that rung (0 is the simplest).
The `ReferenceRule` says what each cell is scored against.
Along the staged axis, rung 2 is scored against the known θ, rung 3 against the simulator output, rungs 4 and 5 against the rung-2 oracle's posterior, and rung 6 against the best validated rung below it.
Along the complexity axis, a richer step is scored against the simpler step on its rung.
These are tables on the rule, not code branches, so a domain without an oracle overrides a table.
Beside its primary reference, a rung may name further ones: rung 5 is also scored against the known θ, but only by scorers that list it in their `references`, so a scorer written for a posterior is never handed the raw truth.
`CellResult.by_reference` holds a cell's scores against each reference, and `BenchmarkReport.to_report()` turns a run into the same `EvaluationReport`, keyed `"<cell>:<reference>:<scorer>.<metric>"`.

### Protocols, not inheritance

`Task`, `Estimator`, `Estimate`, `CovarianceEstimate`, `Prior`, `Oracle` and `Scorer` are runtime-checkable, in the style of `pipekit_cycle.protocols`.
Domain packages (`plumax`, `somax`, `xtremax`) satisfy them without importing pipekit-evaluate, and a `Task` reuses pipekit-cycle's `ForwardModel` and `ObservationOperator` as its simulator and observation operator, so a benchmark is built from the same parts as a `DACycle`.

## Learn more

- API reference: [pipekit-evaluate](https://jejjohnson.github.io/pipekit/api/pipekit-evaluate/).
- Design: [the benchmark ladder](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/benchmark_ladder.md), the domain-independent framework.
- The four pilots, each working the ladder for one domain:
  - [Plume](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/plume_pilot.md): methane retrieval through emission inference to persistency forecasts; the recommended first pilot.
  - [Ocean](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/ocean_pilot.md): SSH, SST, SSS and ocean colour; a new domain joins by shipping one `Task`.
  - [Land extremes](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/land_extremes_pilot.md): the Event unit, the Detection lens and tail calibration.
  - [Fusion](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/fusion_pilot.md): merging GEO, LEO and in-situ observations into one L3 analysis.
- Report 14 of the GeoStack: [pipekit-evaluate](https://github.com/jejjohnson/geostack/blob/main/docs/reports/14_pipekit_evaluate.md), workstream A.
- The foundations: [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) and [pipekit-array](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array).
- Siblings: [pipekit-cycle](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) (the simulator and observation protocols), [pipekit-experiment](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment) (registry and run records).

## License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/LICENSE).
