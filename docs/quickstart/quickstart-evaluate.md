# pipekit-evaluate quickstart

This page picks up where the [README](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/README.md) stops. The README scores a retrieved plume with two tagged scorers and collects the values into one report; here we work the benchmark ladder for the source rate behind that plume, give the reference rule in full, and say how the scorers, the reports and the runner behave at their edges. The notation is the README's.

An evaluation is a set of tagged values, one for every value every scorer returns,

$$
\mathcal{R}(\hat{\mathbf{y}}, \mathbf{y}_\text{ref}) = \left\{ \left( k, m, s_{k,m}(\hat{\mathbf{y}}, \mathbf{y}_\text{ref}), \tau_k \right) : k \in \mathcal{K}, \ m \in \mathcal{M}_k \right\}
$$

where $\hat{\mathbf{y}}$ is the prediction and $\mathbf{y}_\text{ref}$ the reference, in whatever carrier the scorers take, $\mathcal{K}$ is the set of scorers, $\mathcal{M}_k$ the names of the values scorer $k$ returns, $s_{k,m}(\hat{\mathbf{y}}, \mathbf{y}_\text{ref}) \in \mathbb{R}$ one of those values, and $\tau_k = (u_k, \ell_k, \mathcal{S}_k)$ the scorer's unit, lens and set of stages. Every name on this page is in the [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-evaluate/).

## Scorers and their tags

A scorer places itself on three axes. **Unit** is what is scored: a field, a statistic, a trajectory, an event or a budget. **Lens** is what kind of failure it can see: point-wise, probabilistic, spectral, structural, detection or physical constraint. **Stage** is where in a model's life it may run: training, validation, final evaluation or monitoring, which bounds how cheap and differentiable it has to be.

The tags are optional attributes on the scorer (`unit`, `lens`, `stages`, `differentiable`), not on the metric, and `tags_of(scorer)` reads them as a `ScorerTags`, filling in the defaults. So a scorer with only a `name` is a point-wise field score for validation and final evaluation (`DEFAULT_STAGES`), and it is not differentiable. `ScorerTags` accepts the string value of a member wherever it accepts the member, and one stage on its own, so `ScorerTags(lens="structural", stages="final")` is valid; a scorer must run at one stage or more, and `tags_of` raises a `ValueError` for a value outside its axis.

A scorer is anything with a `name` and a `__call__(prediction, reference)` that returns named floats (the `Scorer` protocol). Its name may not contain `.` or `:`, which separate the parts of a report key, and the names of the scorers in one run must be unique; `evaluate` and `Benchmark.run` check both before any scorer runs.

## `MetricScorer`

`MetricScorer(metric, lens=..., stages=...)` wraps any callable `(prediction, reference) -> result`, such as an `xrops.metrics` operator, and flattens its result into named floats:

- a Dataset-like result (one with `data_vars`) gives one value per data variable;
- a mapping gives one value per key;
- anything else is one value, keyed by its `name` when it has a string one (a named `DataArray`) and by `"value"` otherwise.

Every value must be a scalar with no dimensions, even singleton ones, so a metric with a lead-time or region dimension is reduced first. The scorer's name defaults to the metric's `__name__`, or its class name.

Handed an `Estimate` as the prediction or the reference, a `MetricScorer` passes the metric the estimate's point by default. A probabilistic scorer refuses that, because a point would drop the uncertainty it is meant to score, so we say what to take with `from_estimate=` (for example `lambda e: e.sample(100)`); an estimate whose `point()` returns `None` is refused too. Because the values become Python floats, a `MetricScorer` is never differentiable; a scorer meant for use as a loss implements `Scorer` itself.

`references=` names the reference sources a scorer is run against in the benchmark ladder, as members of `ReferenceSource` or their string values. Without it a scorer is run against each cell's primary reference only (`scores_against`), so a scorer written for an oracle's posterior is never handed the raw truth because a rung gained a further reference.

## The evaluation report

`evaluate(prediction, reference, scorers)` returns an `EvaluationReport` with one `ReportEntry` per value, keyed `"<scorer>.<metric>"`, each carrying its scorer's tags. `stage=` runs only the scorers tagged for that stage, and `model_ref`, `dataset_ref` and `config` are recorded on the report.

- `filter(unit=, lens=, stage=, scorer=, cell=, reference=)` keeps the entries that match every given value, and keeps the report's references, config and timestamp.
- `diff(baseline)` returns `{key: (before, after)}` for every key whose value changed, with `None` on the side that lacks the key; two NaNs count as equal, so a report's diff against itself is empty.
- `content_hash` is a sha256 over the entries, `model_ref`, `dataset_ref` and the config. It ignores entry order and the timestamp, so two runs that score the same values on the same references hash equal.
- The config is snapshotted as JSON when the report is made, so changing our dict afterwards changes neither the report nor its hash; a value that is not JSON (a `Path`, an object) is stored as its `repr`.
- `to_dict()` is strict JSON with the content hash, writing a non-finite float as `"NaN"`, `"Infinity"` or `"-Infinity"`; `to_records()` gives one flat record per entry, and `to_pandas()` the same as a DataFrame, which needs pandas installed.

## The benchmark ladder

A `Cell` is a data level, an inference rung and a complexity step, $c = (L, r, j)$. The levels run from `L0` (unstructured observations) through `L1` (structured observations), `L2` (gap-filled observations) and `L3` (analysis) to `L4` (reanalysis and forecast); the rungs are 1 (simple model), 2 (model-based inference, the oracle), 3 (emulator), 4 (emulator-based inference), 5 (amortized predictor) and 6 (improve); and the complexity step counts from 0, the simplest method on its rung. `Cell.label` is the compact key `"L4/r5/c0"`, and `LEVEL_NAMES` and `RUNG_NAMES` hold the names.

In a benchmark run each value is

$$
v_{c,\rho,k,m} = s_{k,m}(q_c, \mathbf{y}_\rho)
$$

where $q_c$ is the estimate the estimator in cell $c$ returned, $\rho$ a reference source and $\mathbf{y}_\rho$ what it resolves to. `ReferenceRule` says which $\rho$ each cell is scored against, as tables and not code branches, so a domain without an oracle overrides a table. Along the staged axis (the default) the primary reference of each rung is the simpler thing below it:

| Rung | Primary reference | Resolves to |
|---|---|---|
| 1 | `NONE` | not scored: generative |
| 2 | `KNOWN_THETA` | `task.datasets()["theta"]` |
| 3 | `SIMULATOR_OUTPUT` | `task.datasets()["state"]` |
| 4, 5 | `ORACLE_POSTERIOR` | the rung-2 oracle's estimate |
| 6 | `BEST_VALIDATED_BELOW` | the highest validated rung below |

Along the complexity axis (`axis=Axis.COMPLEXITY`), a step is scored against the step below it on its rung (`SIMPLER_STEP`) on rungs 1 to 4, rung 5 against the oracle's posterior, and rung 6 against the best validated rung below it, while step 0 has no simpler step and is not scored. The tables are `DEFAULT_STAGED` and `DEFAULT_COMPLEXITY`; `ReferenceSource` also has `PREVIOUS_RUNG`, which no default table uses.

Beside its primary reference, a rung may name further ones (`DEFAULT_ALSO`): rung 5 is also scored against the known $\boldsymbol{\theta}$, on either axis whenever its primary reference is scored, but only by the scorers that list it in their `references`. The oracle is the `reference=` estimator when one is given, and otherwise the rung-2 estimator with the highest complexity (NUTS rather than OI, say); an explicit oracle that is also one of the estimators is run once, so its cell and the cells scored against it see the same posterior. `BEST_VALIDATED_BELOW` stays on the estimator's own complexity slice, because the ladder walks one axis at a time, and a cell counts as validated once its primary reference resolved.

`Benchmark.run()` fits every estimator, predicts from `task.datasets()["obs"]`, resolves each cell's references and scores against each. `obs_key`, `truth_key` and `state_key` change the three keys, `level` sets the cells' level (`"L4"` by default), and two estimators in one cell raise a `ValueError`. `CellResult.by_reference` holds a cell's scores against each reference that resolved, `CellResult.note` says why any other did not, and `BenchmarkReport.to_report()` turns the run into the same `EvaluationReport`, keyed `"<cell>:<reference>:<scorer>.<metric>"`, with the run's task, domain and axis in its config.

### Worked example: the source rate behind the plume

The parameter is the source rate $\theta = Q$ in kg/h (a scalar here, so not bold). We have $N = 25$ per-pass estimates $y_n = \theta + \varepsilon_n$ with $\varepsilon_n \sim \mathcal{N}(0, \sigma^2)$ and $\sigma = 20$ kg/h, and a prior $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$ with $\mu_0 = 100$ kg/h and $\sigma_0 = 50$ kg/h. Rung 2 is the exact conjugate posterior $\mathcal{N}(\mu_\text{post}, \sigma_\text{post}^2)$, with

$$
\sigma_\text{post}^{-2} = \sigma_0^{-2} + N \sigma^{-2}
$$

and

$$
\mu_\text{post} = \sigma_\text{post}^2 \left( \mu_0 \sigma_0^{-2} + \sigma^{-2} \sum_{n=1}^{N} y_n \right)
$$

Rung 5 is a cheap amortized head, the median of the passes with a fixed variance of 4 (kg/h)². The ladder scores rung 2 against the known truth, and rung 5 against rung 2's posterior and, because the scorer asks for it, against the truth as well. The scorer is the absolute error $\vert \hat{\theta} - \theta_\text{ref} \vert$, where $\hat{\theta}$ is the point of the estimate being scored and $\theta_\text{ref}$ the reference (the known $\theta$, or the point of the oracle's posterior). This block needs the `cycle` extra for the task's simulator.

```python
import numpy as np
import pipekit_cycle as pc
import pipekit_evaluate as pev


# Shapes: N = 25 per-pass estimates of one source rate θ = Q in kg/h

rng = np.random.default_rng(0)
# θ, the true source rate
theta_true = 120.0
# yₙ = θ + εₙ, with εₙ ~ 𝒩(0, σ²) and σ = 20 kg/h
# → (N,)
obs = theta_true + rng.normal(0.0, 20.0, size=25)


# user
# satisfies pev.Task: the scenario, its simulator and its data
class PlumeTask:
    name = "single-source"
    domain = "plume"

    def prior(self):
        # user
        # a Prior with sample and log_prob would go here; no rung below uses it
        return None

    def simulator(self):
        # today (pipekit-cycle)
        # θ ↦ θ, the simplest forward model
        return pc.CallableForward(lambda rate, dt: rate, dt=1.0)

    def observation_operator(self):
        # today (pipekit-cycle)
        return pc.IdentityObs()

    def datasets(self):
        return {"obs": obs, "theta": theta_true}


# user
# satisfies pev.CovarianceEstimate: 𝒩(μ, v) over θ
class Gaussian:
    def __init__(self, mean: float, var: float):
        self.mean = mean
        self.var = var

    def point(self):
        return self.mean

    def sample(self, n, *, key=None):
        return np.random.default_rng(key).normal(self.mean, np.sqrt(self.var), n)

    def log_prob(self, theta):
        # log 𝒩(θ; μ, v) up to a constant
        return -0.5 * (theta - self.mean) ** 2 / self.var

    def covariance(self):
        return np.array([[self.var]])


# user
# rung 2: the exact conjugate posterior, the oracle
class ConjugateOracle:
    rung = 2
    complexity = 0

    def fit(self, task):
        return self

    def __call__(self, y):
        # σ_post⁻² = σ₀⁻² + N σ⁻², with σ₀ = 50 and σ = 20 kg/h
        # → ()
        precision = 1 / 50.0**2 + len(y) / 20.0**2
        # μ_post = σ_post² (μ₀ σ₀⁻² + Σₙ yₙ σ⁻²), with μ₀ = 100 kg/h
        # → ()
        mean = (100.0 / 50.0**2 + np.sum(y) / 20.0**2) / precision
        return Gaussian(mean, 1 / precision)


# user
# rung 5: a cheap amortized head, the median with a fixed spread
class MedianHead:
    rung = 5
    complexity = 0

    def fit(self, task):
        return self

    def __call__(self, y):
        return Gaussian(float(np.median(y)), 4.0)


# user
# |θ̂ − θ_ref|, on the point of each estimate
def abs_error(prediction: float, reference: float) -> float:
    return abs(prediction - reference)


# today (pipekit-evaluate)
# scored against the oracle's posterior and, where a rung names it, the known θ
scorer = pev.MetricScorer(abs_error, references={"oracle_posterior", "known_theta"})
# today (pipekit-evaluate)
# rung 2 against the known θ; rung 5 against the oracle, and the known θ beside it
bench = pev.Benchmark(
    task=PlumeTask(),
    estimators=[ConjugateOracle(), MedianHead()],
    scorers=[scorer],
)
# today (pipekit-evaluate)
result = bench.run()
# today (pipekit-evaluate)
# every value as a report entry keyed "<cell>:<reference>:<scorer>.<metric>"
truth_view = result.to_report().filter(reference="known_theta")

for label, cell in result.to_dict()["cells"].items():
    errors = {
        source: round(scores["abs_error.value"], 2)
        for source, scores in cell["scores"].items()
    }
    print(label, cell["reference"], errors)
print(sorted(truth_view.values))
```

This prints

```
L4/r2/c0 known_theta {'known_theta': 1.59}
L4/r5/c0 oracle_posterior {'oracle_posterior': 0.98, 'known_theta': 2.57}
['L4/r2/c0:known_theta:abs_error.value', 'L4/r5/c0:known_theta:abs_error.value']
```

The rung-2 posterior mean is 1.59 kg/h from the true rate of 120 kg/h. The amortized head is 0.98 kg/h from the oracle's mean, which is how well it reproduces the slow method, and 2.57 kg/h from the truth, which is what the user of the fast method gets. Only the known-$\theta$ entries are left after the filter, one per rung, because the report carries each entry's cell and reference.

## Protocols, not inheritance

`Task`, `Estimator`, `Estimate`, `CovarianceEstimate`, `Prior`, `Oracle` and `Scorer` are runtime-checkable, in the style of `pipekit_cycle.protocols`, and their members are on the [protocols](../protocols.md) page. Domain packages (`plumax`, `somax`, `xtremax`) can satisfy them without importing pipekit-evaluate, as the classes of the worked example do, and a `Task` reuses pipekit-cycle's `ForwardModel` and `ObservationOperator` as its simulator and observation operator, so a benchmark is built from the same parts as a `DACycle`. pipekit-evaluate names those two types under `TYPE_CHECKING` only, so importing it never imports pipekit-cycle.

`Estimate.sample` is the required posterior surface and `log_prob` is optional, and a point-only estimate returns `None` from `sample`. A scorer that needs second moments (Gaussian coverage, a Mahalanobis distance, a spread-skill ratio) checks `isinstance(estimate, CovarianceEstimate)`, whose `covariance()` is a $(d, d)$ matrix for a $(d,)$ point. `Prior` is deliberately thin (`sample` and an optional `log_prob`), so neither NumPyro nor distrax is a dependency.

## Where it sits

pipekit-evaluate is built on pipekit-ops, which it needs, beside NumPy and SciPy; the report hash uses `pipekit_ops.saving`, and a metric can be any callable, a pipekit operator included. It does not use pipekit-array. pipekit-cycle is the optional `cycle` extra, for the `ForwardModel` and `ObservationOperator` a `Task` returns. The `xskill` and `proper` extras bring in xskillscore and properscoring for scorers that wrap their metrics; this package imports nothing from either. The metrics themselves stay in the carrier libraries (`xrops.metrics`, `geotoolz`), which never import this package, and the domain packages supply a `Task` and its estimators.

## Not built yet

The build order is §7 of the [benchmark ladder design](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/benchmark_ladder.md); step 1 has landed, with the level × rung × complexity report of step 6.

- **Probabilistic scorers**: simulation-based calibration ranks, coverage, CRPS.
- **Posterior-distance scorers** to the oracle: C2ST, MMD, sliced Wasserstein.
- **Physical-constraint scorers** (mass and flux budgets, balance laws); these belong with the domain operators.
- **A pluggable oracle.** The `Oracle` protocol exists but `Benchmark` does not consume it yet; it takes `reference=` or the richest rung-2 estimator.
- **Per-cell runs.** `Benchmark` does not write a `pipekit_experiment.Run` per cell yet.
- **The pilot tasks** in the design docs; a domain joins by shipping its own `Task`, outside this package.

## Learn more

- The design: [the benchmark ladder](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/benchmark_ladder.md), the domain-independent framework, and the [decisions](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/decisions.md).
- The four pilots, each working the ladder for one domain: [plume](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/plume_pilot.md), [ocean](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/ocean_pilot.md), [land extremes](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/land_extremes_pilot.md) and [fusion](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-evaluate/docs/design/fusion_pilot.md).
- Report 14 of the GeoStack: [pipekit-evaluate](https://github.com/jejjohnson/geostack/blob/main/docs/reports/14_pipekit_evaluate.md), workstream A.
- The foundations: [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) and [pipekit-array](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array).
- Siblings: [pipekit-cycle](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) (the simulator and observation protocols) and [pipekit-experiment](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment) (registry and run records).
