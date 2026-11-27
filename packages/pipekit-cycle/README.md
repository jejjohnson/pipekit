<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-cycle/logo-dark.svg">
  <img alt="pipekit-cycle" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-cycle/logo-light.svg" width="389">
</picture>

**Time stepping and data-assimilation cycles as pipeline operators for any array library: forecast, observe and analyse with any forward model, observation operator and analysis, on pipekit-ops and, through an extra, pipekit-array.**

[![Tests](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pipekit/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pipekit/)
[![codecov](https://codecov.io/gh/jejjohnson/pipekit/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pipekit)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-cycle/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

📖 [Documentation](https://jejjohnson.github.io/pipekit/) ·
🚀 [Quickstart](https://jejjohnson.github.io/pipekit/quickstart-cycle/) ·
📚 [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-cycle/) ·
🧭 [Design](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle/docs/design)

</div>

---

Consider assimilating along-track sea surface height from an altimeter such as SWOT into an ocean model. Every few hours we run the model forward, compare the forecast with the passes that arrived, and pull the state toward them, which is the forecast, observe and analyse loop that every assimilation system repeats. The science lives in three pieces: the model, the observation operator and the analysis (an EnKF, a 3D-Var, a particle filter). However, most of the code in a real system is the loop around them: the clock, the cycle counter, which observations belong to which step, and resuming a run where it stopped. So pipekit-cycle owns the loop and nothing else, and its central object is one cycle,

```math
\mathbf{x}_{t \mid t} = a(f(\mathbf{x}_{t-1 \mid t-1}), \mathbf{y}_t; h, \boldsymbol{\Sigma}_\varepsilon)
```

where $`\mathbf{x}_{t \mid t} \in \mathbb{R}^{D_x}`$ is the analysis at cycle $`t`$, $`f: \mathbb{R}^{D_x} \to \mathbb{R}^{D_x}`$ is the forward model, so that $`f(\mathbf{x}_{t-1 \mid t-1}) = \mathbf{x}_{t \mid t-1}`$ is the forecast, $`\mathbf{y}_t \in \mathbb{R}^{D_y}`$ holds the observations of cycle $`t`$, $`h: \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}`$ is the observation operator, $`\boldsymbol{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}`$ is the observation-error covariance, and $`a`$ is the analysis, which takes the forecast and the observations and returns the next state; $`h`$ and $`\boldsymbol{\Sigma}_\varepsilon`$ stay fixed while it runs. $`f`$, $`h`$ and $`a`$ are plain objects that satisfy three runtime-checkable protocols (`ForwardModel`, `ObservationOperator` and `AnalysisStep`), so the same cycle runs a toy model in a notebook and an ocean model in production, with an analysis from any library.

```python
# Shapes: D_x state elements; D_y observations; N_ℰ members; T cycles
# (advect, obs_operator, obs_err_cov and the rest as in the quickstart below)

# today (pipekit-cycle)
# f, a ForwardModel made from a plain function
forward = pc.CallableForward(advect, dt=1.0)
# today (pipekit-cycle)
# the cycle checks f, h and a against their protocols when it is built
cycle = pc.EnsembleDACycle(
    forward, obs_operator, analysis, obs_source, n_steps=10, n_members=20
)
# today (pipekit-cycle)
# the clock at t = 0, carrying Σ_ε to the analysis
start = pc.DAState(obs_err_cov=obs_err_cov)
# today (pipekit-cycle)
# x⁽ᵉ⁾_t∣t = a(f(x⁽ᵉ⁾_t−1∣t−1), y_t; h, Σ_ε), for t = 1, …, T
members, state = cycle(prior, start)
```

None of it is a new framework. Every cycle is a [`pipekit-ops`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) `StatefulOperator` that takes and returns `(carrier, state)`, with a `CarryState` such as `DAState` holding the clock, so a cycle chains with `|` into a `Sequential` and sits inside a `Graph` like any other step. The observations come from any pipekit-ops `Operator` called with the step index, so a fetch, quality-control and bias-correction chain is one `Sequential`. pipekit-cycle imports nothing else, and its `array` extra installs [`pipekit-array`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array) beside it for a forward model or an observation operator built from that package's Array API operators.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_cycle_overview/pipekit_cycle_overview.png" alt="pipekit-cycle in one picture: you supply a forward model f, an observation operator h, an analysis a and the observations; one cycle x_t∣t = a(f(x_t−1∣t−1), y_t; h, Σ_ε) repeated on a clock; four areas (protocols, cycles, components, assimilation); out comes the analysis, its carry state, the history and a step of a pipeline; on pipekit-ops, with the analyses from filterax and vardax" width="100%"></p>

## 🌟 Highlights

- 🔁 **Eight cycles, one contract**: `Cycle`, `EnsembleCycle`, `WindowedCycle` and `Recurrence` step a model, and `DACycle`, `EnsembleDACycle`, `SmootherCycle` and `BFNCycle` assimilate; each takes `(carrier, state)` and returns `(carrier, state)`.
- 🧩 **Seams by shape**: 14 runtime-checkable protocols, the core three plus the reduced-order, ensemble, iterative-inversion and adjoint seams, and the cycles check their components against them with `isinstance` when they are built, so an adapter that never imports pipekit-cycle (`filterax.pipekit`, `plumax.adapters`) plugs in.
- 🛰️ **Observations on a clock**: `obs_source` and `obs_err_source` are operators called with the step index, and `obs_clock="state"` continues that index from `DAState.cycle_count`, so two calls of five cycles give the same analysis as one call of ten.
- 🧱 **Ready-made components**: `IdentityObs`, `LinearObs`, `CallableObs` and `CompositeObs`, whose `linearize` composes the tangent linears by the chain rule, and `CallableForward`, `CompositeForward` and `NeuralForward` for a function, a chain or a trained emulator.
- ↔️ **Assimilation without an adjoint**: `BFNCycle` nudges forth and back over a window until the back-pass initial states agree (Auroux and Blum, 2008).
- 🧮 **Gradients as configuration**: five frozen adjoint specs, from `DirectAdjoint` to `TruncatedAdjoint`, with no behaviour of their own, which integrix, argminax, filterax and vardax translate into their own adjoints.
- 🪶 **Pure Python**: the only dependency is pipekit-ops, which has none, and `LinearObs` needs only `@`, so the same cycle steps NumPy or JAX arrays.

## 📐 Eight cycles over three seams

Each cycle repeats one step, and the table says what that step is and which seams it calls. The ensemble cycles carry a list of $`N_{\mathcal{E}}`$ members $`\mathbf{x}^{(e)}`$, $`e \in \mathcal{E}`$, and $`W`$ is a window of forward steps.

| Cycle | One step | Seams it calls | Area |
|---|---|---|---|
| `Cycle` | $`\mathbf{x}_t = f(\mathbf{x}_{t-1})`$ | `ForwardModel` or a stateful step | `cycles` |
| `EnsembleCycle` | the same for each member | `ForwardModel` or a stateful step | `cycles` |
| `WindowedCycle` | $`W`$ steps from each start of a stream | `ForwardModel` or a stateful step | `cycles` |
| `Recurrence` | one step, until a condition holds | `ForwardModel` or a stateful step | `cycles` |
| `DACycle` | $`\mathbf{x}_{t \mid t} = a(f(\mathbf{x}_{t-1 \mid t-1}), \mathbf{y}_t; h, \boldsymbol{\Sigma}_\varepsilon)`$ | all three | `assimilation` |
| `EnsembleDACycle` | $`f`$ per member, one $`a`$ for the ensemble | all three | `assimilation` |
| `SmootherCycle` | $`W`$ forecasts, one $`a`$ over the window | all three | `assimilation` |
| `BFNCycle` | forth and back over $`W`$ steps, nudging | `ForwardModel`, `ObservationOperator`, a gain | `assimilation` |

The three seams stand for the two halves of the state-space model, with the analysis $`a`$ as what estimates the state from both,

```math
\mathbf{x}_t = f(\mathbf{x}_{t-1}) + \boldsymbol{\eta}_t
```

```math
\mathbf{y}_t = h(\mathbf{x}_t) + \boldsymbol{\varepsilon}_t
```

where $`\boldsymbol{\eta}_t \in \mathbb{R}^{D_x}`$ is the model noise and $`\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\varepsilon)`$ the observation noise. `ForwardModel` asks for `step(state, dt)`, `dt` and `state_signature`, `ObservationOperator` for `__call__(state)` and `linearize(state)`, and `AnalysisStep` for one call, `(forecast, obs, *, obs_op, obs_err_cov) -> analysis`. $`\boldsymbol{\Sigma}_\varepsilon`$ travels on `DAState.obs_err_cov` and reaches the analysis as `obs_err_cov`. pipekit-cycle ships no analysis: the quickstart writes an EnKF of its own, and filterax and vardax ship real ones.

`BFNCycle` is the one row with no analysis step. On each pass it relaxes the state toward the observations,

```math
\mathbf{x} \leftarrow \mathbf{x} + \mathbf{K}(\mathbf{y}_t - h(\mathbf{x}))
```

with the nudging gain $`\mathbf{K}`$ given as a function of the innovation, after each forward step and before each backward one, and it repeats the sweep until the initial state of the backward pass stops changing (Auroux and Blum, 2008).

## 📥 Install

pipekit-cycle is not on PyPI, and neither is pipekit-ops, its one dependency, so install both from GitHub. pipekit-ops has no dependencies of its own, so the base install is pure Python; it needs Python 3.12 or newer.

```bash
pip install \
  "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops" \
  "pipekit-cycle @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-cycle"
```

The one extra, `array`, adds pipekit-array, for a forward model or an observation operator made of its Array API operators; pipekit-cycle itself imports nothing from it. The quickstart below also needs NumPy, einx and jaxtyping, which are not dependencies of the package.

```bash
pip install \
  "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops" \
  "pipekit-array @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-array" \
  "pipekit-cycle[array] @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-cycle"
```

A project that depends on pipekit-cycle pins a release tag (the [changelog](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-cycle/CHANGELOG.md) lists them) and gives every pipekit package the same one. The packages are members of one uv workspace that point at each other at the same commit, so uv needs pipekit-ops from the commit pipekit-cycle came from, which is what this block in the project's `pyproject.toml` says:

```toml
[project]
dependencies = ["pipekit-ops", "pipekit-cycle"]

[tool.uv.sources]
pipekit-ops = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-ops", tag = "pipekit-array-v0.3.1" }
pipekit-cycle = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-cycle", tag = "pipekit-array-v0.3.1" }
```

uv applies `[tool.uv.sources]` only to a project's direct dependencies, so every pipekit package named there is listed in `dependencies` too.

## 🌊 Quickstart: an ensemble Kalman filter on a periodic channel

The state is a sea-surface-height anomaly on $`D_x = 8`$ grid cells of a periodic channel, and each cycle the forward model moves it one cell east and damps it, $`f(\mathbf{x}) = \gamma \mathbf{S} \mathbf{x}`$, with the cyclic shift $`\mathbf{S} \in \mathbb{R}^{D_x \times D_x}`$ and $`\gamma = 0.98`$. We observe every other cell, $`h(\mathbf{x}) = \mathbf{H} \mathbf{x}`$ with the selection matrix $`\mathbf{H} \in \mathbb{R}^{D_y \times D_x}`$ and $`D_y = 4`$, with noise $`\boldsymbol{\Sigma}_\varepsilon = \sigma_\varepsilon^2 \mathbf{I}`$ and $`\sigma_\varepsilon = 0.1`$, for $`T = 10`$ cycles of a truth run that starts from one sine wave. The analysis $`a`$ is a perturbed-observation EnKF over $`N_{\mathcal{E}} = 20`$ members drawn from $`\mathcal{N}(\mathbf{0}, \mathbf{I})`$. It forms the anomalies $`\tilde{\mathbf{x}}^{(e)} = \mathbf{x}^{(e)} - \bar{\mathbf{x}}`$ and $`\tilde{\mathbf{y}}^{(e)} = h(\mathbf{x}^{(e)}) - \bar{\mathbf{y}}`$ about the ensemble means $`\bar{\mathbf{x}}`$ and $`\bar{\mathbf{y}}`$, from them the cross-covariance $`\boldsymbol{\Sigma}_{xy} \in \mathbb{R}^{D_x \times D_y}`$ and the covariance $`\boldsymbol{\Sigma}_y \in \mathbb{R}^{D_y \times D_y}`$ of the predicted observations plus $`\boldsymbol{\Sigma}_\varepsilon`$, and moves each member by the gain $`\mathbf{G} = \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_y^{-1} \in \mathbb{R}^{D_x \times D_y}`$ times its innovation $`\mathbf{r}^{(e)} = \mathbf{y}^{(e)} - h(\mathbf{x}^{(e)})`$, where $`\mathbf{y}^{(e)} = \mathbf{y}_t + \boldsymbol{\varepsilon}^{(e)}`$ is a perturbed copy of the observations. The model and the analysis are ours (`# user`); the loop is the package's.

```python
import einx
import numpy as np
import pipekit_cycle as pc
from jaxtyping import Float
from pipekit_ops import Lambda


# Shapes: D_x = 8 grid cells; D_y = 4 observed cells; N_ℰ = 20 members; T = 10 cycles
State = Float[np.ndarray, " state"]
Obs = Float[np.ndarray, " obs"]
Covariance = Float[np.ndarray, "row col"]
state_size, obs_size, num_members, num_cycles = 8, 4, 20, 10
rng = np.random.default_rng(0)


# user
# f(x) = γ S x: one cell east on the periodic channel, damped by γ = 0.98
def advect(state: State, dt: float) -> State:
    return 0.98 * np.roll(state, 1)


# H, which keeps every other cell
# → (D_y, D_x)
obs_matrix = np.eye(state_size)[::2]
# today (pipekit-cycle)
# h(x) = H x
obs_operator = pc.LinearObs(obs_matrix)
# Σ_ε = σ_ε² I, with σ_ε = 0.1
# → (D_y, D_y)
obs_err_cov = 0.01 * np.eye(obs_size)

# x_0,i = sin(2π i / D_x), for the cells i = 0, …, D_x − 1
# → (D_x,)
truth = [np.sin(2 * np.pi * np.arange(state_size) / state_size)]
# x_t = f(x_t−1), for t = 1, …, T
for _ in range(num_cycles):
    truth.append(advect(truth[-1], 1.0))
# ε_t ~ 𝒩(0, Σ_ε)
# → (T, D_y)
noise = rng.multivariate_normal(np.zeros(obs_size), obs_err_cov, size=num_cycles)
# y_t = h(x_t) + ε_t
# → T × (D_y,)
observations = [obs_operator(x) + e for x, e in zip(truth[1:], noise, strict=True)]


# user
class StochasticEnKF:
    """Perturbed-observation EnKF; an `AnalysisStep` by its shape alone."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        forecast: list[State],
        obs: Obs,
        *,
        obs_op: pc.ObservationOperator,
        obs_err_cov: Covariance,
    ) -> list[State]:
        # N_ℰ
        ensemble_size = len(forecast)
        # x⁽ᵉ⁾_t∣t−1 and h(x⁽ᵉ⁾_t∣t−1), stacked over the members
        # → (N_ℰ, D_x), (N_ℰ, D_y)
        states = np.stack(forecast)
        predicted = np.stack([obs_op(x) for x in forecast])
        # x̄ and ȳ, the ensemble means
        # → (D_x,), (D_y,)
        state_mean = einx.mean("member state -> state", states)
        predicted_mean = einx.mean("member obs -> obs", predicted)
        # x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄ and ỹ⁽ᵉ⁾ = h(x⁽ᵉ⁾) − ȳ
        # → (N_ℰ, D_x), (N_ℰ, D_y)
        state_anom = einx.subtract("member state, state", states, state_mean)
        obs_anom = einx.subtract("member obs, obs", predicted, predicted_mean)
        # Σ_xyᵀ = ∑ₑ ỹ⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ / (N_ℰ − 1)
        # → (D_y, D_x)
        cross_products = einx.dot(
            "member obs, member state -> obs state", obs_anom, state_anom
        )
        cross_cov_transposed = cross_products / (ensemble_size - 1)
        # Σ_y = ∑ₑ ỹ⁽ᵉ⁾ ỹ⁽ᵉ⁾ᵀ / (N_ℰ − 1) + Σ_ε
        # → (D_y, D_y)
        obs_products = einx.dot("member row, member col -> row col", obs_anom, obs_anom)
        obs_covariance = obs_products / (ensemble_size - 1) + obs_err_cov
        # Gᵀ = Σ_y⁻¹ Σ_xyᵀ, by a solve, since Σ_y is symmetric
        # → (D_y, D_x)
        gain_transposed = np.linalg.solve(obs_covariance, cross_cov_transposed)
        # y⁽ᵉ⁾ = y_t + ε⁽ᵉ⁾, with ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε)
        # → (N_ℰ, D_y)
        perturbed = self.rng.multivariate_normal(obs, obs_err_cov, size=ensemble_size)
        # r⁽ᵉ⁾ = y⁽ᵉ⁾ − h(x⁽ᵉ⁾_t∣t−1)
        # → (N_ℰ, D_y)
        innovation = perturbed - predicted
        # x⁽ᵉ⁾_t∣t = x⁽ᵉ⁾_t∣t−1 + G r⁽ᵉ⁾
        # → (N_ℰ, D_x)
        increment = einx.dot(
            "obs state, member obs -> member state", gain_transposed, innovation
        )
        return list(states + increment)


# user
# RMSE(x̂, x) = √(∑ᵢ (x̂ᵢ − xᵢ)² / D_x)
def rmse(estimate: State, reference: State) -> float:
    error = estimate - reference
    mean_square = einx.mean("state ->", error**2)
    return float(np.sqrt(mean_square))


# today (pipekit-cycle)
# f as a ForwardModel, with Δt = 1
forward = pc.CallableForward(advect, dt=1.0)
# today (pipekit-ops)
# t ↦ y_t, called by the cycle with the step index
obs_source = Lambda(lambda t: observations[t])
# user
# a, the perturbed-observation EnKF
analysis = StochasticEnKF(seed=1)
# today (pipekit-cycle)
# x⁽ᵉ⁾_t∣t = a(f(x⁽ᵉ⁾_t−1∣t−1), y_t; h, Σ_ε), for t = 1, …, T
cycle = pc.EnsembleDACycle(
    forward_model=forward,
    obs_op=obs_operator,
    analysis_step=analysis,
    obs_source=obs_source,
    n_steps=num_cycles,
    n_members=num_members,
)

# x⁽ᵉ⁾_0∣0 ~ 𝒩(0, I), for e = 1, …, N_ℰ
# → N_ℰ × (D_x,)
prior = [rng.normal(0.0, 1.0, state_size) for _ in range(num_members)]
# today (pipekit-cycle)
# the clock at t = 0, with Σ_ε for the analysis
start = pc.DAState(obs_err_cov=obs_err_cov)
# today (pipekit-cycle)
# T cycles: the analysis ensemble x⁽ᵉ⁾_T∣T and the advanced clock
members, state = cycle(prior, start)

# x̄_0∣0 and x̄_T∣T, the ensemble means before and after
# → (D_x,), (D_x,)
prior_mean = einx.mean("member state -> state", np.stack(prior))
analysis_mean = einx.mean("member state -> state", np.stack(members))
prior_error = rmse(prior_mean, truth[0])
analysis_error = rmse(analysis_mean, truth[-1])
print(state.cycle_count, state.t, f"{prior_error:.2f}", f"{analysis_error:.2f}")
```

This prints `10 10.0 0.77 0.02`: ten cycles done with the clock at $`t = 10`$, and the ensemble mean, 0.77 (RMSE) from the truth before the first cycle, ends 0.02 from it after the tenth. `StochasticEnKF` never imports pipekit-cycle, yet the cycle accepted it as an `AnalysisStep`, because the protocols are structural. So swapping `advect` for the step of an ocean model and `StochasticEnKF` for filterax's `FilterAnalysisStep` or a vardax method's `.as_analysis_step()` leaves the cycle as it is.

The [quickstart](https://jejjohnson.github.io/pipekit/quickstart-cycle/) takes this cycle further: a forecast with no observations, the protocol checks, resuming a run on the state's clock, the calling convention, the components, back-and-forth nudging and the adjoint specs.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_cycle_roles/pipekit_cycle_roles-dark.svg">
    <img alt="pipekit-cycle on the GeoStack role ladder: the eight cycles as recipes, the observation operators and forward models as steps, the analysis left to filterax and vardax, the protocols as the contracts of a model, the adjoint specs as strategies, and the carry states as structures" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_cycle_roles/pipekit_cycle_roles-light.svg" width="100%">
  </picture>
</p>

Each public name sits on one rung of the GeoStack role ladder, by the job it does. The eight cycles are ⑥ recipes, the observation operators and forward models are ⑤ steps, the protocols are the contracts a ③ model satisfies, the adjoint specs are ② strategies, and the carry states are ⓪ structures; the ④ algorithm, the analysis itself, is left to the libraries that implement the protocols. The package has four areas, and `pipekit_cycle` re-exports every name below.

| Area | Symbols | Notes |
|---|---|---|
| `pipekit_cycle.protocols` | `ForwardModel`, `ObservationOperator`, `AnalysisStep`, `TangentLinearModel`, `ReducedBasis`, `ErrorSubspace`, `ReducedOrderModel`, `LatentMap`, `ObservationNoise`, `CovarianceLocalizer`, `EnsembleInflator`, `IterativeProcess`, `StepScheduler`, `SupportsAdjoint` | The runtime-checkable seams |
| `pipekit_cycle.cycles` | `Cycle`, `EnsembleCycle`, `WindowedCycle`, `Recurrence`, `DAState`, `IterationState`, `WindowState`, `AdjointSpec`, `DirectAdjoint`, `RecursiveCheckpointAdjoint`, `TruncatedAdjoint`, `ImplicitAdjoint`, `BacksolveAdjoint` | Time stepping, its carry states and the adjoint specs |
| `pipekit_cycle.components` | `IdentityObs`, `LinearObs`, `CallableObs`, `CompositeObs`, `CallableForward`, `CompositeForward`, `NeuralForward` | Observation operators and forward models |
| `pipekit_cycle.assimilation` | `DACycle`, `EnsembleDACycle`, `SmootherCycle`, `BFNCycle` | Forecast, observe, analyse |

**The protocols** are the core three, then the reduced-order seams (`TangentLinearModel` to `LatentMap`), the ensemble seams (`ObservationNoise`, `CovarianceLocalizer`, `EnsembleInflator`), the iterative-inversion seams (`IterativeProcess`, `StepScheduler`) and `SupportsAdjoint`; the [protocols page](https://jejjohnson.github.io/pipekit/protocols/) gives the members each asks for.

**The areas** import only from the ones before them, in the order of the table ([decision C1](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-cycle/docs/design/decisions.md)). Every operator is recorded under `pipekit_cycle` whatever file it lives in, so a pipeline saved under the flat modules of earlier releases (`pipekit_cycle.da`, `pipekit_cycle.cycle`, …) still loads.

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [Run a DA cycle](https://jejjohnson.github.io/pipekit/da-cycle/) | a forecast-only `Cycle`, then an `EnsembleDACycle` with an analysis from another library |
| 2 | [Ensemble data assimilation](https://jejjohnson.github.io/pipekit/ensemble-da/) | an EnKF on Lorenz-96 for 200 cycles, with observation pre-processing, quality control and verification |
| 3 | [Pipelines under jit](https://jejjohnson.github.io/pipekit/pipekit-under-jit/) | pipekit-jax's `ScanCycle` in place of `Cycle` under `jax.jit` |
| 4 | [Cycle handoff](https://jejjohnson.github.io/pipekit/pipekit-train-cycle-handoff/) | an emulator trained with pipekit-train, then forecasting in a `Cycle` |

## 📖 Documentation

- **[Quickstart](https://jejjohnson.github.io/pipekit/quickstart-cycle/)**: the calling convention, the protocol checks, resuming a run, the components, back-and-forth nudging and the adjoint specs.
- **[Protocols](https://jejjohnson.github.io/pipekit/protocols/)**: every seam of `pipekit_cycle.protocols`, with the members it asks for and the libraries that satisfy it.
- **[API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-cycle/)**: every public name, generated from the docstrings.
- **[Design](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle/docs/design)**: the decision behind the four areas, and the draft design of latent data assimilation; the package's original plan is [Report 10](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_8_pipekit_cycle.md) of the master plan.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_cycle_ecosystem/pipekit_cycle_ecosystem-dark.svg">
    <img alt="The packages that import pipekit-cycle on their main branches, by layer, each with the areas it imports: pipekit-jax, vardax, sesmax-experiment and somax; pipekit-cycle builds on pipekit-ops alone" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_cycle_ecosystem/pipekit_cycle_ecosystem-light.svg" width="100%">
  </picture>
</p>

pipekit-cycle is a package of the pipekit [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), in the Core tier of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the loop, and composes the rest in:

| Concern | Owner |
|---|---|
| `Operator`, `StatefulOperator`, `CarryState`, `Sequential` and `Graph` | [`pipekit-ops`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) |
| Array API operators for a model or an observation operator (the `array` extra) | [`pipekit-array`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array) |
| Cycles under `jax.jit` (`ScanCycle`, `EnsembleScanCycle`, `Rollout`) | [`pipekit-jax`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax) |
| Ensemble and variational analyses | [`filterax`](https://github.com/jejjohnson/filterax), [`vardax`](https://github.com/jejjohnson/vardax) |
| Reading an adjoint spec (`to_diffrax_adjoint`, `to_optimistix_adjoint`) | [`integrix`](https://github.com/jejjohnson/integrix), [`argminax`](https://github.com/jejjohnson/argminax) |

Dependencies point one way: pipekit-cycle imports only pipekit-ops. On `main`, vardax imports it in its source (the core protocols, `DACycle` and `SmootherCycle`, and its own subclasses of `LinearObs` and `CallableObs`), and sesmax-experiment and somax run their models in chunks with `Cycle`. In the workspace, pipekit-jax takes `ForwardModel` for its scan cycles, and pipekit-evaluate's `cycle` extra takes `ForwardModel` and `ObservationOperator` for its benchmark tasks. filterax, plumax and xtremax satisfy the protocols by shape and import pipekit-cycle only in their tests, and integrix and argminax read the adjoint specs by class name without importing it.

## 🌳 Repository layout

```
packages/pipekit-cycle/
├── src/pipekit_cycle/
│   ├── __init__.py      ← every public name, recorded under pipekit_cycle
│   ├── protocols/       ← the runtime-checkable seams
│   ├── cycles/          ← Cycle, EnsembleCycle, WindowedCycle, Recurrence, the states, the adjoint specs
│   ├── components/      ← observation operators and forward models
│   └── assimilation/    ← DACycle, EnsembleDACycle, SmootherCycle, BFNCycle
├── tests/               ← each cycle and component, the protocols, the area import rule, config round trips
└── docs/design/         ← decisions.md (the four areas) and latent.md (latent DA, a draft)
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install                                        # every dependency group and the pre-commit hooks
uv run pytest -o addopts= packages/pipekit-cycle/tests   # this package's tests
make lint                                           # ruff check on the whole repo
make typecheck                                      # ty check on every package
make docs                                           # the MyST prose and the API reference, into public/
```

`make help` lists every task. The root [`AGENTS.md`](https://github.com/jejjohnson/pipekit/blob/main/AGENTS.md) holds the rules for contributors and coding agents.

## 🎓 References

- Auroux and Blum (2008). [A nudging-based data assimilation method: the Back and Forth Nudging (BFN) algorithm](https://doi.org/10.5194/npg-15-305-2008). *Nonlinear Processes in Geophysics*.
- Bolte, Pauwels and Vaiter (2023). [One-step differentiation of iterative algorithms](https://arxiv.org/abs/2305.13768). *NeurIPS*.
- Griewank (1992). [Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation](https://doi.org/10.1080/10556789208805505). *Optimization Methods and Software*.

## 📄 License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-cycle/LICENSE).
