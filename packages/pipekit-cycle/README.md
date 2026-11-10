<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-cycle/logo-dark.svg">
    <img alt="pipekit-cycle" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-cycle/logo-light.svg" width="389">
  </picture>
</p>

**Time stepping and data-assimilation cycles as pipekit operators, with the protocols that let any forward model, observation operator and analysis plug in.**

<p align="center">
  <a href="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Version 0.1.0" src="https://img.shields.io/badge/version-0.1.0-blue">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="https://jejjohnson.github.io/pipekit/"><b>Docs</b></a> ·
  <a href="https://jejjohnson.github.io/pipekit/api/pipekit-cycle/"><b>API</b></a> ·
  <a href="https://jejjohnson.github.io/pipekit/tutorials/da-cycle/"><b>DA tutorial</b></a> ·
  <a href="https://jejjohnson.github.io/pipekit/tutorials/ensemble-da/"><b>Ensemble tutorial</b></a>
</p>

## Why it exists

Consider assimilating along-track sea surface height from an altimeter such as SWOT into an ocean model.
Every few hours we run the model forward, compare the forecast with the passes that arrived, and pull the state toward them, which is the forecast, observe and analyse loop that every assimilation system repeats.
The science lives in three pieces: the model, the observation operator and the analysis (an EnKF, a 3D-Var, a particle filter).
However, most of the code in a real system is the loop around them: the clock, the cycle counter, which observations belong to which step, the history, and resuming a run where it stopped.

We write the state-space model as

```math
\mathbf{x}_t = f(\mathbf{x}_{t-1}) + \boldsymbol{\eta}_t
```

```math
\mathbf{y}_t = h(\mathbf{x}_t) + \boldsymbol{\varepsilon}_t
```

```math
\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\varepsilon)
```

where $\mathbf{x}_t \in \mathbb{R}^{D_x}$ is the state at cycle $t$, $\mathbf{y}_t \in \mathbb{R}^{D_y}$ the observations, $f$ the forward model, $h$ the observation operator (the matrix $\mathbf{H} \in \mathbb{R}^{D_y \times D_x}$ when it is linear), $\boldsymbol{\eta}_t \in \mathbb{R}^{D_x}$ the model noise and $\boldsymbol{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$ the observation-error covariance.
pipekit-cycle owns the loop and nothing else.
$f$, $h$ and the analysis are plain objects that satisfy three runtime-checkable protocols, so the same cycle runs a toy model in a notebook and an ocean model in production, with an analysis from any library.

## Where it sits

pipekit-cycle is built on pipekit-ops; it uses pipekit-array through the `[array]` extra.
Both are libraries in their own right, which you can build on directly.

```
   your DA system (filterax, vardax, plumax, ...)
                     │
   pipekit-jax ──────┤  (ScanCycle, DiffraxForwardModel)
   pipekit-train ────┤  ([cycle] extra: SimulationDataset)
   pipekit-evaluate ─┤  ([cycle] extra: benchmark Task)
                     │
          ┌──────────┴──────────┐
          │    pipekit-cycle    │  ◀ this package
          │  cycles, DA cycles, │
          │      protocols      │
          └──────────┬──────────┘
            ┌────────┴────────┐
            │                 │
       pipekit-ops       pipekit-array
       (dependency:      ([array] extra:
        Operator,         Array API ops,
        StatefulOperator, named dims)
        CarryState)
```

- **pipekit-ops** is the one required dependency.
  Every cycle is a `pipekit_ops.StatefulOperator` carrying a `pipekit_ops.CarryState`, so a cycle composes with `|` into a `Sequential`, sits inside a `Graph`, and wraps in a `Retry` or a `Tap` like any other operator.
- **pipekit-array** is optional.
  pipekit-cycle imports nothing from it; the `[array]` extra installs it beside, for when your forward model or observation operator is built from pipekit-array operators.

## Install

```bash
uv add pipekit-cycle
uv add 'pipekit-cycle[array]'    # plus pipekit-array
```

From a checkout of the workspace, `uv sync` installs every package.

| Extra | What it adds | When you need it |
| --- | --- | --- |
| *(none)* | `pipekit-ops` | always; the core install is pure Python |
| `array` | `pipekit-array` | your forward model or observation operator is made of pipekit-array operators |

## Quickstart

The blocks use NumPy, and the first one jaxtyping for its shape types; neither is a dependency of the package.

### Forecast only

A forecast is a `Cycle` around a forward model.
Here the model is one explicit Euler step of exponential decay, $\mathrm{d}x / \mathrm{d}t = -k x$ with $k = 0.1$.

```python
import numpy as np
from jaxtyping import Float

import pipekit_cycle as pc

# Shapes: N state variables
State = Float[np.ndarray, " N"]


# user
def decay(x: State, dt: float) -> State:
    """One explicit Euler step of dx/dt = −k x, with k = 0.1."""
    # x_{t+1} = x_t − Δt · k · x_t
    return x - dt * 0.1 * x


# today (pipekit-cycle)
# x_t ↦ x_{t+1}, Δt = 1
step = pc.CallableForward(decay, dt=1.0)
# today (pipekit-cycle)
# x_0 ↦ x_24
forecast = pc.Cycle(step_op=step, n_steps=24)

# x₂₄ = (1 − 0.1)²⁴ x₀ ≈ 0.080 x₀
x24, _ = forecast(np.ones(3), None)                   # (N,) → (N,)
```

Swap `CallableForward` for `NeuralForward` around a trained model from the registry and the same `Cycle` forecasts with an emulator.
Add an observation operator and an analysis step and it assimilates (`DACycle`, `EnsembleDACycle`), which is the next block.

### Forecast, observe, analyse

A toy ensemble DA cycle.
The state is a sea-surface-height anomaly on $D_x = 8$ grid cells of a periodic channel; each cycle the forward model moves it one cell east and damps it, $f(\mathbf{x}) = a \mathbf{S} \mathbf{x}$ with the shift $\mathbf{S}$ and $a = 0.98$.
We observe every other cell ($D_y = 4$) with noise $\boldsymbol{\Sigma}_\varepsilon = 0.01 \mathbf{I}$, and assimilate with a perturbed-observation EnKF of $N_{\mathcal{E}} = 20$ members $\mathbf{x}^{(e)}$.
The analysis uses the gain $\mathbf{G}_t = \boldsymbol{\Sigma}_{xy,t} \boldsymbol{\Sigma}_{y,t}^{-1} \in \mathbb{R}^{D_x \times D_y}$, where $\boldsymbol{\Sigma}_{xy,t}$ is the state-observation cross-covariance of the forecast ensemble and $\boldsymbol{\Sigma}_{y,t}$ the covariance of the predicted observations plus $\boldsymbol{\Sigma}_\varepsilon$.
The model and the analysis are ours (`# user`); the loop is the package's.

```python
import numpy as np
import pipekit_cycle as pc
from pipekit_ops import Lambda

# Shapes: D_x = 8 grid cells; D_y = 4 observed cells; N_E = 20 members; T = 10 cycles
n_cells, n_members, n_cycles = 8, 20, 10
rng = np.random.default_rng(0)


# user
# f(x) = a S x: shift one cell east (periodic), damp by a = 0.98
def advect(x, dt):
    return 0.98 * np.roll(x, 1)                              # (D_x,) → (D_x,)


# user
# H picks every other cell
H = np.eye(n_cells)[::2]                                     # (D_y, D_x)
# Σ_ε = σ_ε² I, σ_ε = 0.1
obs_err_cov = 0.01 * np.eye(4)                               # (D_y, D_y)

# user
# a truth run x_{0:T} and its observations y_t = H x_t + ε_t, ε_t ~ 𝒩(0, Σ_ε)
truth = [np.sin(2 * np.pi * np.arange(n_cells) / n_cells)]  # (D_x,)
for _ in range(n_cycles):
    truth.append(advect(truth[-1], 1.0))
noise = rng.multivariate_normal(np.zeros(4), obs_err_cov, size=n_cycles)  # (T, D_y)
observations = [H @ x + e for x, e in zip(truth[1:], noise)]  # T × (D_y,)


class StochasticEnKF:  # user
    """Perturbed-observation EnKF; satisfies pc.AnalysisStep structurally."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
        X = np.stack(forecast)                               # (N_E, D_x)
        Y = np.stack([obs_op(x) for x in forecast])          # (N_E, D_y)
        n = len(forecast)
        # x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄,  ỹ⁽ᵉ⁾ = h(x⁽ᵉ⁾) − ȳ
        X_anom = X - np.einsum("ed->d", X) / n               # (N_E, D_x)
        Y_anom = Y - np.einsum("ed->d", Y) / n               # (N_E, D_y)
        # Σ_xy = ∑ₑ x̃⁽ᵉ⁾ ỹ⁽ᵉ⁾ᵀ / (N_E − 1)
        cov_xy = np.einsum("ei,ej->ij", X_anom, Y_anom) / (n - 1)  # (D_x, D_y)
        # Σ_y = ∑ₑ ỹ⁽ᵉ⁾ ỹ⁽ᵉ⁾ᵀ / (N_E − 1) + Σ_ε
        cov_y = np.einsum("ei,ej->ij", Y_anom, Y_anom) / (n - 1) + obs_err_cov
        # G_t = Σ_xy Σ_y⁻¹  (Σ_y is symmetric, so solve Σ_y Gᵀ = Σ_xyᵀ)
        gain = np.linalg.solve(cov_y, cov_xy.T).T            # (D_x, D_y)
        # y⁽ᵉ⁾ = y + ε⁽ᵉ⁾,  ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε)
        obs_draws = self.rng.multivariate_normal(obs, obs_err_cov, size=n)  # (N_E, D_y)
        # x⁽ᵉ⁾_{t|t} = x⁽ᵉ⁾_{t|t−1} + G_t (y⁽ᵉ⁾ − h(x⁽ᵉ⁾_{t|t−1}))
        return list(X + np.einsum("ij,ej->ei", gain, obs_draws - Y))  # N_E × (D_x,)


# today (pipekit-cycle)
da = pc.EnsembleDACycle(
    forward_model=pc.CallableForward(advect, dt=1.0),
    obs_op=pc.LinearObs(H),
    analysis_step=StochasticEnKF(seed=1),
    obs_source=Lambda(lambda t: observations[t]),
    n_steps=n_cycles,
    n_members=n_members,
)
prior = [rng.normal(0.0, 1.0, n_cells) for _ in range(n_members)]  # N_E × (D_x,)
members, state = da(prior, pc.DAState(obs_err_cov=obs_err_cov))

mean = np.einsum("ed->d", np.stack(members)) / n_members    # (D_x,)
print(state.cycle_count, state.t)                            # 10 10.0
print(np.sqrt(np.mean((mean - truth[-1]) ** 2)) < 0.1)       # True
```

The prior ensemble starts about 0.77 (RMSE) from the truth and ends about 0.02 from it after ten cycles.

The analysis never imported pipekit-cycle, yet it is an `AnalysisStep`, because the protocols are structural.
Run the same cycle as two calls of five with `obs_clock="state"` and the second call reads $\mathbf{y}_6, \ldots, \mathbf{y}_{10}$, so the result matches the single run:

```python
# today (pipekit-cycle)
print(isinstance(StochasticEnKF(seed=1), pc.AnalysisStep))  # True


# today (pipekit-cycle)
# two calls of 5 cycles on obs_clock="state" read y_1..y_5, then y_6..y_10
def half_cycle():
    return pc.EnsembleDACycle(
        forward_model=pc.CallableForward(advect, dt=1.0),
        obs_op=pc.LinearObs(H),
        analysis_step=StochasticEnKF(seed=1),
        obs_source=Lambda(lambda t: observations[t]),
        n_steps=5,
        n_members=n_members,
        obs_clock="state",
    )


half = half_cycle()
mid, mid_state = half(prior, pc.DAState(obs_err_cov=obs_err_cov))
resumed, end_state = half(mid, mid_state)
print(end_state.cycle_count)                                    # 10
print(np.allclose(np.stack(resumed), np.stack(members)))        # True
```

## What ships

Version 0.1.0.
Every name below is exported from `pipekit_cycle`.

| Module | Public names | Purpose |
| --- | --- | --- |
| `cycle` | `Cycle`, `EnsembleCycle`, `WindowedCycle`, `Recurrence` | Repeat a step operator: $n$ steps, per member, over sliding windows, or until a condition holds |
| `da` | `DACycle`, `EnsembleDACycle`, `SmootherCycle` | Forecast, observe and analyse for one state, an ensemble, or a 4D-Var-shaped window |
| `bfn` | `BFNCycle` | Back-and-forth nudging (Auroux and Blum, 2008): an assimilation cycle that needs no adjoint |
| `protocols` | `ForwardModel`, `ObservationOperator`, `AnalysisStep` | The core triple: predict, compare, update |
| `protocols` | `ReducedBasis`, `TangentLinearModel`, `ErrorSubspace`, `ReducedOrderModel`, `LatentMap` | Reduced-order and variational seams |
| `protocols` | `ObservationNoise`, `CovarianceLocalizer`, `EnsembleInflator` | Ensemble-method seams |
| `protocols` | `IterativeProcess`, `StepScheduler` | Iterative-inversion seams (EKI, EKS, UKI) |
| `protocols` | `SupportsAdjoint` | A component that says how gradients flow through it |
| `obs` | `IdentityObs`, `LinearObs`, `CallableObs`, `CompositeObs` | Ready-made observation operators |
| `forward` | `CallableForward`, `CompositeForward`, `NeuralForward` | Ready-made forward models: a function, a chain, a trained emulator |
| `state` | `DAState`, `IterationState`, `WindowState` | Carry states for DA cycles, recurrences and windows |
| `adjoints` | `AdjointSpec`, `DirectAdjoint`, `RecursiveCheckpointAdjoint`, `TruncatedAdjoint`, `ImplicitAdjoint`, `BacksolveAdjoint` | Declarative adjoint strategies, interpreted by pipekit-jax or an algorithm library |

### Not built yet

- **Latent data assimilation.** `LatentDACycle`, the `Encoder`, `Decoder` and `LatentForwardModel` protocols, and `LatentDAState` are designed in [`docs/design/latent.md`](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-cycle/docs/design/latent.md) (draft v0.1); only the `LatentMap` protocol ships.
- **A loop for the iterative-inversion seams.** `IterativeProcess` and `StepScheduler` are protocols only; no cycle drives them yet.
- **Localisation and inflation inside the cycles.** `CovarianceLocalizer` and `EnsembleInflator` are protocols only; today an analysis step applies them itself.

## How it works

### The protocols are structural

`ForwardModel` asks for `step(state, dt)`, a `dt` and a `state_signature`.
`ObservationOperator` asks for `__call__(state)` and `linearize(state)`.
`AnalysisStep` is one call, `(forecast, obs, *, obs_op, obs_err_cov) -> analysis`.
All of them are `runtime_checkable`, so `isinstance(obj, pc.ForwardModel)` is true for any class with those members, and the cycles check their components this way when you build them.
Algorithm libraries (filterx, vardax, plumax, ...) ship adapter classes that satisfy the protocols **without importing pipekit-cycle**; the usual place is `<libname>.adapters.pipekit`.

### A cycle is a stateful operator

Every cycle takes `(carrier, state)` and returns `(carrier, state)`.
The carrier is whatever your model steps (an array, an xarray Dataset, a list of members); pipekit-cycle never looks inside it.
The state is a `CarryState` such as `DAState(t, cycle_count, obs_err_cov, extras)`, and the cycles return an advanced copy rather than mutating it, so you can keep earlier states as snapshots.
`Cycle(..., save_history=True)` and `DACycle` record per-step results on `self.history`, which each call overwrites.

### Observations and their clock

`obs_source` is any `pipekit_ops.Operator`, called with a step index, so a whole pre-processing chain (fetch, deduplicate, bias-correct, QC) can be one `Sequential`.
By default the index restarts at 0 on every call (`obs_clock="call"`).
With `obs_clock="state"` it continues from `state.cycle_count`, so calling a cycle again on its returned state reads the next observations and gives the same analysis as one uninterrupted run.
`obs_err_source` works the same way for an observation-error covariance that changes per step.

### The helpers in `obs.py` and `forward.py`

`IdentityObs` and `LinearObs(H)` cover direct and linear measurements; `LinearObs` relies only on `@`, so it works for NumPy and JAX arrays alike.
`CallableObs(fn, linearize_fn=None)` wraps a nonlinear operator such as a radiative transfer model, and `CompositeObs` chains operators and builds the composite tangent linear by the chain rule.
On the model side, `CallableForward(fn, dt)` wraps a function, `CompositeForward` runs components in turn (its `dt` is their sum), and `NeuralForward(model_op, dt)` turns a trained emulator operator into a forward model at a fixed cadence.
`CallableObs`, `LinearObs` and `CallableForward` hold closures or arrays, so they set `forbid_in_yaml = True` and do not round-trip through `dumps` / `loads`; rebuild them in code.

### BFN and the adjoint specs

`BFNCycle` integrates the window forward, then backward, nudging toward the observations on both passes with your gain, and repeats until the back-pass initial states agree within `tol`.
It needs a model that steps with negative `dt` (or a `backward_model`), and it refuses `CompositeForward` and `NeuralForward` without one, because both ignore the sign of `dt`.
The specs in `adjoints` (`DirectAdjoint`, `RecursiveCheckpointAdjoint`, `TruncatedAdjoint`, `ImplicitAdjoint`, `BacksolveAdjoint`) are frozen dataclasses with no behaviour.
A component that satisfies `SupportsAdjoint` carries one, the cycles pass it through untouched, and an execution layer interprets it: `integrix.adjoint.to_diffrax_adjoint` for a diffrax solve, `integrix.adjoint.truncated_scan` for a truncated rollout.
So choosing how gradients flow is configuration, not code.

## Learn more

- API reference: [pipekit-cycle](https://jejjohnson.github.io/pipekit/api/pipekit-cycle/).
- Tutorials: [Run a DA cycle](https://jejjohnson.github.io/pipekit/tutorials/da-cycle/), [Ensemble data assimilation](https://jejjohnson.github.io/pipekit/tutorials/ensemble-da/), [Run pipelines under jit](https://jejjohnson.github.io/pipekit/tutorials/pipekit-under-jit/).
- Design: [Latent data assimilation](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-cycle/docs/design/latent.md); master plan [Report 10](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_8_pipekit_cycle.md).
- The foundations: [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) and [pipekit-array](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array).
- Packages built on this one: [pipekit-jax](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax) (jit-native cycles, diffrax models), [pipekit-train](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train) (emulators trained on simulations), [pipekit-evaluate](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-evaluate) (benchmark tasks).

## License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-cycle/LICENSE).
