# pipekit-cycle quickstart

This page takes the [README](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) further. It covers the install, the cycle as an equation, the calling convention every cycle shares, a forecast with no observations, the README's ensemble Kalman filter in full, how the seams are checked, resuming a run on the state's clock, the ready-made components, back-and-forth nudging, the adjoint specs, saving a cycle, where the package sits in pipekit, and what is not built yet. The code blocks run in order as one script, with NumPy, einx and jaxtyping installed beside the package.

## Install

pipekit-cycle and pipekit-ops, its one dependency, are not on PyPI, so install both from GitHub. pipekit-ops has no dependencies of its own, so the base install is pure Python; it needs Python 3.12 or newer.

```bash
pip install \
  "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops" \
  "pipekit-cycle @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-cycle"
```

| Extra | What it adds | When you need it |
| --- | --- | --- |
| *(none)* | `pipekit-ops` | always; the core install is pure Python |
| `array` | `pipekit-array` | your forward model or observation operator is made of pipekit-array operators |

pipekit-cycle imports nothing from pipekit-array; the extra only installs it beside. A project that depends on pipekit-cycle pins a release tag and gives every pipekit package the same one, because the workspace members point at each other at one commit; the README's [install section](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle#-install) has the `[tool.uv.sources]` block. From a checkout of the workspace, `make install` installs every package.

## The cycle

The package is built around one equation, the cycle itself:

```{math}
:label: eq-cycle
\mathbf{x}_{t \mid t} = a(f(\mathbf{x}_{t-1 \mid t-1}), \mathbf{y}_t; h, \boldsymbol{\Sigma}_\varepsilon)
```

where $\mathbf{x}_{t \mid t} \in \mathbb{R}^{D_x}$ is the analysis at cycle $t$, $f: \mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$ the forward model, $\mathbf{x}_{t \mid t-1} = f(\mathbf{x}_{t-1 \mid t-1})$ the forecast, $\mathbf{y}_t \in \mathbb{R}^{D_y}$ the observations of cycle $t$, $h: \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$ the observation operator (the matrix $\mathbf{H} \in \mathbb{R}^{D_y \times D_x}$ when it is linear), $\boldsymbol{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$ the observation-error covariance, and $a$ the analysis. The three functions are the three seams of the state-space model

```{math}
:label: eq-dynamics
\mathbf{x}_t = f(\mathbf{x}_{t-1}) + \boldsymbol{\eta}_t
```

```{math}
:label: eq-observations
\mathbf{y}_t = h(\mathbf{x}_t) + \boldsymbol{\varepsilon}_t
```

with the model noise $\boldsymbol{\eta}_t \in \mathbb{R}^{D_x}$ and the observation noise $\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\varepsilon)$. $f$ is a `ForwardModel`, $h$ an `ObservationOperator` and $a$ an `AnalysisStep`, and pipekit-cycle owns the loop that applies {eq}`eq-cycle` once per cycle and nothing else. An ensemble cycle carries $N_{\mathcal{E}}$ members $\mathbf{x}^{(e)}$, $e \in \mathcal{E}$, steps each through $f$, and gives the whole ensemble to one call of $a$.

## The calling convention

**A cycle is a stateful operator.** Every cycle takes `(carrier, state)` and returns `(carrier, state)`. The carrier is whatever your model steps (an array, an xarray Dataset, a list of members), and pipekit-cycle never looks inside it. The state is a `CarryState` such as `DAState(t, cycle_count, obs_err_cov, extras)`, where `t` is the model time in your own units, `cycle_count` the number of completed cycles, `obs_err_cov` the $\boldsymbol{\Sigma}_\varepsilon$ handed to the analysis and `extras` a free dictionary for the algorithm. The assimilation cycles return an advanced copy of the state rather than mutating it, so earlier states stay valid as snapshots. `IterationState` and `WindowState` are the states of a `Recurrence` and of a windowed cycle, and an algorithm library is free to subclass `CarryState` with fields of its own.

**History.** `Cycle(..., save_history=True)` appends `(carrier, state)` to `self.history` every `history_stride` steps, and `DACycle` (whose `save_history` is on by default) records `(forecast, analysis)` pairs. Each call overwrites the list.

**The protocols are structural.** `ForwardModel` asks for `step(state, dt)`, a `dt` and a `state_signature`. `ObservationOperator` asks for `__call__(state)` and `linearize(state)`. `AnalysisStep` is one call, `(forecast, obs, *, obs_op, obs_err_cov) -> analysis`. All of them are `runtime_checkable`, so `isinstance(obj, pc.ForwardModel)` is true for any object with those members, and the cycles check their components this way when you build them. `isinstance` checks that the members exist, not their signatures. Algorithm libraries ship adapters that satisfy the protocols without importing pipekit-cycle: filterax's `FilterAnalysisStep` in `filterax.pipekit`, plumax's forward model and observation operator in `plumax.adapters`, and the `.as_analysis_step()` method of each vardax method. The [protocols page](https://jejjohnson.github.io/pipekit/protocols/) lists all fourteen seams with their members.

## A forecast with no observations

A forecast is a `Cycle` around a forward model. Here the model is one explicit Euler step of exponential decay, $\mathrm{d}x / \mathrm{d}t = -\kappa x$ with $\kappa = 0.1$ and $\Delta t = 1$, on $D_x = 8$ cells, for $K = 24$ steps.

```python
import numpy as np
import pipekit_cycle as pc
from jaxtyping import Float


# Shapes: D_x = 8 grid cells; K = 24 steps
State = Float[np.ndarray, " state"]


# user
# x_k = x_k−1 − Δt κ x_k−1, one explicit Euler step of dx/dt = −κ x, with κ = 0.1
def decay(state: State, dt: float) -> State:
    return state - dt * 0.1 * state


# today (pipekit-cycle)
# f as a ForwardModel, with Δt = 1
step = pc.CallableForward(decay, dt=1.0)
# today (pipekit-cycle)
# x_0 ↦ x_K, keeping every step on forecast.history
forecast = pc.Cycle(step_op=step, n_steps=24, save_history=True)
# today (pipekit-cycle)
# x_K = (1 − Δt κ)ᴷ x_0 = 0.9²⁴ x_0
# (D_x,) → (D_x,)
final, _ = forecast(np.ones(8), None)
print(final[0].round(4), len(forecast.history))
```

This prints `0.0798 24`: each cell has decayed to $0.9^{24} \approx 0.080$ of its start, and the history holds all 24 steps. A forecast with no algorithm-level bookkeeping passes `state=None`. Swap `CallableForward` for `NeuralForward` around a trained model from the registry and the same `Cycle` forecasts with an emulator; add an observation operator and an analysis step and it assimilates, which is the next section.

## A first assimilation cycle

This is the README's quickstart. The state is a sea-surface-height anomaly on $D_x = 8$ grid cells of a periodic channel, and each cycle the forward model moves it one cell east and damps it, $f(\mathbf{x}) = \gamma \mathbf{S} \mathbf{x}$, with the cyclic shift $\mathbf{S} \in \mathbb{R}^{D_x \times D_x}$ and $\gamma = 0.98$. We observe every other cell ($D_y = 4$) with $\boldsymbol{\Sigma}_\varepsilon = \sigma_\varepsilon^2 \mathbf{I}$ and $\sigma_\varepsilon = 0.1$, over $T = 10$ cycles, and assimilate with a perturbed-observation EnKF of $N_{\mathcal{E}} = 20$ members. The analysis forms the anomalies $\tilde{\mathbf{x}}^{(e)} = \mathbf{x}^{(e)} - \bar{\mathbf{x}}$ and $\tilde{\mathbf{y}}^{(e)} = h(\mathbf{x}^{(e)}) - \bar{\mathbf{y}}$ about the ensemble means, the cross-covariance $\boldsymbol{\Sigma}_{xy} \in \mathbb{R}^{D_x \times D_y}$ and the covariance $\boldsymbol{\Sigma}_y \in \mathbb{R}^{D_y \times D_y}$ of the predicted observations plus $\boldsymbol{\Sigma}_\varepsilon$, and moves each member by the gain $\mathbf{G} = \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_y^{-1} \in \mathbb{R}^{D_x \times D_y}$ times its innovation $\mathbf{r}^{(e)} = \mathbf{y}^{(e)} - h(\mathbf{x}^{(e)})$, with $\mathbf{y}^{(e)} = \mathbf{y}_t + \boldsymbol{\varepsilon}^{(e)}$ a perturbed copy of the observations.

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

This prints `10 10.0 0.77 0.02`: ten cycles done with the clock at $t = 10$, and the ensemble mean, 0.77 (RMSE) from the truth before the first cycle, ends 0.02 from it after the tenth.

## Checking the seams

The EnKF above never imports pipekit-cycle, yet it is an `AnalysisStep`, because the protocols are structural. A bare function is not a `ForwardModel` (it has no `step` or `dt`), and a cycle refuses it when it is built rather than on its first step.

```python
# Shapes: D_x, D_y, N_ℰ and T as in the first cycle above

# today (pipekit-cycle)
# each component against its protocol, by shape: a, f, h, and the bare function
checks = (
    isinstance(StochasticEnKF(seed=1), pc.AnalysisStep),
    isinstance(forward, pc.ForwardModel),
    isinstance(obs_operator, pc.ObservationOperator),
    isinstance(advect, pc.ForwardModel),
)
# today (pipekit-cycle)
# a cycle checks its components the same way when it is built
try:
    pc.DACycle(advect, obs_operator, StochasticEnKF(seed=1))
except TypeError as error:
    refusal = str(error)
print(checks, refusal)
```

This prints `(True, True, True, False) DACycle.forward_model must satisfy ForwardModel, got function.`.

## Observations and their clock

`obs_source` is any `pipekit_ops.Operator`, called with a step index, so a whole pre-processing chain (fetch, deduplicate, bias-correct, quality control) can be one `Sequential`. When it is `None`, the assimilation cycles skip the analysis and pass the forecast on. By default the index restarts at 0 on every call (`obs_clock="call"`). With `obs_clock="state"` it continues from `state.cycle_count`, so calling a cycle again on its returned state reads the next observations and gives the same analysis as one uninterrupted run. `obs_err_source` works the same way for an observation-error covariance that changes per step: it is called with the same index, and its value is written to a copy of the state and passed to the analysis as `obs_err_cov` (`SmootherCycle` passes the list of the window's covariances, as it does the window's observations).

Run the cycle above as two calls of five on `obs_clock="state"`, and the second call reads $\mathbf{y}_6, \ldots, \mathbf{y}_{10}$:

```python
# Shapes: D_x, D_y, N_ℰ and T as in the first cycle above

# today (pipekit-cycle)
# five cycles per call, reading y_t from state.cycle_count onwards
half = pc.EnsembleDACycle(
    forward_model=forward,
    obs_op=obs_operator,
    analysis_step=StochasticEnKF(seed=1),
    obs_source=obs_source,
    n_steps=5,
    n_members=num_members,
    obs_clock="state",
)
# today (pipekit-cycle)
# the first call reads y_1, …, y_5
mid_members, mid_state = half(prior, start)
# today (pipekit-cycle)
# the second reads y_6, …, y_10
end_members, end_state = half(mid_members, mid_state)
same = np.allclose(np.stack(end_members), np.stack(members))
print(start.cycle_count, mid_state.cycle_count, end_state.cycle_count, same)
```

This prints `0 5 10 True`: the starting state still counts zero cycles (each call returned a copy), the first call ended at five and the second at ten, and the members match those of the single run of ten.

## The components

`IdentityObs` and `LinearObs(H)` cover direct and linear measurements; `LinearObs` relies only on `@`, so it works for NumPy and JAX arrays alike, and both return themselves from `linearize`. `CallableObs(fn, linearize_fn=None)` wraps a nonlinear operator such as a radiative transfer model, and its `linearize` raises `NotImplementedError` when no `linearize_fn` was given. `CompositeObs` chains operators, $h = h_k \circ \ldots \circ h_1$, and builds the composite tangent linear by the chain rule.

On the model side, `CallableForward(fn, dt)` wraps a function `(state, dt) -> state`, `CompositeForward` runs its components in turn (its `dt` is their sum, and each component steps by its own `dt`), and `NeuralForward(model_op, dt)` turns a trained emulator operator into a forward model at the fixed cadence it was trained for, ignoring the `dt` it is called with.

## Back-and-forth nudging

`BFNCycle` (Auroux and Blum, 2008) is an assimilation cycle that needs no adjoint and no analysis step. It integrates the window of $W$ steps forward, then backward, nudging toward the observations on both passes with your gain,

```{math}
:label: eq-nudging
\mathbf{x} \leftarrow \mathbf{x} + \mathbf{K}(\mathbf{y}_t - h(\mathbf{x}))
```

where $\mathbf{K}$ is the nudging gain, given as a function of the innovation. The forward pass steps and then nudges; the backward pass nudges and then steps, so each backward nudge lands on the same time level as its forward one. The sweeps repeat, up to `max_iters`, until the initial states of successive backward passes agree within `tol`, and `spectral_sigma` smooths the innovation with a Gaussian filter first, which needs SciPy. The backward pass steps the model with $-\Delta t$, so it needs a model that accepts a negative `dt` (or a `backward_model`), and `BFNCycle` refuses `CompositeForward` and `NeuralForward` without one, because both ignore the sign of `dt`.

## The adjoint specs

The specs in `pipekit_cycle.cycles` (`DirectAdjoint`, `RecursiveCheckpointAdjoint`, `TruncatedAdjoint`, `ImplicitAdjoint`, `BacksolveAdjoint`, with `AdjointSpec` their union) are frozen dataclasses with no behaviour. They name how a gradient flows back through a rollout: store every step, checkpoint (Griewank, 1992), stop the gradient more than `k` steps back (`k=1` is the one-step adjoint of Bolte et al., 2023), differentiate a converged fixed point implicitly, or solve the continuous adjoint backwards. A component that satisfies `SupportsAdjoint` carries one, the cycles pass it through untouched, and an execution layer interprets it: `integrix.adjoint.to_diffrax_adjoint` for a diffrax solve, `integrix.adjoint.truncated_scan` for a truncated rollout, and argminax's `to_optimistix_adjoint` for an optimistix solve, each matching the specs by class name. So choosing how gradients flow is configuration, not code. The [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-cycle/) has the table of what each spec means at the dynamics, the inner solve and the cycle.

## Saving a cycle

A cycle's configuration goes through `pipekit_ops.dumps` like any operator's, and `loads` rebuilds it when its components are operators that round-trip. `CallableObs`, `LinearObs` and `CallableForward` hold closures or arrays, so they set `forbid_in_yaml = True`: `dumps` writes a debug payload for them (the shape of `H`, the name of the function) and `loads` refuses it with a `RuntimeError`, so rebuild them in code. The same holds for a `Cycle` around a plain `ForwardModel`, which wraps it in an internal adapter that also forbids the round trip, and for `BFNCycle`, which holds the gain as a closure.

Every operator is recorded under `pipekit_cycle`, wherever its file is, so moving a file does not change a pipeline's hash. A state saved under a module of the flat layout of earlier releases (`pipekit_cycle.da`, `pipekit_cycle.cycle`, `pipekit_cycle.obs`, `pipekit_cycle.forward`, `pipekit_cycle.bfn`) still loads ([decision C1](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-cycle/docs/design/decisions.md)).

## Where it sits in pipekit

pipekit-cycle is built on pipekit-ops, and three packages of the workspace build on it.

| Package | What it takes |
| --- | --- |
| pipekit-jax | `ScanCycle`, `EnsembleScanCycle` and `Rollout` keep `Cycle`'s `(carrier, state)` contract under `jax.jit`, and take a `ForwardModel` |
| pipekit-train | its `cycle` extra, for a `SimulationDataset` that rolls out a `ForwardModel` or a `Cycle` |
| pipekit-evaluate | its `cycle` extra, for a benchmark `Task` whose simulator is a `ForwardModel` and an `ObservationOperator` |

## Not built yet

- **Latent data assimilation.** `LatentDACycle`, the `Encoder`, `Decoder` and `LatentForwardModel` protocols, and `LatentDAState` are designed in [`docs/design/latent.md`](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-cycle/docs/design/latent.md), a draft; only the `LatentMap` protocol ships.
- **A loop for the iterative-inversion seams.** `IterativeProcess` and `StepScheduler` are protocols only; no cycle drives them yet.
- **Localisation and inflation inside the cycles.** `CovarianceLocalizer` and `EnsembleInflator` are protocols only; today an analysis step applies them itself.

## References

- Auroux and Blum (2008), back-and-forth nudging, which `BFNCycle` implements.
- Bolte et al. (2023), the one-step adjoint, which `TruncatedAdjoint(k=1)` names.
- Griewank (1992), the checkpointing schedule behind `RecursiveCheckpointAdjoint`.
