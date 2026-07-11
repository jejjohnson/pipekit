---
status: draft
version: 0.1.0
---

# Design — BlackJAX inference adapter (`backend="blackjax"`)

> A fifth `pipekit-train` backend that exposes **BlackJAX**'s sampler zoo
> (NUTS, MCLMC, stochastic-gradient MCMC, SMC, variational) as a
> `TrainingLoop` target. Sibling to [`numpyro_adapter.md`](numpyro_adapter.md)
> and ADR **D14** in `decisions.md`. **Design only — no code yet.** Scope:
> a **separate** backend that **interoperates** with the NumPyro adapters
> (`numpyro-svi` / `numpyro-mcmc`) through a shared Bayesian seam.

---

## 0. Relationship to the NumPyro adapters (read this first)

BlackJAX and NumPyro sit at **different layers**:

- **NumPyro is a PPL** — a modelling language (`numpyro.sample(...)`) *plus*
  built-in engines (its own NUTS / SVI).
- **BlackJAX is a sampler library, not a PPL.** It operates on a
  user-supplied **log-density function** `logdensity_fn(position) -> float`
  and is explicitly PPL-agnostic: it consumes a log-density from NumPyro,
  PyMC, Oryx, or a hand-written function.

So this is a **separate adapter** (`backend="blackjax"`), not a NumPyro
mode, for four reasons (ADR D14):

1. **Different tool / extra / module** (D5) — `[blackjax]`, not `[numpyro]`;
   forcing NumPyro on BlackJAX users would deny its PPL-agnostic point.
2. **Different task contract** — a `logdensity_fn`, not a `numpyro.model`.
3. **It fits the per-step loop *better* than NumPyro.** NumPyro's `MCMC.run`
   is one blocking call; BlackJAX exposes `kernel.step(rng, state) ->
   (state, info)` — **one sample per step** — so the adapter reuses the
   existing per-step `TrainingLoop` (callbacks / `metric_writer` /
   checkpoint / early-stop firing *per sample*).
4. **A bigger algorithm zoo** — MCLMC, **stochastic-gradient MCMC**
   (`sgld`/`sghmc`, minibatch MCMC that pairs with `_BatchSource`), tempered
   SMC, and VI (`pathfinder`/`svgd`) that NumPyro doesn't all provide.

**Shared seam (not duplicated):** both adapters produce the same artifact (a
posterior-predictive `Operator`) and both can start from a NumPyro model.
The **model→logdensity bridge + `Predictive` wrapping** is factored into a
shared helper (proposed `pipekit_train.adapters.bayes`); a `NumpyroTask`
is *one accepted task source* for this adapter. Net UX: **write a NumPyro
model once, switch the MCMC engine by flipping `backend="numpyro-mcmc"` ↔
`backend="blackjax"`** — without bundling.

---

## 1. User story

> *As a Bayesian practitioner I have a (potentially unnormalised) **log-density**
> — either hand-written in JAX, or derived from a NumPyro / PyMC model — and
> I want to sample it with a **specific, modern algorithm** (NUTS, MCLMC, or
> minibatch SG-MCMC) through the same `TrainingLoop` I use for everything
> else: per-step diagnostics (acceptance, energy, divergences) streamed to
> my writer, checkpointable sampler state, early stopping, and a
> `pipekit.Operator` + `TrainingArtifact` at the end.*

Two personas:

- **The sampler connoisseur** who wants an algorithm NumPyro's built-in MCMC
  doesn't expose (MCLMC for high dimensions; `sgld`/`sghmc` for large-data
  minibatch MCMC; tempered SMC for multimodal posteriors).
- **The "same model, different engine" user** who wrote a NumPyro model for
  a NumPyro adapter and wants to A/B it against a BlackJAX sampler without
  rewriting anything — flip the backend.

---

## 2. Motivation

- **Per-step MCMC is a natural `TrainingLoop`.** BlackJAX's
  `kernel.step` is literally one transition; the loop's per-step machinery
  (callbacks, `metric_writer`, checkpoint cadence, early-stop) applies
  *unchanged* and *per sample* — a cleaner fit than the NumPyro MCMC oracle,
  which the `numpyro-mcmc` adapter has to run as a single opaque `fit`.
- **Minibatch MCMC.** `sgld` / `sghmc` consume a *minibatch* gradient of the
  log-density per step — so the existing `_BatchSource` Grain iterator feeds
  them directly. This is large-data Bayesian inference that neither the
  Equinox nor the NumPyro adapters offer.
- **Engine choice without lock-in.** Reusing the NumPyro model→logdensity
  bridge makes BlackJAX a drop-in alternative engine for an existing model,
  strengthening the benchmark ladder's rung-2/rung-4 story (a second,
  independent way to produce the oracle posterior — useful for
  cross-checking).
- **Reuse, not reinvention** — adapter pattern (D5), per-backend task seam
  (D9), artifact/registry (D6/D7), the `_BatchSource` iterator, and the
  shared `bayes` predictive seam.

---

## 3. Mathematical background

Given a target density $\pi(\theta) \propto \exp\bigl(\ell(\theta)\bigr)$
known only through its (unnormalised) log-density $\ell(\theta) =
\log \tilde\pi(\theta)$, BlackJAX draws samples
$\{\theta^{(s)}\} \sim \pi$.

**Hamiltonian / NUTS.** Augment with momentum $p \sim \mathcal{N}(0, M)$ and
simulate Hamiltonian dynamics on $H(\theta, p) = -\ell(\theta) +
\tfrac12 p^\top M^{-1} p$ with leapfrog steps; NUTS (Hoffman & Gelman 2014)
chooses the trajectory length automatically. Each `kernel.step` is one such
transition. Warmup adapts the step size (dual averaging) and the mass matrix
$M$ — BlackJAX's `window_adaptation`.

**MCLMC** (microcanonical Langevin Monte Carlo, Robnik et al. 2023) — a
deterministic-dynamics sampler with strong high-dimensional scaling; same
`init`/`step` shape, with its own `find_L_and_step_size` tuning.

**Stochastic-gradient MCMC.** When $\ell(\theta) = \sum_{i} \ell_i(\theta)$
over data, SGLD/SGHMC replace the full gradient with a **minibatch** estimate
$\widehat{\nabla \ell}(\theta) = \tfrac{N}{|B|}\sum_{i \in B} \nabla \ell_i(\theta)$
plus injected noise, sampling the posterior at scale (Welling & Teh 2011;
Chen et al. 2014). This is exactly a *per-minibatch step* — the
`TrainingLoop` shape.

**Variational / SMC.** BlackJAX also ships `meanfield_vi` / `fullrank_vi`,
`pathfinder`, `svgd`, and tempered `smc` — alternative posterior
approximations on the same `logdensity_fn`.

**Posterior predictive** is identical to the NumPyro case (§3 there): the
collected samples $\{\theta^{(s)}\}$ define
$p(y^\ast \mid x^\ast) \approx \tfrac1S \sum_s p(y^\ast \mid x^\ast, \theta^{(s)})$,
which the served `Operator` wraps.

---

## 4. CS / coding background (BlackJAX surface used)

BlackJAX is a thin, functional, JAX-native library — no global state, no
modelling DSL. The pieces the adapter touches:

- **Kernel construction** — `blackjax.nuts(logdensity_fn, step_size,
  inverse_mass_matrix)`, `blackjax.mclmc(...)`, `blackjax.sgld(grad_estimator)`,
  `blackjax.hmc/mala/ghmc/barker(...)`. Each returns an object with:
  - `init(position) -> state`
  - **`step(rng_key, state) -> (state, info)`** — one transition; `info`
    carries acceptance probability, energy, `num_integration_steps`,
    `is_divergent`, …
- **Warmup / adaptation** — `blackjax.window_adaptation(algorithm,
  logdensity_fn, num_warmup)` → `(last_state, tuned_parameters)`; MCLMC uses
  `mclmc_find_L_and_step_size`.
- **Scan helper** — `blackjax.util.run_inference_algorithm(...)` runs the
  sampling loop with `jax.lax.scan` (the fast path); the adapter uses the
  explicit `step` form when it needs per-step callbacks, and the scan form
  for tight inner loops.
- **PPL bridge** — `numpyro.infer.util.initialize_model(key, model,
  model_args=(x, y))` returns the initial (**unconstrained**) params, the
  `potential_fn`, and a **`postprocess_fn`** (the constrain / transform map).
  The shared `bayes` seam builds `logdensity_fn = lambda z: -potential_fn(z)`
  *and keeps `postprocess_fn`* — sampled positions are in unconstrained
  space, so they must be mapped back to the model's constrained sites before
  `Predictive` (essential for transformed latents such as a `HalfNormal`
  scale).
- **PRNG** — `jax.random.key(loop.seed)`, split per step.

Everything is JAX, so `jit` / `vmap` (for multiple chains) / sharding behave
as in the Equinox and NumPyro adapters.

---

## 5. Current state

- `pipekit-train` ships the **Equinox** backend (implemented) and
  **Lightning / Keras** scaffolds; the **NumPyro** adapters
  (`numpyro-svi` / `numpyro-mcmc`) are designed
  ([`numpyro_adapter.md`](numpyro_adapter.md), ADR D13) — `numpyro-mcmc`'s
  oracle runs as a single blocking `mcmc.run`.
- There is **no sampler-library backend**: no way to choose a specific
  BlackJAX algorithm, no per-step MCMC diagnostics through the loop, and **no
  minibatch MCMC** (SG-MCMC) at all.
- A user wanting BlackJAX must hand-roll the warmup + sampling loop and the
  NumPyro→logdensity bridge, forfeiting callbacks / writer / checkpoint /
  artifact / registry.

---

## 6. Target state

- `backend="blackjax"` is selectable, gated behind `[blackjax]`.
- A **`BlackjaxTask`** carries a `logdensity_fn` (+ initial position) *or* a
  NumPyro model (bridged to a logdensity), plus a `sampler` choice and warmup
  config.
- `run()` returns `(predictive_op, backend_info)` — the **same**
  posterior-predictive `Operator` the NumPyro adapters return (shared
  `bayes` seam) plus per-run diagnostics (acceptance, divergences, ESS).
- **Per-step sampling** drives the existing loop: each `kernel.step` is a
  loop step, so diagnostics stream to the `metric_writer`, the sampler state
  checkpoints, and early-stopping works.
- **SG-MCMC** uses `_BatchSource` minibatches — minibatch Bayesian inference
  at scale.
- Same NumPyro model → either engine by flipping `backend`.

---

## 7. Proposal & API

### 7.1 Registration

- `[blackjax]` extra: `blackjax>=1.2`, `optax`, `grain` (JAX via BlackJAX);
  `numpyro` only required for the NumPyro-model task path (a soft, lazily
  imported dependency). Add `"blackjax":
  "pipekit_train.adapters.blackjax"` to `_BACKEND_MODULES` + the `Literal`.
- Scaffold `run()` raising `NotImplementedError` ships first.

### 7.2 `BlackjaxTask` (the per-backend task — D9)

```python
@dataclass
class BlackjaxTask:
    """A BlackJAX inference target (loop.task for backend='blackjax')."""
    # Either supply a raw target + initial position …
    logdensity_fn: Callable | None = None    # position -> log p(position)  (full-batch)
    init_position: Any = None
    predict_fn: Callable | None = None       # (params, x) -> prediction (raw path)
    # … or derive everything from a NumPyro model (the shared bridge):
    # logdensity_fn, init_position, the constrain map, a grad estimator,
    # and the Predictive op are all built from the model.
    numpyro_task: NumpyroTask | None = None

    sampler: Literal["nuts", "mclmc", "hmc", "sgld", "sghmc"] = "nuts"
    num_warmup: int = 1000                    # full-batch: window_adaptation steps;
                                              # SG-MCMC: burn-in transitions to discard
    num_chains: int = 1                       # vmapped chains
    num_integration_steps: int | None = None  # required for hmc + sghmc
    # SG-MCMC (sgld / sghmc): the gradient is a *minibatch* estimate, not a
    # full logdensity_fn. Supply a grad estimator + a step size (BlackJAX's
    # sgld/sghmc are built from the estimator; the step takes the minibatch
    # and the step size per transition).
    grad_estimator: Callable | None = None    # (position, (x, y)) -> grad logpost
    step_size: float | Callable[[int], float] = 1e-3   # constant or schedule
    predictive_site: str = "obs"              # for the NumPyro-model path
    predictive_samples: int = 200             # thinned posterior draws used
```

Exactly one task **source** must be set: `numpyro_task` (which derives all of
the above from the model), or a raw target. The raw target is
`logdensity_fn` + `init_position` for the full-batch samplers, or
`grad_estimator` + `init_position` + `step_size` for SG-MCMC (BlackJAX's
`sgld`/`sghmc` are constructed from a `grad_estimator(position, minibatch)`,
not from a full `logdensity_fn`). `num_integration_steps` is **required** for
`hmc` and `sghmc` (their constructors — and `window_adaptation` for HMC —
need it; NUTS/MCLMC/SGLD do not). `predict_fn` is needed only to build a
predictive `Operator` from a raw task (see §7.5). The `numpyro_task` path
reuses the model, the constrain map, and `Predictive` via the shared
`bayes` seam.

### 7.3 `run(loop)` flow

1. **Resolve the target.** If `numpyro_task` is set, bridge via
   `initialize_model` → `logdensity_fn` (+ a minibatch `grad_estimator` for
   SG-MCMC) + `init_position` + `postprocess_fn` (the constrain map); else
   use the raw fields. (Error if neither / both.)
2. **Prepare data — sampler-dependent.**
   - Full-batch samplers (`nuts`, `mclmc`, `hmc`) materialise the dataset
     into the closure of `logdensity_fn` (one pass).
   - **SG-MCMC** (`sgld`, `sghmc`) reuse **`_BatchSource`** — each `step`
     takes a fresh minibatch and uses `grad_estimator(position, minibatch)`.
3. **Warmup / tuning.**
   - Full-batch: `adapt = blackjax.window_adaptation(algorithm,
     logdensity_fn)` (HMC also needs `num_integration_steps`);
     `(last_state, tuned_parameters), _ = adapt.run(key, init_position,
     num_warmup)`; build the tuned `kernel` from `tuned_parameters` (MCLMC
     uses `mclmc_find_L_and_step_size`).
   - SG-MCMC: no `window_adaptation`; the `kernel` is built from
     `grad_estimator` (and `num_integration_steps` for `sghmc`); the step
     size is supplied **per transition** (see step 4).
4. **Per-step loop** (the heart). The transition shape differs by family:
   - **Full-batch** (`nuts`/`hmc`/`mclmc`): `state, info = kernel.step(
     step_key, state)`; record `θ = state.position`; log `info` (acceptance,
     energy, `is_divergent`) to the `metric_writer`.
   - **SG-MCMC** (`sgld`/`sghmc`): BlackJAX's step returns the **new
     position** (no `info`) and takes the minibatch + step size per call —
     `position = kernel.step(step_key, position, minibatch, step_size(t))`;
     log the step size / grad-norm (no acceptance for SGLD).
   Fire callbacks and checkpoint the sampler state (Orbax) each step;
   `num_chains>1` is a `vmap`. (A `jax.lax.scan` fast path via
   `run_inference_algorithm` is used when no per-step callback is active.)
   **Burn-in:** the first `num_warmup` SG-MCMC transitions are discarded
   (the chain starts at an arbitrary `init_position`, so early states are
   transient, not posterior draws); only steps after burn-in are recorded
   into `posterior_samples`. (Full-batch chains start from the warmup-tuned
   `last_state`, so they record from step 0.)
5. **Constrain + wrap the artifact.** Collect the unconstrained positions
   `{z^(s)}`. **NumPyro path:** map them through `postprocess_fn` to
   constrained samples `{θ^(s)}` (required — `Predictive` expects the
   model's constrained sites), then
   `NumpyroPredictiveOp(Predictive(model, posterior_samples=θ))`, returning
   the mean of `predictive_site`. **Raw path:** if `task.predict_fn` is set,
   `BlackjaxPredictiveOp(predict_fn, posterior_samples)` whose `_apply(x)`
   averages `predict_fn(θ^(s), x)`; otherwise the op returns the posterior
   samples only (no predictive map available).

### 7.4 `TrainingLoop`-field mapping

| `TrainingLoop` field | BlackJAX (full-batch: nuts/mclmc) | BlackJAX (SG-MCMC: sgld/sghmc) |
|----------------------|-----------------------------------|-------------------------------|
| `task` (required) | `BlackjaxTask(..., sampler="nuts")` | `BlackjaxTask(..., sampler="sgld")` |
| `loss` | N/A — target is the log-density | N/A |
| `optimizer_config` | not used (step size from warmup) | not used (`step_size` is a task field) |
| `max_steps` | posterior draws (recorded from step 0) | total transitions; first `num_warmup` discarded as burn-in |
| `batch_size` | ignored — full-batch log-density | **`_BatchSource` minibatch** |
| per-step loop / callbacks | `kernel.step → (state, info)`; acceptance / energy / divergences | `kernel.step(key, pos, batch, step_size)` → new position; step-size / grad-norm only |
| `checkpoint_dir` | sampler state + samples (Orbax) | same |
| trained `model_op` | `bayes` predictive op → `predictive_site` | same |
| `backend_info` | sampler, num_warmup/draws, accept rate, divergences, ESS | + minibatch size |

### 7.5 Open design decisions (where to push back)

1. **Per-step vs scan.** Per-step `kernel.step` enables callbacks but is
   slower than `run_inference_algorithm`'s `scan`. Proposed: per-step when a
   callback/writer is attached, scan otherwise (chosen automatically).
2. **Raw-logdensity predictive.** Resolved: `predict_fn(params, x)` is an
   optional `BlackjaxTask` field; with it the op returns posterior-predictive
   means, without it the raw path returns posterior samples only. (The
   `numpyro_task` path always has a predictive via the model.)
3. **Shared seam location.** `pipekit_train.adapters.bayes` (model→logdensity
   + `Predictive` wrapping), imported by both `numpyro` and `blackjax`
   adapters. Confirm the factoring.
4. **Multiple chains** as `vmap` (single device) vs sharded across devices
   (reuse the Equinox `ShardingSpec`?). Proposed: `vmap` for v1.

---

## 8. Example API usage

Same Bayesian linear-regression model as the NumPyro doc, sampled with
BlackJAX — once from a NumPyro model, once from a raw log-density.

```python
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from pipekit_train import TrainingLoop, IterableDataset
from pipekit_train.adapters.bayes import NumpyroTask
from pipekit_train.adapters.blackjax import BlackjaxTask

def model(x, y=None):
    w = numpyro.sample("w", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Normal(w * x[..., 0] + b, sigma), obs=y)

dataset = IterableDataset(source=pairs, content_hash="linreg-64")

# --- BlackJAX NUTS on a NumPyro model (same model, different engine) ------
loop = TrainingLoop(
    model_op=...,                       # ignored; the task carries the model
    dataset=dataset,
    backend="blackjax",
    task=BlackjaxTask(numpyro_task=NumpyroTask(model), sampler="nuts",
                      num_warmup=1000, predictive_site="obs"),
    max_steps=1000,                     # = posterior draws
    seed=0,
)
predictive_op, artifact = loop.run()
artifact.backend_info["mean_acceptance"], artifact.backend_info["num_divergences"]
y_hat = predictive_op(x_new)            # posterior-predictive mean of "obs"

# --- BlackJAX SGLD on a raw target (minibatch MCMC) -----------------------
def grad_estimator(theta, batch):       # minibatch grad of the log-posterior
    x, y = batch
    ...
loop_sgld = TrainingLoop(
    model_op=...,                       # ignored; the task carries the target
    dataset=dataset,
    backend="blackjax",
    task=BlackjaxTask(grad_estimator=grad_estimator, init_position=theta0,
                      sampler="sgld", step_size=1e-4,
                      predict_fn=lambda th, x: th["w"] * x[..., 0] + th["b"]),
    max_steps=5000,
    batch_size=128,                     # _BatchSource feeds the minibatch grad
    seed=0,
)
samples_op, _ = loop_sgld.run()
```

Both ops are ordinary `pipekit.Operator`s — they compose into inference
pipelines and round-trip through the model registry like any trained model
(D7), identically to the NumPyro adapters' outputs.

---

## 9. Steps to completion

```
1. [blackjax] extra + registry/Literal entry + scaffold run() raising
   NotImplementedError (matches the lightning/keras scaffolds).
2. Factor the shared bayes seam (model->logdensity bridge + Predictive
   wrapping) shared with the NumPyro adapters; BlackjaxTask + predictive op.
3. Full-batch per-step path: window_adaptation warmup + kernel.step loop
   (NUTS), per-sample diagnostics to the writer, Orbax checkpoint of
   sampler state.
   -> test: NUTS on a NumPyro linreg recovers the slope; posterior covers
      truth; matches the NumPyro-adapter oracle within MC error.
4. SG-MCMC path: sgld/sghmc consuming _BatchSource minibatches.
   -> test: SGLD on a large synthetic posterior concentrates on the truth.
5. Extra samplers (mclmc) + multi-chain vmap + scan fast-path; docs + ADR.
```

Tests gate on `importorskip("blackjax")`, like the Equinox / NumPyro suites.

---

## 10. References

**BlackJAX.**

- BlackJAX docs (sampler zoo, `init`/`step`, `window_adaptation`):
  <https://blackjax-devs.github.io/blackjax/>.
- BlackJAX × NumPyro how-to (the `initialize_model` → logdensity bridge):
  <https://blackjax-devs.github.io/blackjax/examples/howto_use_numpyro.html>.
- BlackJAX repository: <https://github.com/blackjax-devs/blackjax>.

**Methods literature.**

- Hoffman & Gelman (2014), *The No-U-Turn Sampler*, JMLR — NUTS.
- Robnik, De Luca, Silverstein & Seljak (2023), *Microcanonical Hamiltonian
  Monte Carlo* — MCLMC.
- Welling & Teh (2011), *Bayesian Learning via Stochastic Gradient Langevin
  Dynamics* — SGLD.
- Chen, Fox & Guestrin (2014), *Stochastic Gradient Hamiltonian Monte Carlo*
  — SGHMC.
- Del Moral, Doucet & Jasra (2006), *Sequential Monte Carlo Samplers*.

**pipekit context.**

- [`numpyro_adapter.md`](numpyro_adapter.md) — the sibling PPL backend and
  the shared Bayesian seam.
- `api/adapters.md` — the per-backend reference (concise BlackJAX entry).
- `decisions.md` — ADRs D5 (adapter pattern), D6/D7 (artifact / Operator),
  D9 (per-backend task), D13 (NumPyro adapters), **D14** (this adapter).
- `pipekit-evaluate` `benchmark_ladder.md` — NUTS / SG-MCMC as rung-2/rung-4
  estimators.
