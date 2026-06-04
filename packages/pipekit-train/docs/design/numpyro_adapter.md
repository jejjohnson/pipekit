---
status: draft
version: 0.1.0
---

# Design — NumPyro inference adapter (`backend="numpyro"`)

> A fourth `pipekit-train` backend that makes **Bayesian inference**
> (variational + MCMC) a first-class `TrainingLoop` target. Companion to
> `api/adapters.md` (the per-backend reference) and ADR **D13** in
> `decisions.md`. **Design only — no code yet.** Scope agreed: **SVI +
> NUTS**, design-doc-first.

---

## 1. User story

> *As a scientist with a probabilistic model — a NumPyro
> `model(x, y=None)` that places priors on parameters `θ` and a likelihood
> on observations — I want to **fit it through the same `TrainingLoop` I
> already use for neural nets**: pick variational (SVI) or MCMC (NUTS)
> inference, get the usual callbacks / metric logging / checkpoints /
> `TrainingArtifact`, and receive back a `pipekit.Operator` I can drop into
> an inference pipeline and store in the model registry — without
> hand-rolling an inference script that re-implements all of that.*

Two concrete personas, both already in the ecosystem:

- **The oracle builder.** A plume / ocean / land-extremes modeller whose
  "ground truth" estimator is a NumPyro **NUTS** posterior (the benchmark
  ladder's *rung-2 oracle*; `plumax`, `xtremax`, `somax` all use NumPyro
  here). They want that oracle to be a reproducible `TrainingLoop` run, not
  a notebook.
- **The fast-posterior user.** Someone who wants an amortizable /
  variational posterior (**SVI**, the ladder's *rung-4*) with the same
  optimizer-config, minibatching, early-stopping and checkpoint machinery
  the Equinox adapter already provides.

---

## 2. Motivation

- **Inference is "training" too.** `TrainingLoop` is a carrier-agnostic
  orchestration layer (ADR D3); fitting a posterior is exactly the
  `(model, data) → fitted thing + artifact` shape it already owns. A
  probabilistic backend is a natural fourth adapter, not a new primitive.
- **Ecosystem coherence with the benchmark ladder.** `pipekit-evaluate`'s
  `benchmark_ladder.md` names **NUTS = rung-2 (model-based oracle)** and
  **SVI = rung-4 (emulator-based inference)**. A NumPyro backend makes both
  rungs *trainable through the same loop*, so a benchmark `Task` can fit its
  oracle with `TrainingLoop(backend="numpyro")` and register the result.
- **Reuse, not reinvention.** It rides the existing adapter pattern (D5),
  the per-backend task seam (D9), the artifact/registry handshake (D6/D7),
  and the Equinox data iterator (`_BatchSource`). It adds no hard
  dependency — gated behind a `[numpyro]` extra.

---

## 3. Mathematical background

A NumPyro model is a generative story for parameters `θ` and observations
`y` given inputs `x`:

$$
\theta \sim p(\theta), \qquad y \mid x, \theta \sim p\bigl(y \mid f(x,\theta)\bigr),
\qquad p(\theta, y \mid x) = p(\theta)\, p(y \mid x, \theta).
$$

The target is the **posterior** $p(\theta \mid x, y) \propto
p(y \mid x, \theta)\, p(\theta)$, which is intractable in general. The two
inference modes are the two standard ways to get at it:

**Variational inference (SVI).** Pick a parametric family $q_\phi(\theta)$
(the *guide*) and maximise the evidence lower bound

$$
\mathrm{ELBO}(\phi) \;=\; \mathbb{E}_{q_\phi(\theta)}\!\bigl[\log p(\theta, y \mid x) - \log q_\phi(\theta)\bigr] \;\le\; \log p(y \mid x).
$$

Maximising the ELBO minimises $\mathrm{KL}\!\left(q_\phi \,\|\, p(\theta \mid x,y)\right)$.
This is a **gradient-ascent loop** in $\phi$ — one stochastic gradient step
per minibatch — which is exactly why it maps onto the per-step
`TrainingLoop`. NumPyro minimises $-\mathrm{ELBO}$.

**MCMC (NUTS).** The No-U-Turn Sampler (Hoffman & Gelman, 2014) is adaptive
Hamiltonian Monte Carlo: it draws correlated samples
$\{\theta^{(s)}\}_{s=1}^{S} \sim p(\theta \mid x, y)$ using gradients of the
log-joint, with a warmup phase that adapts the step size and mass matrix.
It is **not** a per-gradient-step optimiser — it is a single run of
`warmup + sampling` that consumes the **full** likelihood each step. It is
asymptotically exact (the oracle), at higher cost.

**Posterior predictive (the served artifact).** Whichever mode produced the
posterior, prediction at a new input $x^\ast$ marginalises over it:

$$
p(y^\ast \mid x^\ast, \text{data}) = \int p(y^\ast \mid x^\ast, \theta)\, p(\theta \mid \text{data})\, \mathrm{d}\theta
\;\approx\; \frac{1}{S}\sum_{s=1}^{S} p\!\bigl(y^\ast \mid x^\ast, \theta^{(s)}\bigr).
$$

The trained `Operator` wraps this map; `_apply(x)` returns its mean (full
samples optionally).

---

## 4. CS / coding background (NumPyro surface used)

NumPyro is JAX-native, so PRNG, JIT, and `vmap` all behave like the Equinox
adapter. The pieces the adapter touches:

- **Model** — a Python callable using primitives: `numpyro.sample("obs",
  dist, obs=y)` (the likelihood site, conditioned on data when `y` is
  given), `numpyro.plate(name, size, subsample_size=…)` (conditional
  independence + minibatch scaling), `numpyro.deterministic`.
- **SVI** — `SVI(model, guide, optim, loss)`. Key methods:
  `init(key, *args) → SVIState`; **`update(state, *args) → (state, loss)`**
  performs *one* gradient step; `get_params(state)`; `evaluate(state,
  *args)` returns the ELBO (for eval). `optim` accepts an **optax
  `GradientTransformation` directly**, so `optimizer_config` reuses the
  Equinox translation.
- **Guides** — `numpyro.infer.autoguide.AutoNormal` (mean-field; the
  default), `AutoDelta` (MAP), `AutoMultivariateNormal`, `AutoDAIS`, …
- **MCMC** — `MCMC(sampler, num_warmup=…, num_samples=…, num_chains=…)`
  (the count args are **keyword-only** in `numpyro>=0.15`), with the
  `NUTS(model)` / `HMC(model)` kernels. `run(key, *args)` is one blocking
  call; `get_samples()` returns a dict; `print_summary()` /
  `last_state` expose $\hat r$, effective sample size, divergences.
- **Predictive** — `Predictive(model, posterior_samples=samples)` (MCMC) or
  `Predictive(model, guide=guide, params=params, num_samples=N)` (SVI; the
  `num_samples` is **required** — a guide + params alone is not enough).
  Called `predictive(key, *args)` → a dict keyed by **site name**, so the
  adapter must be told *which* site is the prediction.
- **PRNG** — `jax.random.key(loop.seed)`, split per step (as the Equinox
  adapter already does).

---

## 5. Current state

`pipekit-train` today:

- `TrainingLoop` (a `StatefulOperator`, D3) dispatches to
  `pipekit_train.adapters.<backend>.run(loop)` via `_BACKEND_MODULES` and a
  `backend: Literal["equinox", "lightning", "keras"]`.
- **Equinox** is the implemented reference (loss-first *or* task-first); it
  synthesises a `TrainTask` from the carrier-agnostic `Loss(pred, target)`
  (D4), runs an `eqx.filter_jit` per-step loop, threads callbacks /
  `metric_writer` / Orbax checkpoints, and returns
  `(EquinoxModelOp, backend_info)`.
- **Lightning / Keras** are `NotImplementedError` scaffolds.
- **There is no probabilistic / Bayesian backend.** A user who wants a
  NumPyro posterior must hand-roll an inference script *outside* pipekit and
  forfeit the artifact, registry, checkpoint, callback and metric-writer
  machinery — and there is no `TrainingLoop`-shaped way to fit the benchmark
  ladder's NUTS oracle.

---

## 6. Target state

After this lands:

- `backend="numpyro"` is selectable, gated behind `[numpyro]` (clean
  `ImportError` without the extra, per D5).
- A **`NumpyroTask`** (model + guide + method) drives **SVI** (per-step,
  reusing the full loop) or **NUTS** (single `mcmc.run`).
- `run()` returns `(NumpyroPredictiveOp, backend_info)` — a
  posterior-predictive `pipekit.Operator` plus diagnostics (final ELBO for
  SVI; $\hat r$ / divergences / sample counts for NUTS).
- The trained `Operator` drops into inference pipelines and the model
  registry like any other (D7), its weight-blob being the variational
  params / posterior samples PyTree.
- Benchmark-ladder rung-2 (NUTS) and rung-4 (SVI) are now `TrainingLoop`
  targets, fit and registered reproducibly.

---

## 7. Proposal & API

### 7.1 Registration

- `[numpyro]` extra: `numpyro>=0.15`, `optax`, `grain` (JAX arrives via
  NumPyro). Add `"numpyro": "pipekit_train.adapters.numpyro"` to
  `_BACKEND_MODULES` and extend the `backend` `Literal`.
- A scaffold `run()` that raises `NotImplementedError` ships first (matching
  the Lightning / Keras pattern), then the real implementation.

### 7.2 `NumpyroTask` (the per-backend task — D9, **not** a `Loss`)

The adapter is **task-first**: NumPyro's objective is the ELBO / joint
log-density *defined by the model*, which does not match `Loss(pred,
target)` (D4). So `loop.task` is required; passing only `loop.loss` errors
(there is no loss→task synthesis, unlike the Equinox adapter).

```python
@dataclass
class NumpyroTask:
    """User-defined NumPyro inference target (loop.task for backend='numpyro')."""
    model: Callable           # def model(x, y=None): numpyro.sample("obs", ..., obs=y)
    guide: Callable | None = None      # default: AutoNormal(model) for SVI
    method: Literal["svi", "nuts"] = "svi"
    loss: Any = None                   # SVI objective; default Trace_ELBO()
    predictive_site: str = "obs"       # which model site the trained op returns
    predictive_samples: int = 200      # SVI: Predictive draw count (num_samples)
    num_warmup: int = 1000             # NUTS only
    num_samples: int = 1000            # NUTS only
    num_chains: int = 1                # NUTS only
```

`predictive_site` resolves the "which site?" ambiguity (`Predictive`
returns a dict keyed by site name); `predictive_samples` is required because
NumPyro's SVI `Predictive` needs a sample count.

### 7.3 `NumpyroPredictiveOp` (the trained artifact — D7)

```python
class NumpyroPredictiveOp(Operator):
    forbid_in_yaml: ClassVar[bool] = True   # carries the model closure
    # built around Predictive(...); _apply(x) draws S predictive samples and
    # returns the mean of task.predictive_site (full samples optional).
```

Registry weight-blob = the SVI params dict (SVI) or the posterior-samples
PyTree (NUTS) — directly analogous to `pipekit-jax.JaxModelOp`.

### 7.4 What `run(loop)` does

1. **Validate the task.** Require `loop.task` to be a `NumpyroTask` (error
   if only `loop.loss` is set). Default `guide=AutoNormal(model)` and
   `loss=Trace_ELBO()` for SVI.
2. **Prepare the data — mode-dependent.**
   - **SVI** reuses `_BatchSource` (Grain / streaming) exactly as Equinox
     does — `(x, y)` minibatches per step.
   - **NUTS** does *not* use the minibatch iterator: it **materialises the
     full dataset** into single `(X, y)` arrays (one pass over
     `loop.dataset`) and samples from all of it, since `mcmc.run` consumes
     the whole likelihood in one call. (Data subsampling for MCMC is a
     separate HMCECS path, out of scope for v1.)
3. **Dispatch on `task.method`.**
   - **svi:** `svi = SVI(model, guide, optax_optim, loss)`;
     `state = svi.init(key, *first_batch)`; per-step `svi.update` (log the
     ELBO, eval via `svi.evaluate` on `val_dataset`, checkpoint the
     `SVIState` params through Orbax) — the existing per-step loop unchanged.
   - **nuts:** `mcmc = MCMC(NUTS(model), num_warmup=task.num_warmup,
     num_samples=task.num_samples, num_chains=task.num_chains)`;
     `mcmc.run(key, X, y)`; `samples = mcmc.get_samples()`. One blocking
     call, `on_train_begin` / `on_train_end` only; `max_steps` is ignored.
4. **Wrap the artifact.** `NumpyroPredictiveOp` around
   `Predictive(model, guide=guide, params=svi_params,
   num_samples=task.predictive_samples)` (SVI) or
   `Predictive(model, posterior_samples=samples)` (NUTS); `_apply(x)`
   returns the mean of `task.predictive_site`. Fill `backend_info`.

### 7.5 `TrainingLoop`-field mapping

| `TrainingLoop` field | SVI | NUTS |
|----------------------|-----|------|
| `task` (required) | `NumpyroTask(model, guide, method="svi")` | `…method="nuts", num_warmup, num_samples` |
| `loss` | N/A — objective is the model's ELBO | N/A — joint log-density |
| `optimizer_config` | → optax → `SVI(optim=...)` | ignored |
| `max_steps` | SVI steps | → `num_samples` |
| `batch_size` | `_BatchSource` minibatch (`plate(subsample_size=…)`) | ignored — full dataset materialised |
| per-step loop / callbacks | `svi.update` + `svi.evaluate` | one `mcmc.run`, begin/end only |
| `checkpoint_dir` | `SVIState` params (Orbax) | posterior samples |
| trained `model_op` | `NumpyroPredictiveOp` → `predictive_site` mean | `NumpyroPredictiveOp` → `predictive_site` mean |
| `backend_info` | numpyro/jax ver, method, final ELBO | num_warmup/samples, divergences, $\hat r$ |

### 7.6 Open design decisions (where to push back)

1. **Task-first, no synthesis.** Confirmed: a NumPyro model *is* the
   objective, so no `Loss`→task path. Acceptable?
2. **One module, method-dispatched** (SVI + NUTS in `adapters/numpyro.py`)
   vs two backends (`"numpyro-svi"` / `"numpyro-mcmc"`). Proposed: one
   module (matches D5's "one module per backend").
3. **SVI minibatching.** Default to full-batch for v1 and add
   `numpyro.plate(subsample_size=…)` minibatch as a follow-up, or wire
   minibatch from the start? Proposed: full-batch first.
4. **Predictive return.** Mean of `predictive_site` by default; expose a
   flag to return full samples / multiple sites?

---

## 8. Example API usage

A Bayesian linear regression, fit two ways through the same loop.

```python
import jax, numpyro, numpyro.distributions as dist
from numpyro.infer import Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from pipekit_train import TrainingLoop, IterableDataset
from pipekit_train.adapters.numpyro import NumpyroTask

def model(x, y=None):
    w = numpyro.sample("w", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    mu = w * x[..., 0] + b
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)   # likelihood site

dataset = IterableDataset(source=pairs, content_hash="linreg-64")

# --- SVI: fast variational posterior (per-step; reuses the whole loop) ----
svi_loop = TrainingLoop(
    model_op=...,                       # ignored for numpyro; the task carries the model
    dataset=dataset,
    backend="numpyro",
    task=NumpyroTask(model, guide=AutoNormal(model), method="svi",
                     loss=Trace_ELBO(), predictive_site="obs",
                     predictive_samples=200),
    optimizer_config={"name": "adam", "lr": 1e-2},   # → optax → SVI(optim=...)
    max_steps=2000,
    batch_size=32,
    seed=0,
)
predictive_op, artifact = svi_loop.run()
y_hat = predictive_op(x_new)            # posterior-predictive mean of "obs"

# --- NUTS: the exact oracle (single mcmc.run; full dataset) ---------------
nuts_loop = TrainingLoop(
    model_op=...,
    dataset=dataset,
    backend="numpyro",
    task=NumpyroTask(model, method="nuts", num_warmup=1000, num_samples=1000,
                     predictive_site="obs"),
    seed=0,
)
oracle_op, oracle_artifact = nuts_loop.run()
oracle_artifact.backend_info["num_divergences"]   # NUTS diagnostics
y_hat_oracle = oracle_op(x_new)
```

Both `predictive_op` and `oracle_op` are ordinary `pipekit.Operator`s: they
compose into inference pipelines and round-trip through the model registry
exactly like a trained neural net (D7).

---

## 9. Steps to completion

```
1. [numpyro] extra + registry/Literal entry + scaffold run() raising
   NotImplementedError (matches the lightning/keras scaffolds).
2. NumpyroTask + NumpyroPredictiveOp + task-vs-loss validation
   (predictive_site / predictive_samples wired into the op).
3. SVI per-step path — svi.init/update/evaluate; ELBO logging; Orbax
   checkpoint of SVIState params; callbacks/writer/eval reuse the loop.
   -> test: Bayesian linear regression; ELBO decreases; slope recovered.
4. NUTS single-call path — full-dataset materialisation; MCMC(keyword args);
   get_samples(); divergences/r-hat into backend_info.
   -> test: posterior covers the true slope; predictive op round-trips.
5. SVI minibatching via numpyro.plate(subsample_size); finalise docs + ADR.
```

Tests gate on `importorskip("numpyro")`, like the Equinox suite (CI does not
install the inference extras, so they run locally / in an extra-enabled job).

---

## 10. References

**NumPyro / inference engine.**

- NumPyro SVI: <https://num.pyro.ai/en/stable/svi.html>
  (`SVI`, `Trace_ELBO`, `optax_to_numpyro`).
- NumPyro MCMC: <https://num.pyro.ai/en/stable/mcmc.html>
  (`MCMC`, `NUTS`, `HMC`).
- NumPyro autoguides: <https://num.pyro.ai/en/stable/autoguide.html>
  (`AutoNormal`, `AutoDelta`, `AutoMultivariateNormal`).
- NumPyro `Predictive`: <https://num.pyro.ai/en/stable/utilities.html>.

**Methods literature.**

- Hoffman & Gelman (2014), *The No-U-Turn Sampler*, JMLR — NUTS.
- Hoffman et al. (2013), *Stochastic Variational Inference*, JMLR.
- Ranganath, Gerrish & Blei (2014), *Black Box Variational Inference*.
- Kucukelbir et al. (2017), *Automatic Differentiation Variational
  Inference* (ADVI) — the basis of the autoguides.
- Blei, Kucukelbir & McAuliffe (2017), *Variational Inference: A Review for
  Statisticians*.
- Phan, Pradhan & Jankowiak (2019), *Composable Effects for Flexible and
  Accelerated Probabilistic Programming in NumPyro*.

**pipekit context.**

- `api/adapters.md` — the per-backend reference (concise NumPyro entry).
- `decisions.md` — ADRs D4 (Loss protocol), D5 (adapter pattern), D6/D7
  (artifact / Operator), D9 (per-backend `TrainTask`), **D13** (this
  adapter).
- `pipekit-evaluate` `benchmark_ladder.md` — NUTS = rung-2 oracle, SVI =
  rung-4.
