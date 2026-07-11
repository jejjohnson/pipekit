---
status: draft
version: 0.1.0
---

# Design — NumPyro inference adapters (`numpyro-svi` + `numpyro-mcmc`)

> **Two** `pipekit-train` backends that make **Bayesian inference** a
> first-class `TrainingLoop` target, split by paradigm:
> **`numpyro-svi`** (variational — a per-step optimizer) and
> **`numpyro-mcmc`** (sampling — a single blocking run). Companion to
> `api/adapters.md` and ADR **D13** in `decisions.md`. **Status: implemented
> (v1)** — both backends + the shared `adapters.bayes` seam ship; SVI
> minibatching (a `numpyro.plate` subsample hook) remains the documented
> follow-on. Tests are gated behind the `[numpyro]` extra and the
> `slow`/`integration` markers (CI runs the fast subset).
>
> **Why two adapters, not one?** SVI and MCMC only *look* alike. SVI's
> `svi.update` returns `(state, loss)` per step, so it genuinely *is* the
> per-step `TrainingLoop` — `max_steps`, `optimizer_config`, per-step
> eval/checkpoint all apply (and the `_BatchSource` minibatch iterator once a
> subsample plate is added; v1 is full-batch). MCMC's `mcmc.run` is one
> blocking call — none of those fields apply. A single
> method-dispatched module forced awkward "ignored for nuts" caveats;
> splitting removes them and lines NumPyro up cleanly with BlackJAX:
> **`numpyro-mcmc` and `blackjax` are sibling *sampler* backends**, while
> **`numpyro-svi` is the *variational* one**. All three share the
> model→predictive seam (`adapters.bayes`); see
> [`blackjax_adapter.md`](blackjax_adapter.md).

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
  **SVI = rung-4 (emulator-based inference)**. The two NumPyro backends make
  both rungs *trainable through the same loop* — a benchmark `Task` fits its
  oracle with `backend="numpyro-mcmc"` and its fast posterior with
  `backend="numpyro-svi"`, registering each result.
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

- `backend="numpyro-svi"` and `backend="numpyro-mcmc"` are both selectable,
  gated behind the shared `[numpyro]` extra (clean `ImportError` without it).
- One shared **`NumpyroTask`** (the model + predictive config, plus
  SVI-specific and MCMC-specific fields) is consumed by *either* backend —
  so a benchmark can swap engines without rewriting the task.
- `numpyro-svi` is a **per-step** optimizer reusing the full loop
  (callbacks / `metric_writer` / eval / checkpoint, `optimizer_config`,
  minibatching). `numpyro-mcmc` is a **single `mcmc.run`** (begin/end
  callbacks only).
- Both `run()`s return `(NumpyroPredictiveOp, backend_info)` — a
  posterior-predictive `pipekit.Operator` plus diagnostics (final ELBO for
  SVI; $\hat r$ / divergences / sample counts for MCMC).
- The trained `Operator` drops into inference pipelines and the model
  registry like any other (D7), its weight-blob being the variational
  params / posterior-samples PyTree.
- Benchmark-ladder rung-2 (`numpyro-mcmc`) and rung-4 (`numpyro-svi`) are now
  `TrainingLoop` targets, fit and registered reproducibly.

---

## 7. Proposal & API

### 7.1 Registration — two modules, one extra

- `[numpyro]` extra: `numpyro>=0.15`, `optax`, `grain` (JAX arrives via
  NumPyro). Two backend strings → two modules behind the *same* extra:
  `"numpyro-svi": "pipekit_train.adapters.numpyro_svi"` and
  `"numpyro-mcmc": "pipekit_train.adapters.numpyro_mcmc"` in
  `_BACKEND_MODULES` + the `backend` `Literal`. (Two backends sharing one
  extra is a deliberate, minor exception to D5's "one extra per backend" —
  the dependency is identical; the *paradigm* differs.)
- Each ships a scaffold `run()` raising `NotImplementedError` first.

### 7.2 Shared `NumpyroTask` (the per-backend task — D9, **not** a `Loss`)

Both adapters are **task-first**: NumPyro's objective is the ELBO /
joint log-density *defined by the model*, which does not match `Loss(pred,
target)` (D4). So `loop.task` is required; passing only `loop.loss` errors.
One shared task carries the common config plus each paradigm's fields, so
the *same task object works under either backend* (swap the engine, keep the
task):

```python
@dataclass
class NumpyroTask:
    """A NumPyro inference target (loop.task for backend='numpyro-svi'|'numpyro-mcmc')."""
    model: Callable           # def model(x, y=None): numpyro.sample("obs", ..., obs=y)
    predictive_site: str = "obs"       # which model site the trained op returns
    predictive_samples: int = 200      # Predictive draw count (num_samples)
    # numpyro-svi fields
    guide: Callable | None = None      # default: AutoNormal(model)
    loss: Any = None                   # SVI objective; default Trace_ELBO()
    # numpyro-mcmc fields
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 1
```

There is no `method` field — the **backend string** selects the paradigm;
each adapter reads only its own fields. `predictive_site` resolves the
"which site?" ambiguity; `predictive_samples` is required because NumPyro's
SVI `Predictive` needs a sample count.

### 7.3 Shared seam — `NumpyroPredictiveOp` (the artifact, D7)

`NumpyroPredictiveOp`, the model→logdensity bridge, and the `Predictive`
wrapping live in **`pipekit_train.adapters.bayes`**, imported by both
NumPyro adapters (and the BlackJAX adapter):

```python
class NumpyroPredictiveOp(Operator):
    forbid_in_yaml: ClassVar[bool] = True   # carries the model closure
    # _apply(x): draw S predictive samples, return the mean of predictive_site.
```

Registry weight-blob = the SVI params dict or the posterior-samples PyTree —
analogous to `pipekit-jax.JaxModelOp`.

### 7.4 `numpyro-svi` — what `run(loop)` does

1. **Validate.** Require a `NumpyroTask`; default `guide=AutoNormal(model)`,
   `loss=Trace_ELBO()`.
2. **Data.** Reuse `_BatchSource` (Grain / streaming) — `(x, y)` minibatches
   per step, exactly as Equinox.
3. **Fit (per-step).** `svi = SVI(model, guide, optax_optim, loss)`;
   `state = svi.init(key, *first_batch)`; per-step `svi.update` returning
   `(state, -ELBO)`; log the ELBO; eval via `svi.evaluate` on `val_dataset`;
   checkpoint the `SVIState` params (Orbax). The existing per-step loop,
   `optimizer_config`, `max_steps`, callbacks and writer all apply unchanged.
4. **Artifact.** `NumpyroPredictiveOp` around `Predictive(model, guide=guide,
   params=svi_params, num_samples=task.predictive_samples)`; `_apply(x)`
   returns the mean of `task.predictive_site`. `backend_info`: final ELBO,
   versions.

### 7.5 `numpyro-mcmc` — what `run(loop)` does

1. **Validate.** Require a `NumpyroTask` (the SVI fields are ignored).
2. **Data.** *No* minibatch iterator: **materialise the full dataset** into
   single `(X, y)` arrays (one pass over `loop.dataset`); `mcmc.run` consumes
   the whole likelihood in one call. (`max_steps`, `optimizer_config`,
   `batch_size`, per-step eval do not apply.)
3. **Fit (single call).** `mcmc = MCMC(NUTS(model),
   num_warmup=task.num_warmup, num_samples=task.num_samples,
   num_chains=task.num_chains)` (the count args are keyword-only in
   `numpyro>=0.15`); `mcmc.run(key, X, y)`; `samples = mcmc.get_samples()`.
   Fire `on_train_begin` / `on_train_end` only.
4. **Artifact.** `NumpyroPredictiveOp` around `Predictive(model,
   posterior_samples=samples)`; `_apply(x)` returns the mean of
   `task.predictive_site`. `backend_info`: `num_warmup`/`num_samples`,
   `num_divergences`, $\hat r$ summary.

### 7.6 `TrainingLoop`-field mapping

| `TrainingLoop` field | `numpyro-svi` | `numpyro-mcmc` |
|----------------------|---------------|----------------|
| `task` (required) | `NumpyroTask(model, guide=…)` | `NumpyroTask(model, num_warmup=…, num_samples=…)` |
| `loss` | N/A — objective is the model's ELBO | N/A — joint log-density |
| `optimizer_config` | → optax → `SVI(optim=...)` | not used |
| `max_steps` | SVI steps | not used (→ `num_samples`) |
| `batch_size` | full-batch in v1; minibatch needs a `plate(subsample_size=…)` (follow-on) | not used — full dataset materialised |
| per-step loop / callbacks | `svi.update` + `svi.evaluate`, full per-step loop | one `mcmc.run`, begin/end only |
| `checkpoint_dir` | `SVIState` params (Orbax) | posterior samples |
| trained `model_op` | `NumpyroPredictiveOp` → `predictive_site` mean | `NumpyroPredictiveOp` → `predictive_site` mean |
| `backend_info` | versions, final ELBO | num_warmup/samples, divergences, $\hat r$ |

### 7.7 Open design decisions (where to push back)

1. **Two adapters, one extra.** Decided: split by paradigm (`numpyro-svi` /
   `numpyro-mcmc`) — the per-step vs blocking shapes are too different to
   method-dispatch, and it aligns with BlackJAX (§0 of
   [`blackjax_adapter.md`](blackjax_adapter.md)). The shared `[numpyro]`
   extra is the accepted cost.
2. **Shared task vs two tasks.** Proposed: one `NumpyroTask` (above), so the
   same task swaps between backends. Alternative: `SviTask` / `McmcTask`.
3. **SVI minibatching.** Full-batch for v1, add
   `numpyro.plate(subsample_size=…)` later — or wire it from the start?
4. **Predictive return.** Mean of `predictive_site` by default; flag for
   full samples / multiple sites?

---

## 8. Example API usage

A Bayesian linear regression. **One task object**, fit by either backend —
flip the `backend` string to switch engines.

```python
import jax, numpyro, numpyro.distributions as dist
from numpyro.infer.autoguide import AutoNormal
from pipekit_train import TrainingLoop, IterableDataset
from pipekit_train.adapters.bayes import NumpyroTask

def model(x, y=None):
    w = numpyro.sample("w", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    mu = w * x[..., 0] + b
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)   # likelihood site

dataset = IterableDataset(source=pairs, content_hash="linreg-64")
task = NumpyroTask(model, guide=AutoNormal(model), predictive_site="obs",
                   num_warmup=1000, num_samples=1000)   # used by either backend

# --- numpyro-svi: fast variational posterior (per-step; reuses the loop) --
# v1 is FULL-BATCH: the model above has no `numpyro.plate(..., subsample_size=)`,
# so each step uses the whole dataset (correctly scaled ELBO). Minibatch SVI
# (batch_size < N) requires wrapping the likelihood in a subsample plate so the
# ELBO is rescaled — a documented follow-on (build order step 5).
svi_op, svi_artifact = TrainingLoop(
    model_op=...,                       # ignored; the task carries the model
    dataset=dataset,
    backend="numpyro-svi",
    task=task,
    optimizer_config={"name": "adam", "lr": 1e-2},   # → optax → SVI(optim=...)
    max_steps=2000,
    seed=0,
).run()
y_hat = svi_op(x_new)                    # posterior-predictive mean of "obs"

# --- numpyro-mcmc: the exact oracle (single mcmc.run; full dataset) -------
oracle_op, oracle_artifact = TrainingLoop(
    model_op=...,
    dataset=dataset,
    backend="numpyro-mcmc",
    task=task,                           # same task; SVI fields are ignored here
    seed=0,
).run()
oracle_artifact.backend_info["num_divergences"]   # MCMC diagnostics
y_hat_oracle = oracle_op(x_new)
```

Both `svi_op` and `oracle_op` are ordinary `pipekit.Operator`s: they compose
into inference pipelines and round-trip through the model registry exactly
like a trained neural net (D7).

---

## 9. Steps to completion

```
1. [numpyro] extra + two registry/Literal entries (numpyro-svi, numpyro-mcmc)
   + scaffold run()s raising NotImplementedError.
2. Shared adapters.bayes: NumpyroTask + NumpyroPredictiveOp + the
   model/Predictive helpers + task-vs-loss validation.
3. numpyro-svi adapter — svi.init/update/evaluate; ELBO logging; Orbax
   checkpoint of SVIState params; callbacks/writer/eval reuse the loop.
   -> test: Bayesian linear regression; ELBO decreases; slope recovered.
4. numpyro-mcmc adapter — full-dataset materialisation; MCMC(keyword args);
   get_samples(); divergences/r-hat into backend_info.
   -> test: posterior covers the true slope; predictive op round-trips;
      agrees with the numpyro-svi posterior within MC error.
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

- `api/adapters.md` — the per-backend reference (concise NumPyro entries).
- [`blackjax_adapter.md`](blackjax_adapter.md) — the sibling BlackJAX
  sampler backend; `numpyro-mcmc` and `blackjax` share the seam.
- `decisions.md` — ADRs D4 (Loss protocol), D5 (adapter pattern), D6/D7
  (artifact / Operator), D9 (per-backend `TrainTask`), **D13** (these two
  adapters).
- `pipekit-evaluate` `benchmark_ladder.md` — `numpyro-mcmc` = rung-2 oracle,
  `numpyro-svi` = rung-4.
