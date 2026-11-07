<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-train/logo-dark.svg">
    <img alt="pipekit-train" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-train/logo-light.svg" width="373">
  </picture>
</p>

**Training loops for emulators, amortized inference and Bayesian models, as pipekit operators: a thin layer over Equinox and Optax, NumPyro and BlackJAX that hands back a trained operator and a record of how it was made.**

<p align="center">
  <a href="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Version 0.2.0" src="https://img.shields.io/badge/version-0.2.0-blue">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="https://jejjohnson.github.io/pipekit/"><b>Docs</b></a> ·
  <a href="https://jejjohnson.github.io/pipekit/api/pipekit-train/"><b>API</b></a> ·
  <a href="https://jejjohnson.github.io/pipekit/tutorials/train-emulator/"><b>Tutorial</b></a> ·
  <a href="https://jejjohnson.github.io/pipekit/design-train/"><b>Design</b></a>
</p>

## Why it exists

Consider a methane plume model that takes the emission rate $Q$ and the wind speed $u$ and returns the enhancement downwind.
It is too slow to call inside an inversion thousands of times, so we train a neural emulator on simulations of it, and later a network that goes the other way, from an observed plume to $Q$ (amortized inference).
The training code for this is never the hard part, but it is the part that gets rewritten for every project: the dataset of simulations, the split, the loop, the checkpoints, the sweep over learning rates, and the record of which simulator, which data and which weights produced the model we deployed.

pipekit-train keeps that part in one place and stays thin.
It does not write a fourth training loop; it hands your model, data and loss to an existing stack (Equinox with Optax, Orbax and Grain, or NumPyro, or BlackJAX) and gets back a trained `pipekit_ops.Operator`, which drops into an inference pipeline or a `pipekit_cycle.NeuralForward` without a rewrite.
Beside it comes a training artifact: the dataset's content hash, the trained model's hash, the config and the lockfile.
There are three shapes of training, with one framework:

1. **Direct supervised**, on labelled pairs (cloud masks, super-resolution, retrieval networks).
2. **Emulator training**, with `SimulationDataset` over a `pipekit_cycle.ForwardModel` (surrogates for radiative transfer, chemistry, plume dispersion).
3. **Amortized inference**, with `SimulationDataset(invert=True)`, where the network learns the inverse map (neural posterior estimation, neural likelihood, neural ratio).

## Where it sits

pipekit-train is built on pipekit-ops; it does not use pipekit-array.
Both are libraries in their own right, which you can build on directly.

```
   your emulators, amortized posteriors, Bayesian fits
                     │
          ┌──────────┴──────────┐
          │    pipekit-train    │  ◀ this package
          │ TrainingLoop, data, │
          │  sweeps, callbacks  │
          └──────────┬──────────┘
                     │
   backends (extras): Equinox + Optax + Orbax + Grain,
                      NumPyro, BlackJAX, Keras 3
   siblings (extras): pipekit-cycle ([cycle]: SimulationDataset)
                      pipekit-experiment ([experiment]: artifacts,
                                          registry, trackers)
                     │
            ┌────────┴────────┐
            │                 │
       pipekit-ops       pipekit-array
       (dependency:      (not used)
        Operator,
        StatefulOperator,
        CarryState, hashing)
```

- **pipekit-ops** is the one required dependency.
  `TrainingLoop` and `HyperSweep` are `StatefulOperator`s, datasets and losses are `Operator`s, and the content hashes use `pipekit_ops.hashing`.
- **pipekit-array** is not a dependency and is not used; array carriers pass straight through to the backend.
- **pipekit-cycle** and **pipekit-experiment** are optional siblings, behind the `[cycle]` and `[experiment]` extras.
  [pipekit-jax](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax) is not an extra, but its `JaxModelOp` is the model wrapper to use when you want to reload trained weights from a registry.

## Install

```bash
uv add 'pipekit-train[equinox]'                       # the reference backend
uv add 'pipekit-train[equinox,cycle,experiment]'      # plus simulations and the registry
```

| Extra | What it adds | When you need it |
| --- | --- | --- |
| *(none)* | `pipekit-ops` | the protocols, datasets, losses, callbacks, writers and sweeps; no backend |
| `equinox` | `equinox`, `optax`, `jax`, `grain`, `orbax-checkpoint`, `jaxtyping` | `backend="equinox"`, the reference backend |
| `numpyro` | `numpyro`, `optax` | `backend="numpyro-svi"` and `backend="numpyro-mcmc"` |
| `blackjax` | `blackjax`, `numpyro`, `optax` | `backend="blackjax"` (NUTS), with NumPyro as the model-to-logdensity bridge |
| `keras` | `keras` | nothing runs yet: `backend="keras"` raises `NotImplementedError` until v0.3 |
| `cycle` | `pipekit-cycle` | `SimulationDataset` over a `ForwardModel` or a `Cycle` |
| `experiment` | `pipekit-experiment` | `run()` returns a real `TrainingArtifact`; `Checkpoint(registry=...)` and `LogToExperiment` |
| `cache` | `zarr` | `CachedDataset` on a local directory |
| `s3` | `zarr`, `fsspec`, `s3fs` | `CachedDataset` on `s3://` |
| `gcs` | `zarr`, `fsspec`, `gcsfs` | `CachedDataset` on `gcs://` |

Without `[experiment]`, `run()` returns a local dataclass with the same field names as `TrainingArtifact`.

**Note**: in the workspace, the adapter tests skip unless the `train-extras` dependency group is installed, which `make install` leaves out.
Install it to run them, as CI does:

```bash
uv sync --group dev --group train-extras
uv run --group train-extras pytest packages/pipekit-train/tests
```

## Quickstart

The blocks need the `equinox` and `cycle` extras, and the third `numpyro`.

### Train an emulator

The toy plume model takes $\mathbf{x} = (Q, u) \in \mathbb{R}^{D_x}$, $D_x = 2$, and returns the peak enhancement $\mathbf{y} \in \mathbb{R}^{D_y}$ at $D_y = 4$ distances downwind, $y_d = Q / (u (1 + d))$.
We train an MLP emulator $g_\text{emu}(\mathbf{x}; \boldsymbol{\theta})$, with weights $\boldsymbol{\theta}$, on 2,000 simulations by minimising $\mathcal{L}_\text{MSE}(\boldsymbol{\theta})$, the mean of $\Vert g_\text{emu}(\mathbf{x}_n; \boldsymbol{\theta}) - \mathbf{y}_n \Vert_2^2$ over a batch.

```python
import equinox as eqx
import jax
import numpy as np
import pipekit_cycle as pc
from pipekit_ops import Lambda
from pipekit_train.adapters.equinox import EquinoxModelOp
from pipekit_train import MSE, SimulationDataset, TrainingLoop

# Shapes: D_x = 2 inputs (Q, u); D_y = 4 downwind distances; N = 2,000 simulations


# user
# x ~ prior: emission rate Q ∈ [0.5, 2] (scaled kg/h), wind speed u ∈ [1, 5] m/s
def prior(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform([0.5, 1.0], [2.0, 5.0]).astype(np.float32)  # (D_x,)


# user
# y_d = Q / (u (1 + d)): the plume's peak enhancement at distance d
def plume_model(x, dt):
    q, u = x
    return (q / (u * (1.0 + np.arange(4)))).astype(np.float32)  # (D_x,) → (D_y,)


# today (pipekit-train, pipekit-cycle)
train = SimulationDataset(
    forward_model=pc.CallableForward(plume_model, dt=1.0),
    prior=Lambda(prior),
    n_samples=2_000,
    simulator_version="toy-plume-v1",
)
val = train.with_split("val")                          # seeds never overlap
print(len(train), len(val))                            # 1600 200

# today (pipekit-train[equinox])
emulator = EquinoxModelOp(
    eqx.nn.MLP(in_size=2, out_size=4, width_size=32, depth=2, key=jax.random.key(0))
)

# today (pipekit-train)
# ℒ_MSE(θ) = mean ‖g_emu(x; θ) − y‖²
loop = TrainingLoop(
    model_op=emulator,
    dataset=train,
    val_dataset=val,
    loss=MSE(),
    optimizer_config={"name": "adam", "lr": 3e-3},
    max_steps=1_000,
    batch_size=64,
    eval_every_n_steps=500,
    backend="equinox",
)
trained, artifact = loop.run()
print(type(trained).__name__)                          # EquinoxModelOp
print(artifact.dataset_hash == train.content_hash())   # True
print(trained(prior(seed=99)).shape)                   # (4,)
print(artifact.backend_info["final_eval_metrics"]["mse"] < 1e-3)  # True
```

### Sweep the learning rate

`HyperSweep` builds one `TrainingLoop` per point of a search space, runs them in turn and ranks them on a validation metric.

```python
from pipekit_train import EarlyStopping, HyperSweep, ParameterGrid


# user
def make_loop(params):
    mlp = eqx.nn.MLP(2, 4, width_size=params["width"], depth=2, key=jax.random.key(0))
    return TrainingLoop(
        model_op=EquinoxModelOp(mlp),
        dataset=train,
        val_dataset=val,
        loss=MSE(),
        optimizer_config={"name": "adam", "lr": params["lr"]},
        max_steps=300,
        batch_size=64,
        eval_every_n_steps=100,
        callbacks=(EarlyStopping("mse", patience=2),),
    )


# today (pipekit-train)
sweep = HyperSweep(
    make_loop,
    ParameterGrid({"lr": [1e-3, 3e-3], "width": [16]}),
    select_metric="mse",
)
results = sweep.run()                                  # best first, by validation mse
print(len(results), sorted(results[0].params))         # 2 ['lr', 'width']
print(results[0].final_metric <= results[1].final_metric)  # True
```

### Fit a Bayesian model

The Bayesian backends take a NumPyro model instead of a loss.
Here we infer the source rate $Q$ from 60 passes, $y_n = Q x_n + \varepsilon_n$, where $x_n$ is the plume shape at pass $n$ and $\varepsilon_n \sim \mathcal{N}(0, \sigma^2)$, with NUTS.
The same `NumpyroTask` runs under `backend="numpyro-svi"`; the backend string picks the paradigm.

```python
import numpy as np
import numpyro
import numpyro.distributions as dist
from pipekit_ops import Identity
from pipekit_train import IterableDataset, TrainingLoop
from pipekit_train.adapters.bayes import NumpyroTask

# Shapes: N = 60 (feature, enhancement) pairs from one source; θ = (Q, σ)
rng = np.random.default_rng(0)
# x_n = 1 / (u_n (1 + d_n)), the plume shape at pass n
x = (1.0 / (rng.uniform(1, 5, 60) * (1 + rng.integers(0, 4, 60)))).astype(np.float32)  # (N,)
# y_n = Q x_n + ε_n,  Q = 1.4,  ε_n ~ 𝒩(0, 0.02²)
y = (1.4 * x + rng.normal(0.0, 0.02, 60)).astype(np.float32)  # (N,)
data = IterableDataset(list(zip(x, y)), content_hash="toy-passes-v1")


# user
# Q ~ LogNormal(0, 1),  σ ~ HalfNormal(0.1),  yₙ ~ 𝒩(Q xₙ, σ²)
def model(x, y=None):
    q = numpyro.sample("q", dist.LogNormal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))
    numpyro.sample("obs", dist.Normal(q * x, sigma), obs=y)


# today (pipekit-train[numpyro])
loop = TrainingLoop(
    model_op=Identity(),  # not used by the Bayesian backends
    dataset=data,
    task=NumpyroTask(model, num_warmup=300, num_samples=300),
    backend="numpyro-mcmc",
)
posterior_op, artifact = loop.run()
q_draws = posterior_op.posterior_samples()["q"]            # (300,)
print(abs(float(q_draws.mean()) - 1.4) < 0.05)            # True
print(posterior_op(np.float32([0.25])).shape)             # (1,)
```

## What ships

Version 0.2.0.
The top-level names are exported from `pipekit_train`; each backend lives in its own module under `pipekit_train.adapters` and is imported only when a loop asks for it.

| Module | Public names | Purpose |
| --- | --- | --- |
| `loop` | `TrainingLoop`, `TrainerCarryState`, `ValidationStep` | The loop as a `StatefulOperator`; `run()` returns `(trained_op, artifact)` |
| `dataset` | `TrainingDataset`, `IterableDataset`, `SimulationDataset`, `CachedDataset` | `(input, target)` pairs with a content hash and deterministic splits |
| `sweep` | `HyperSweep`, `ParameterGrid`, `SweepResult`, `SweepCarryState` | Run one loop per parameter set and rank the trials |
| `loss` | `Loss`, `MSE`, `NLL`, `KL`, `Composite` | The loss protocol and common losses, carrier-agnostic |
| `callbacks` | `Callback`, `EarlyStopping`, `Checkpoint`, `LogToExperiment` | Hooks on train, step, eval and epoch |
| `writer` | `MetricWriter`, `JSONLWriter` | Where per-step metrics go when no tracker is attached |
| `adapters.equinox` | `EquinoxModelOp`, `ShardingSpec`, `TrainState`, `TrainTask`, `train_step`, `save_state`, `restore_state` | The reference backend: Equinox, Optax, Orbax, Grain (`[equinox]`) |
| `adapters.numpyro_svi`, `adapters.numpyro_mcmc` | `run` | NumPyro SVI and NUTS backends (`[numpyro]`) |
| `adapters.bayes` | `NumpyroTask`, `NumpyroPredictiveOp` | The task and trained operator the NumPyro and BlackJAX backends share |
| `adapters.blackjax` | `BlackjaxTask`, `BlackjaxPredictiveOp` | The BlackJAX backend, NUTS (`[blackjax]`) |
| `adapters.keras` | `run` | A stub that raises `NotImplementedError` (`[keras]`) |

The backends, by the `backend=` string of `TrainingLoop`:

| `backend=` | Objective | Trained operator | Status |
| --- | --- | --- | --- |
| `"equinox"` | a `Loss`, or a `TrainTask` | the model wrapper you passed in (`EquinoxModelOp` or `pipekit_jax.JaxModelOp`) | the reference |
| `"numpyro-svi"` | a `NumpyroTask` (model, guide, ELBO) | `NumpyroPredictiveOp` | ships |
| `"numpyro-mcmc"` | a `NumpyroTask` (model, NUTS) | `NumpyroPredictiveOp` | ships |
| `"blackjax"` | a `BlackjaxTask` (a logdensity, or a `NumpyroTask`) | `BlackjaxPredictiveOp` | ships, NUTS only |
| `"keras"` | a `Loss` | none | raises `NotImplementedError` |

### Not built yet

- **The Keras 3 backend**, scheduled for v0.3.
- **`CatalogDataset`** and the `[catalog]` extra, once geocatalog is on PyPI; the design is in [`api/datasets.md`](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/api/datasets.md).
- **More cache formats.** `CachedDataset` writes zarr only; `"parquet"` and `"tfrecord"` raise `NotImplementedError`.
- **More BlackJAX samplers.** MCLMC, HMC, SGLD and SGHMC are planned; v1 is NUTS.
- **Parallel sweeps** and Optuna or Ray Tune samplers; trials run one after another, and any iterable of parameter dicts works as a search space today.
- **Stage-to-stage hand-off.** `TrainingLoop` ignores the carrier and state it is called with, so it does not yet take a model from an earlier stage of a `Sequential`.
- **Stateful layers, async logging and an online trainer**, listed in [`boundaries.md`](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/boundaries.md).

## How it works

### A thin loop over existing tools

`TrainingLoop` holds the model, data, loss and settings, and delegates the actual training to `pipekit_train.adapters.<backend>.run`, which it imports only when called.
The adapter owns the update step and the validation pass; the loop owns the callbacks and the artifact.
`max_steps` counts optimiser steps, not epochs, and `optimizer_config` is a flat dict such as `{"name": "adamw", "lr": 3e-4}` (`lr` and `learning_rate` both work).
For anything a `Loss` cannot say, pass `task=` instead: a `TrainTask` for Equinox, a `NumpyroTask` or `BlackjaxTask` for the Bayesian backends.
The Equinox backend also takes a `ShardingSpec` (`sharding=`) for multi-device and multi-host data-parallel training.

### Datasets are hashed, not just iterated

A `TrainingDataset` yields `(input, target)` pairs and has a `content_hash()`, which goes into the training artifact.
`IterableDataset` wraps any iterable with a hash you supply, because we cannot introspect an arbitrary source.
`SimulationDataset` draws inputs from a prior operator, runs the forward model (one step, or a whole `Cycle` rollout), and can push the output through an observation operator.
Each sample gets its own seed, and `with_split("val")` moves to a disjoint block of seeds (80/10/10), so splits never overlap.
Its hash covers the configs of the forward model, cycle and observation operator, the prior, the seed and the split; a Python function cannot be hashed reliably, so pass `simulator_version=` and change it when the simulator changes.
`CachedDataset` writes the pairs to zarr on the first pass, keyed by the source's hash, so a second run, on this machine or another, reads the cache instead of simulating again.

### Losses, callbacks and writers are protocols

`Loss` is any callable `(predicted, target) -> scalar` or `(scalar, aux_dict)`; `MSE` works on NumPy and JAX arrays by duck typing, `NLL` and `KL` take distribution-shaped predictions, and `Composite` sums weighted losses and reports each part.
A loss that takes a `key` keyword gets a fresh PRNG key per batch.
`Callback` has five optional hooks (`on_train_begin`, `on_step_end`, `on_eval_end`, `on_epoch_end`, `on_train_end`).
`EarlyStopping` watches an eval metric, `Checkpoint` sets the Orbax cadence and, given a registry, stores the final model by hash, and `LogToExperiment` forwards to any `ExperimentTracker`.
`MetricWriter` is one `write` call; `JSONLWriter` appends a line per step.

### What comes back

`run()` returns the trained model as an operator and an artifact with `training_pipeline_yaml`, `dataset_hash`, `trained_model_hash`, `tracker_run_id`, `model_registry_uri`, `backend_info` and `deps_lock`.
`backend_info` carries `final_step`, `final_metrics` and, with a validation set, `final_eval_metrics`, which is what `HyperSweep` ranks on.
The Bayesian backends return a predictive operator: calling it gives the posterior-predictive mean of one site, and `posterior_samples()` and `predictive_draws(x)` give the full draws for credible intervals.

### Reloading trained weights

`EquinoxModelOp` writes its weights with `serialize_weights`, so a registry can store them and two trained models get different hashes, but it cannot read them back; it keeps pipekit-train free of a pipekit-jax dependency.
To reload, train with `pipekit_jax.JaxModelOp` instead (same constructor), which the Equinox backend returns in the same wrapper, and use `JaxModelOp.from_registry`.
Like every model wrapper, both are marked `forbid_in_yaml`, so `registry.load` refuses to rebuild them from JSON; the [pipekit-jax README](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax) shows the pattern.

## Learn more

- The full design: [`packages/pipekit-train/docs/design/`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train/docs/design), also [on the docs site](https://jejjohnson.github.io/pipekit/design-train/).
  Start with the [vision](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/vision.md), the [architecture](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/architecture.md) and the [decisions](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/decisions.md) (ADRs D1 to D14); the backends are in [`api/adapters.md`](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/api/adapters.md), [`numpyro_adapter.md`](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/numpyro_adapter.md) and [`blackjax_adapter.md`](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/blackjax_adapter.md); worked examples for the three training shapes are in [`examples/`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train/docs/design/examples).
- API reference: [pipekit-train](https://jejjohnson.github.io/pipekit/api/pipekit-train/).
- Tutorials: [Train an emulator](https://jejjohnson.github.io/pipekit/tutorials/train-emulator/), [Train, register, forecast](https://jejjohnson.github.io/pipekit/tutorials/train-register-forecast/).
- Notebooks: [quickstart](https://jejjohnson.github.io/pipekit/notebooks/pipekit_train_quickstart/), [cycle hand-off](https://jejjohnson.github.io/pipekit/notebooks/pipekit_train_cycle_handoff/), [NumPyro SVI](https://jejjohnson.github.io/pipekit/notebooks/pipekit_train_numpyro_svi/), [NumPyro NUTS](https://jejjohnson.github.io/pipekit/notebooks/pipekit_train_numpyro_mcmc/), [BlackJAX NUTS](https://jejjohnson.github.io/pipekit/notebooks/pipekit_train_blackjax/).
- Master plan: [Report 11](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_9_pipekit_train.md).
- The foundations: [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) and [pipekit-array](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array).
- Siblings: [pipekit-cycle](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle), [pipekit-experiment](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment), [pipekit-jax](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax).

## License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/LICENSE).
