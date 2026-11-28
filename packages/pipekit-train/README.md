<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-train/logo-dark.svg">
  <img alt="pipekit-train" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-train/logo-light.svg" width="373">
</picture>

**Training as a pipekit operator: hashed datasets, losses, a loop that hands back the trained model as an operator with a record of how it was made, and sweeps, on pipekit-ops, with backends on Equinox and optax, NumPyro, BlackJAX and Keras 3.**

[![Tests](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pipekit/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pipekit/)
[![codecov](https://codecov.io/gh/jejjohnson/pipekit/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pipekit)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

📖 [Documentation](https://jejjohnson.github.io/pipekit/) ·
🚀 [Quickstart](https://jejjohnson.github.io/pipekit/quickstart-train/) ·
📚 [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-train/) ·
🧭 [Design](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train/docs/design)

</div>

---

Consider a methane plume model that takes the emission rate $`Q`$ and the wind speed $`u`$ and returns the enhancement downwind. It is too slow to call thousands of times inside an inversion, so we train a neural emulator on simulations of it, and later a network that goes the other way, from an observed plume to $`Q`$ (amortized inference). The training code is never the hard part, but it is the part that gets rewritten for every project: the dataset of simulations, the split, the loop, the checkpoints, the sweep over learning rates, and the record of which simulator, which data and which weights produced the model we deployed. pipekit-train writes a training run as one pipekit operator, whose result is the trained model (itself an operator) and that record. The run solves

```math
\hat{\boldsymbol{\theta}} = \mathrm{argmin}_{\boldsymbol{\theta}} \ \frac{1}{N} \sum_{n=1}^{N} \ell\left(g(\mathbf{x}_n; \boldsymbol{\theta}), \mathbf{y}_n\right)
```

where $`\mathcal{D} = \{(\mathbf{x}_n, \mathbf{y}_n)\}_{n=1}^{N}`$ are the $`N`$ samples a dataset yields, each an input $`\mathbf{x}_n \in \mathbb{R}^{D_x}`$ and a target $`\mathbf{y}_n \in \mathbb{R}^{D_y}`$, $`g(\cdot; \boldsymbol{\theta})`$ is the model with weights $`\boldsymbol{\theta} \in \mathbb{R}^{D_\theta}`$, and $`\ell`$ is the loss of one sample, a scalar; we write the average as $`\mathcal{L}(\boldsymbol{\theta}; \mathcal{D})`$. The backend approximates the minimum with a fixed number of optimiser steps, each on a batch $`\mathcal{D}^{(i)} \subset \mathcal{D}`$, and the Bayesian backends replace the minimum by the posterior $`p(\boldsymbol{\theta} \mid \mathcal{D})`$.

```python
# Shapes: N samples; D_θ weights
# (simulator, prior_op and emulator as in the quickstart; passes and task in the docs)

# today (pipekit-train)
# 𝒟 = {(xₙ, h(xₙ))}, with xₙ ~ p(x)
train_set = SimulationDataset(forward_model=simulator, prior=prior_op, n_samples=1_000)
# today (pipekit-train)
# θ̂ = argmin_θ ℒ(θ; 𝒟), on the Equinox backend
loop = TrainingLoop(model_op=emulator, dataset=train_set, loss=MSE(), max_steps=300)
# today (pipekit-train)
# g(·; θ̂) as an operator, and the record of the run
trained, artifact = loop.run()
# today (pipekit-train)
# p(θ ∣ 𝒟) instead: a NumPyro model in place of the loss, sampled by NUTS
bayes_loop = TrainingLoop(Identity(), passes, task=task, backend="numpyro-mcmc")
```

None of it is a new trainer. `TrainingLoop` and `HyperSweep` are [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) `StatefulOperator`s, the datasets and losses are `Operator`s, and the content hashes come from `pipekit_ops.saving`, so a training run composes, hashes and serialises like any other pipekit step. The `equinox` backend trains an [Equinox](https://github.com/patrick-kidger/equinox) module with an [optax](https://github.com/google-deepmind/optax) optimiser, reads its batches through [Grain](https://github.com/google/grain) and checkpoints with [Orbax](https://github.com/google/orbax). `numpyro-svi` and `numpyro-mcmc` fit a [NumPyro](https://github.com/pyro-ppl/numpyro) model with NumPyro's own SVI and NUTS, and `blackjax` samples the same model with the NUTS of [BlackJAX](https://github.com/blackjax-devs/blackjax). `keras` trains with [Keras 3](https://github.com/keras-team/keras)'s own `compile` and `fit` on JAX: a Keras model wrapped in a `KerasModelOp`, or an Equinox model, which comes back in the wrapper it was given. Each backend sits behind its own extra and is imported only when a loop runs it, so importing pipekit-train imports none of them.

pipekit-train is one package of the pipekit workspace; the [workspace README](https://github.com/jejjohnson/pipekit/blob/main/README.md) introduces the others.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_train_overview/pipekit_train_overview.png" alt="pipekit-train in one picture: you supply a dataset, a model, a loss and the settings; TrainingLoop runs one training over a backend (Equinox with optax, NumPyro, BlackJAX and Keras 3); out comes the trained model as an operator, its artifact, checkpoints and logs" width="100%"></p>

## 🌟 Highlights

- 🔁 **The trained model is an operator**: `TrainingLoop.run()` returns the trained model as a `pipekit_ops.Operator`, in the wrapper you passed in or as a `NumpyroPredictiveOp`. Beside it comes an artifact with the pipeline's config, the dataset's content hash, the trained model's hash, the backend's final metrics and, where uv is installed, the environment's `uv pip freeze`.
- 🧾 **Datasets that know what they hold**: every `TrainingDataset` has a `content_hash()`. `SimulationDataset` draws one seeded input per sample from a prior and runs a forward model or a whole `Cycle` on it, with train, validation and test splits on disjoint seeds, and `CachedDataset` stores what another dataset produced as zarr (on disk, S3 or GCS), keyed by that hash.
- 🎛️ **One loop, five backends**: the `backend=` string picks Equinox with optax, NumPyro SVI, NumPyro NUTS, BlackJAX NUTS or Keras's `compile` and `fit`, and one `NumpyroTask` serves all three Bayesian backends (BlackJAX takes it inside a `BlackjaxTask`).
- 🧮 **Losses as operators**: `MSE` works on NumPy and JAX arrays alike, `NLL` and `KL` take distribution-shaped predictions, `Composite` sums weighted losses and reports each part, and a loss that takes a `key` gets a fresh PRNG key per batch.
- 🪝 **Callbacks and writers**: `EarlyStopping` on a validation metric, `Checkpoint` for the Orbax cadence and for storing the final model in a registry by hash, `LogToExperiment` for any tracker, and `JSONLWriter` for one line of metrics per step.
- 🧭 **Sweeps**: `HyperSweep` runs one loop per point of a `ParameterGrid` (or of any iterable of parameter dicts) and ranks the trials on a validation metric.
- 🖧 **Data-parallel training**: a `ShardingSpec` places the Equinox model and each batch on a device mesh, and on several hosts each process reads its own shard of an indexable dataset.

## 📐 Four kinds of training, one loop

The dataset says what the pairs are, and the backend says what is done with them.

| Training | Pairs | Dataset | Objective | `backend=` |
|---|---|---|---|---|
| Direct supervised | labelled $`(\mathbf{x}_n, \mathbf{y}_n)`$ | `IterableDataset`, or your own | $`\mathcal{L}(\boldsymbol{\theta}; \mathcal{D})`$ | `"equinox"` |
| Emulator | $`(\mathbf{x}_n, h(\mathbf{x}_n))`$ | `SimulationDataset` | $`\mathcal{L}(\boldsymbol{\theta}; \mathcal{D})`$ | `"equinox"` |
| Amortized inference | $`(h(\mathbf{x}_n), \mathbf{x}_n)`$ | `SimulationDataset(invert=True)` | $`\mathcal{L}(\boldsymbol{\theta}; \mathcal{D})`$ | `"equinox"` |
| Bayesian fit | any $`(\mathbf{x}_n, \mathbf{y}_n)`$ | any | $`p(\boldsymbol{\theta} \mid \mathcal{D})`$ | `"numpyro-svi"`, `"numpyro-mcmc"`, `"blackjax"` |

**The first three rows** minimise the same loss and differ only in the pairs. For an emulator, $`h: \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}`$ is the simulator (a `pipekit_cycle.ForwardModel`, or a `Cycle` rolled out from $`\mathbf{x}_n`$) and $`\mathbf{x}_n \sim p(\mathbf{x})`$ is drawn from a prior operator, so the trained $`g_\text{emu}(\mathbf{x}; \hat{\boldsymbol{\theta}})`$ stands in for $`h`$. Amortized inference reads the same simulations the other way round, so the network $`g_\text{post}(\mathbf{y}; \boldsymbol{\theta})`$ maps an observation back to the input that made it; with `NLL` on a network that returns a distribution, that is neural posterior estimation. The loss is a `Loss` (`MSE`, `NLL`, `KL`, `Composite` or any callable), or a `TrainTask` for anything a loss cannot say, and the trained operator is the model wrapper you passed in.

**The last row** puts a prior on the parameters. The model is a NumPyro function that writes

```math
p(\boldsymbol{\theta} \mid \mathcal{D}) \propto p(\boldsymbol{\theta}) \prod_{n=1}^{N} p(\mathbf{y}_n \mid \mathbf{x}_n, \boldsymbol{\theta})
```

with $`p(\boldsymbol{\theta})`$ its priors and $`p(\mathbf{y}_n \mid \mathbf{x}_n, \boldsymbol{\theta})`$ its likelihood, and it reaches the loop inside a `NumpyroTask`. `numpyro-svi` fits a guide $`q(\boldsymbol{\theta}; \boldsymbol{\nu})`$, by default NumPyro's `AutoNormal` (Kucukelbir et al., 2017), by minimising $`\mathcal{L}_\text{ELBO}(\boldsymbol{\nu}) = -\mathrm{ELBO}(\boldsymbol{\nu})`$ over the whole dataset at every step (Hoffman et al., 2013); `numpyro-mcmc` and `blackjax` draw from the posterior with NUTS (Hoffman and Gelman, 2014). The trained operator is a `NumpyroPredictiveOp` (a `BlackjaxPredictiveOp` for a BlackJAX run on a bare log-density), whose call returns the posterior-predictive mean of one site, with the draws behind it from `posterior_samples()` and `predictive_draws(x)`. `backend="keras"` takes a `Loss`, or a `KerasTask` holding a Keras loss, and logs each step's own loss rather than Keras's running mean.

## 📥 Install

We have not put pipekit-train on PyPI, nor the pipekit packages it needs, so install it from GitHub at a release tag (the [changelog](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/CHANGELOG.md) lists them). The base package brings in pipekit-ops alone, which gives the datasets, losses, callbacks, writers and sweeps but no backend, and it needs Python 3.12 or newer. pip cannot find pipekit-ops by itself, so name it beside pipekit-train, at the same tag:

```bash
pip install \
  "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git@pipekit-array-v0.3.1#subdirectory=packages/pipekit-ops" \
  "pipekit-train @ git+https://github.com/jejjohnson/pipekit.git@pipekit-array-v0.3.1#subdirectory=packages/pipekit-train"
```

Nine extras add the rest. Four are backends: `equinox` for the reference backend (Equinox, optax, JAX, Grain, Orbax and jaxtyping), `numpyro` for `numpyro-svi` and `numpyro-mcmc` (NumPyro and optax), `blackjax` for `blackjax` (BlackJAX, with NumPyro as the bridge from a model to its log-density), and `keras` for the Keras backend (Keras 3, JAX, Grain and Orbax; an Equinox model also needs `equinox`). `cycle` brings pipekit-cycle's forward models and cycles, and `experiment` brings pipekit-experiment's `TrainingArtifact`, registries and trackers; with either, name that package at the same tag too. `cache`, `s3` and `gcs` are for `CachedDataset` (zarr, plus fsspec with s3fs or gcsfs). The quickstart needs `equinox` and `cycle`.

```bash
pip install \
  "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git@pipekit-array-v0.3.1#subdirectory=packages/pipekit-ops" \
  "pipekit-train[equinox] @ git+https://github.com/jejjohnson/pipekit.git@pipekit-array-v0.3.1#subdirectory=packages/pipekit-train"
```

A project managed with [uv](https://docs.astral.sh/uv/) gives each pipekit package a git source in its `pyproject.toml`, all at one tag. The packages of the workspace ask for each other at the same commit, so sources at different tags make uv report conflicting URLs for pipekit-ops:

```toml
[project]
dependencies = ["pipekit-ops", "pipekit-cycle", "pipekit-train[equinox,cycle]"]"]

[tool.uv.sources]
pipekit-ops = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-ops", tag = "pipekit-array-v0.3.1" }
pipekit-cycle = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-cycle", tag = "pipekit-array-v0.3.1" }
pipekit-train = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-train", tag = "pipekit-array-v0.3.1" }
```

uv applies `[tool.uv.sources]` only to a project's direct dependencies, so every pipekit package named there is listed in `dependencies` too.

## 🌫️ Quickstart: a plume emulator

The toy plume model takes $`\mathbf{x} = (Q, u) \in \mathbb{R}^{D_x}`$, $`D_x = 2`$, with the emission rate $`Q \sim \mathcal{U}(0.5, 2)`$ (scaled units) and the wind speed $`u \sim \mathcal{U}(1, 5)`$ m/s, and returns the peak enhancement $`\mathbf{y} = h(\mathbf{x}) \in \mathbb{R}^{D_y}`$ at $`D_y = 4`$ distances downwind, $`y_d = Q / (u (1 + d))`$ for $`d = 0, \dots, 3`$. `SimulationDataset` draws $`N = 1{,}000`$ simulations, one seed each, and keeps the first 800 seeds for training and the next 100 for validation. The emulator $`g_\text{emu}(\mathbf{x}; \boldsymbol{\theta})`$ is an MLP with weights $`\boldsymbol{\theta}`$, trained by 300 Adam steps on batches $`\mathcal{D}^{(i)}`$ of $`N_\text{batch} = 32`$ pairs to minimise

```math
\mathcal{L}_\text{MSE}(\boldsymbol{\theta}) = \frac{1}{N_\text{batch} D_y} \sum_{n \in \mathcal{D}^{(i)}} \Vert g_\text{emu}(\mathbf{x}_n; \boldsymbol{\theta}) - \mathbf{y}_n \Vert_2^2
```

which is the mean of the squared errors over the batch and the distances. Afterwards we compare the emulator with the simulator at a new input $`\mathbf{x}_* = (1.2, 2.5)`$.

```python
import equinox as eqx
import jax
import numpy as np
import pipekit_cycle as pc
from pipekit_ops import Lambda
from pipekit_train import MSE, SimulationDataset, TrainingLoop
from pipekit_train.backends.equinox import EquinoxModelOp


# Shapes: D_x = 2 inputs (Q, u); D_y = 4 distances; N = 1,000 simulations

# d = 0, 1, 2, 3, the distances downwind
# → (D_y,)
distances = np.arange(4, dtype=np.float32)


# user
# x ~ p(x), with Q ~ U(0.5, 2) and u ~ U(1, 5), one draw per seed
def prior(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform([0.5, 1.0], [2.0, 5.0]).astype(np.float32)


# user
# h(x) = Q / (u (1 + d)), the peak enhancement at each distance d
def plume_model(state: np.ndarray, dt: float) -> np.ndarray:
    emission_rate, wind_speed = state
    return emission_rate / (wind_speed * (1.0 + distances))


# today (pipekit-cycle)
# h as a forward model with step(x, dt)
simulator = pc.CallableForward(plume_model, dt=1.0)
# today (pipekit-ops)
# p(x) as an operator, called with seed=
prior_op = Lambda(prior)
# today (pipekit-train)
# 𝒟 = {(xₙ, h(xₙ))}, the training split: the first 80 % of N seeds
train_set = SimulationDataset(
    forward_model=simulator,
    prior=prior_op,
    n_samples=1_000,
    simulator_version="toy-plume-v1",
)
# today (pipekit-train)
# the validation split, the next 10 % of seeds
val_set = train_set.with_split("val")

# today (equinox)
# g_emu(x; θ), an MLP with two hidden layers of 32 units
mlp = eqx.nn.MLP(in_size=2, out_size=4, width_size=32, depth=2, key=jax.random.key(0))
# today (pipekit-train)
emulator = EquinoxModelOp(mlp)

# today (pipekit-train)
# θ̂ = argmin_θ ℒ_MSE(θ), by 300 Adam steps on batches of 32
loop = TrainingLoop(
    model_op=emulator,
    dataset=train_set,
    val_dataset=val_set,
    loss=MSE(),
    optimizer_config={"name": "adam", "lr": 3e-3},
    max_steps=300,
    batch_size=32,
    eval_every_n_steps=300,
)
# today (pipekit-train)
trained, artifact = loop.run()

# x_* = (Q, u) = (1.2, 2.5)
# → (D_x,)
test_state = np.array([1.2, 2.5], dtype=np.float32)
# h(x_*) and g_emu(x_*; θ̂)
# (D_x,) → (D_y,), (D_y,)
simulated = plume_model(test_state, dt=1.0)
emulated = np.asarray(trained(test_state))
# ℒ_MSE on the validation split, after the last step
# → ()
val_mse = artifact.backend_info["final_eval_metrics"]["mse"]
# the artifact records the content hash of 𝒟
same_data = artifact.dataset_hash == train_set.content_hash()
print(simulated.round(3), emulated.round(3), f"{val_mse:.1e}", same_data)
```

This prints `[0.48 0.24 0.16 0.12] [0.456 0.228 0.155 0.111] 7.0e-04 True`: the simulator's enhancement at $`\mathbf{x}_*`$ at the four distances, the emulator's, which is within 0.024 of it at every distance, a validation $`\mathcal{L}_\text{MSE}`$ of $`7.0 \times 10^{-4}`$, and `True` because the artifact holds the training set's content hash. `trained` is an `EquinoxModelOp`, so it drops into a `Sequential` or a `Graph` as it is, and `pipekit_cycle.NeuralForward(trained, dt)` makes it a forward model. Nothing here is specific to this model: swap `plume_model` for a radiative transfer code or a chemistry step and the loop is the same, pass `invert=True` to train the inverse map instead, and change the backend string to fit a Bayesian model.

The [quickstart](https://jejjohnson.github.io/pipekit/quickstart-train/) takes this further: a sweep over the learning rate and the width, a Bayesian fit with NumPyro's NUTS and then BlackJAX's on the same model, the table of backends, how the loop, the datasets, the losses and the artifact work, and what is not built yet.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_train_roles/pipekit_train_roles-dark.svg">
    <img alt="pipekit-train on the GeoStack role ladder: TrainingLoop and HyperSweep as recipes, the model operators as steps, each backend's run as an algorithm, the training state as models, losses, tasks and callbacks as strategies, hashing and checkpoint calls as primitives, and the datasets as structures" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_train_roles/pipekit_train_roles-light.svg" width="100%">
  </picture>
</p>

Each public name sits on one rung of the GeoStack role ladder, by the job it does; the table lists them by area. The package is five areas, and an area imports only from the ones before it. `pipekit_train` re-exports the first four areas' names except `Split`, `hash_config` and `call_loss`, and each backend is imported from its own area.

| Area | Symbols | Notes |
|---|---|---|
| `pipekit_train.data` | `TrainingDataset`, `IterableDataset`, `SimulationDataset`, `CachedDataset`, `Split`, `hash_config` | `(input, target)` pairs with a content hash and seeded splits, and the zarr cache |
| `pipekit_train.losses` | `Loss`, `MSE`, `NLL`, `KL`, `Composite`, `call_loss` | The loss protocol and the common losses, as operators |
| `pipekit_train.loop` | `TrainingLoop`, `TrainerCarryState`, `ValidationStep`, `Callback`, `EarlyStopping`, `Checkpoint`, `LogToExperiment`, `MetricWriter`, `JSONLWriter` | The loop as a `StatefulOperator`, its hooks and its metric writers |
| `pipekit_train.sweep` | `HyperSweep`, `ParameterGrid`, `SweepResult`, `SweepCarryState` | One loop per parameter set, ranked, run one after another |
| `pipekit_train.backends.equinox` | `EquinoxModelOp`, `TrainState`, `TrainTask`, `train_step`, `save_state`, `restore_state`, `ShardingSpec`, `shard_train_state`, `run` | The reference backend (`equinox` extra) |
| `pipekit_train.backends.numpyro` | `NumpyroTask`, `NumpyroPredictiveOp`, `run_svi`, `run_mcmc` | SVI and NUTS (`numpyro` extra) |
| `pipekit_train.backends.blackjax` | `BlackjaxTask`, `BlackjaxPredictiveOp`, `run` | NUTS, from a NumPyro model or a bare log-density (`blackjax` extra) |
| `pipekit_train.backends.keras` | `KerasModelOp`, `KerasTask`, `EquinoxLayer`, `build_keras_optimizer`, `import_keras`, `run` | Keras 3's `compile` and `fit` on JAX (`keras` extra) |

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [Quickstart notebook](https://jejjohnson.github.io/pipekit/pipekit-train-quickstart/) | a dataset, a loss, an `EquinoxModelOp` and a `TrainingLoop` on a synthetic 1-D regression |
| 2 | [Cycle handoff](https://jejjohnson.github.io/pipekit/pipekit-train-cycle-handoff/) | an emulator trained on a `ForwardModel`, then run as a `NeuralForward` inside a `Cycle` |
| 3 | [NumPyro SVI](https://jejjohnson.github.io/pipekit/pipekit-train-numpyro-svi/), [NumPyro NUTS](https://jejjohnson.github.io/pipekit/pipekit-train-numpyro-mcmc/) and [BlackJAX NUTS](https://jejjohnson.github.io/pipekit/pipekit-train-blackjax/) | one Bayesian linear regression under the three Bayesian backends |
| 4 | [Train an emulator](https://jejjohnson.github.io/pipekit/train-emulator/) | an MLP emulator trained, used as the model of a pipekit-cycle forecast and stored in pipekit-experiment's `ModelRegistry` |
| 5 | [Train, register, forecast](https://jejjohnson.github.io/pipekit/train-register-forecast/) | a Lorenz-96 archive cleaned, an emulator trained on it, stored with its preprocessing, reloaded as a cycle's forward model and verified |
| 6 | [The Keras backend](https://jejjohnson.github.io/pipekit/pipekit-train-keras/) | a Keras model trained with plain Keras and then through `TrainingLoop`, and one Equinox emulator under the Keras and Equinox backends |

## 📖 Documentation

- **[Quickstart](https://jejjohnson.github.io/pipekit/quickstart-train/)**: the emulator above in full, a sweep, a Bayesian fit under two engines, the backends, the calling conventions and what is not built yet.
- **[Design](https://jejjohnson.github.io/pipekit/readme-4/)**: the vision, the architecture, the boundaries, the API layer by layer, the NumPyro, BlackJAX and Keras backend designs and worked examples of the three training shapes, also [on GitHub](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train/docs/design).
- **[Decisions](https://jejjohnson.github.io/pipekit/decisions-2/)**: the ADRs D1 to D21, from the thin loop over existing tools to the Keras backend.
- **[API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-train/)**: every public name, generated from the docstrings.
- **[Getting started](https://jejjohnson.github.io/pipekit/getting-started/)**: the workspace's first trained model, stored in a registry and run in a cycle.
- **[`AGENTS.md`](https://github.com/jejjohnson/pipekit/blob/main/AGENTS.md)**: the rules for contributors and coding agents.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_train_ecosystem/pipekit_train_ecosystem-dark.svg">
    <img alt="The packages that import pipekit-train on their main branches, both through an extra, each with the areas it imports: geotoolz-patcher and plumax" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_train_ecosystem/pipekit_train_ecosystem-light.svg" width="100%">
  </picture>
</p>

pipekit-train is a package of the pipekit [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), in the Core tier of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the training run and its record, and composes the rest in:

| Concern | Owner |
|---|---|
| Operators, stateful operators, carry states, content hashes | [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) |
| Modules and their weights (`equinox` backend) | [Equinox](https://github.com/patrick-kidger/equinox) |
| Optimisers, from `optimizer_config` | [optax](https://github.com/google-deepmind/optax) |
| Shuffled batches of an indexable dataset | [Grain](https://github.com/google/grain) |
| Checkpoints | [Orbax](https://github.com/google/orbax) |
| SVI, NUTS, `Predictive`, models to log-densities | [NumPyro](https://github.com/pyro-ppl/numpyro) |
| NUTS on a log-density (`blackjax` backend) | [BlackJAX](https://github.com/blackjax-devs/blackjax) |
| `compile`, `fit` and Keras models (`keras` backend) | [Keras](https://github.com/keras-team/keras) |
| Forward models and cycles (`cycle` extra) | [pipekit-cycle](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) |
| Training artifacts, registries, trackers (`experiment` extra) | [pipekit-experiment](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment) |
| The dataset cache (`cache`, `s3`, `gcs` extras) | [zarr](https://github.com/zarr-developers/zarr-python) and [fsspec](https://github.com/fsspec/filesystem_spec) |

Dependencies point one way. pipekit-train needs pipekit-ops alone, and pipekit-cycle and pipekit-experiment only through extras; it does not use pipekit-array, and nothing in it imports a package built on it. On `main`, [geotoolz](https://github.com/jejjohnson/geotoolz)'s geotoolz-patcher builds its `XarrayWindowDataset` on `pipekit_train.data`, behind its `train` extra. [plumax](https://github.com/jejjohnson/plumax) trains its footprint emulator with the datasets, `MSE`, `EarlyStopping`, `TrainingLoop` and the Equinox model wrapper, behind its `emulate` extra. The notebooks of [kernellib](https://github.com/jejjohnson/kernellib)'s docs train with its loop, and the workspace's `pipekit` package installs it with the others. [pipekit-jax](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax) is not a dependency, but its `JaxModelOp` is the model wrapper to train with when the weights must be reloaded from a registry.

## 🌳 Repository layout

```
packages/pipekit-train/
├── src/pipekit_train/
│   ├── __init__.py     ← re-exports the names of the first four areas
│   ├── data/           ← datasets, splits, content hashes
│   ├── losses/         ← the Loss protocol and the losses
│   ├── loop/           ← TrainingLoop, callbacks, metric writers
│   ├── sweep/          ← HyperSweep, ParameterGrid
│   └── backends/       ← equinox/, numpyro/, blackjax/, keras/, and the seams they share
├── tests/              ← the areas' tests, with adapters/ for the backends
└── docs/design/        ← vision, architecture, decisions, boundaries, api/, examples/
```

## ⚙️ Development

From the workspace root:

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install                                # every dependency group except train-extras, and the pre-commit hooks
uv sync --group dev --group train-extras    # the Equinox, NumPyro and BlackJAX backends
uv run --group train-extras pytest packages/pipekit-train/tests
make lint                                   # ruff over the workspace
make typecheck                              # ty over every package
make docs                                   # the MyST prose and the Zensical API reference, into public/
```

The backend tests skip without the `train-extras` group, which `make install` leaves out; CI installs it as above. `make help` lists every target, and [`AGENTS.md`](https://github.com/jejjohnson/pipekit/blob/main/AGENTS.md) holds the rules for contributors and coding agents: the workflow, the checks a change must pass and the house style.

## 🎓 References

- Hoffman and Gelman (2014). [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](https://arxiv.org/abs/1111.4246). *JMLR*.
- Hoffman, Blei, Wang and Paisley (2013). [Stochastic Variational Inference](https://arxiv.org/abs/1206.7051). *JMLR*.
- Kucukelbir, Tran, Ranganath, Gelman and Blei (2017). [Automatic Differentiation Variational Inference](https://arxiv.org/abs/1603.00788). *JMLR*.
- Phan, Pradhan and Jankowiak (2019). [Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro](https://arxiv.org/abs/1912.11554). *arXiv*.

## 📄 License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/LICENSE).
