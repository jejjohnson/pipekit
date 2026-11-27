<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img alt="pipekit" src="docs/assets/logo-light.svg" width="340">
</picture>

**Composable scientific pipelines in Python: every step is an operator, from reading a scene to training, assimilating, scoring and registering a model, on two foundations, a zero-dependency operator graph and named-dim arrays on any Array API backend.**

[![Tests](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pipekit/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pipekit/)
[![codecov](https://codecov.io/gh/jejjohnson/pipekit/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pipekit)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

📖 [Documentation](https://jejjohnson.github.io/pipekit/) ·
🚀 [Getting started](https://jejjohnson.github.io/pipekit/getting-started/) ·
💡 [Concepts](https://jejjohnson.github.io/pipekit/concepts/) ·
📚 [API reference](https://jejjohnson.github.io/pipekit/reference/)

</div>

---

A research workflow is rarely one program. We read a year of scenes, drop the cloudy ones, composite them, train an emulator on the result, run it inside an assimilation cycle, score the analysis against a reference and store the model that produced it, and each of those steps usually lives in its own script with its own loop, retry and cache. pipekit makes every step an `Operator`, and every way of combining operators (a chain, a graph, a retry, a cache, a parallel map, a training loop, an assimilation cycle) is itself an `Operator`, so a whole workflow is one object,

```math
p = o_K \circ \ldots \circ o_2 \circ o_1
```

where $`o_k : \mathcal{X}_{k-1} \to \mathcal{X}_k`$ is the $`k`$-th of $`K`$ steps, $`\mathcal{X}_k`$ is the set of carriers it returns (a list of scene records, a NumPy array, an `xr.DataArray`, a model state or your own class) and $`p : \mathcal{X}_0 \to \mathcal{X}_K`$ is the pipeline. A step never looks inside a carrier it does not need to, and $`p`$ is an operator too, so it can be called, inspected before it runs, mapped over many inputs, hashed and written to JSON like any one of its steps. The workspace is seven packages, and each one supplies operators of one kind.

```python
# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels; K = 3 months per season
# (cube as in the quickstart below)

# today (pipekit-ops, pipekit-array)
# p = o_3 ∘ o_2 ∘ o_1, one Sequential
pipeline = check | pa.Coarsen("mean", time=3) | pa.Reduce("mean", ("y", "x"))
# today (pipekit-ops)
# (T, B, Y, X) → (T/K, B)
table = pipeline(cube)
# today (pipekit-ops)
# p rebuilt from its JSON, with the same content hash
again = pk.loads(pk.dumps(pipeline))
```

Two packages are foundations, usable on their own. [pipekit-ops](packages/pipekit-ops/README.md) is the operator graph, in pure Python with no dependencies, so a library can build on it without pulling in NumPy or JAX. [pipekit-array](packages/pipekit-array/README.md) writes array operations by dim name on the [Python Array API standard](https://data-apis.org/array-api/latest/), so one operator runs on [NumPy](https://numpy.org/), [JAX](https://github.com/jax-ml/jax), [CuPy](https://cupy.dev/), [PyTorch](https://pytorch.org/) or [dask](https://www.dask.org/) arrays, and on an [`xr.DataArray`](https://xarray.dev/). The other five build on them and hand the real work to the libraries that own it: training to [Equinox](https://github.com/patrick-kidger/equinox) and [optax](https://github.com/google-deepmind/optax), [NumPyro](https://github.com/pyro-ppl/numpyro) and [BlackJAX](https://github.com/blackjax-devs/blackjax), JIT compilation and gradients to JAX, and run tracking to [DVC](https://dvc.org/), [Hydra](https://hydra.cc/) and [Metaflow](https://metaflow.org/), each behind an extra.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_workspace_overview/pipekit_workspace_overview.png" alt="pipekit in one picture: you bring a carrier, your steps, the seams and a model to fit; one pipeline p = o_K ∘ … ∘ o_1, with the stateful operator and the content hash h(p); seven packages, each supplying operators of one kind, pipekit-ops and pipekit-array as the two foundations; out comes the result, its shape before it runs, its configuration as JSON, a registered model and a report; on the standard library, the Array API, the training and inference libraries and the run trackers" width="100%"></p>

## 🌟 Highlights

- 🧱 **One contract for every step**: an `Operator` has one method, `_apply`, and `|`, `Graph`, `Retry`, `Cache`, `Map` and the cycles each take operators and return one, so a step written for one package composes with the steps of all the others.
- 🪶 **Foundations that stand alone**: pipekit-ops declares `dependencies = []`, and pipekit-array needs only pipekit-ops plus the array libraries you choose through its extras.
- 🧭 **Arrays by dim name, on any backend**: a pipekit-array operator says which dims it consumes, runs on any Array API array and on `xr.DataArray`, and stays traceable under `jax.jit` and `jax.vmap`.
- 🧾 **Configuration is data**: `dumps` and `loads` write and rebuild a whole pipeline as JSON, `pipeline_hash` is its content hash, and pipekit-experiment's registries store a trained model under that hash and check it again on load.
- 🌊 **The research loop as operators**: `TrainingLoop` and `HyperSweep` fit a model, `Cycle`, `DACycle` and `EnsembleDACycle` step and assimilate on a clock, and `evaluate` and `Benchmark` score an output against a reference, each as one operator in the same pipeline.
- 🔥 **JAX where it pays**: pipekit-jax makes an operator an Equinox module, and `ScanCycle` runs a whole forecast inside one `jax.lax.scan`, so it compiles once and differentiates end to end.
- 🔌 **Seams without imports**: the forward model, the observation operator, the analysis and the registry are runtime-checkable protocols, so a library such as filterax or vardax plugs in without importing pipekit.

## 📐 One pipeline, seven packages

Each package supplies operators of one kind. The table says what each one adds to $`p`$ and the names to start from.

| Package | Its operators | Start from |
|---|---|---|
| [`pipekit-ops`](packages/pipekit-ops/README.md) | any step, and every way of combining steps | `Operator`, `Sequential`, `Graph`, `Map` |
| [`pipekit-array`](packages/pipekit-array/README.md) | array operations by dim name | `DimArray`, `Reduce`, `Coarsen`, `KernelOp` |
| [`pipekit-cycle`](packages/pipekit-cycle/README.md) | a forecast and an analysis, repeated on a clock | `Cycle`, `DACycle`, `EnsembleDACycle` |
| [`pipekit-train`](packages/pipekit-train/README.md) | a fit that returns a trained model | `TrainingLoop`, `HyperSweep` |
| [`pipekit-evaluate`](packages/pipekit-evaluate/README.md) | a score against a reference | `MetricScorer`, `evaluate`, `Benchmark` |
| [`pipekit-experiment`](packages/pipekit-experiment/README.md) | a model stored under its hash | `LocalModelRegistry`, `S3ModelRegistry` |
| [`pipekit-jax`](packages/pipekit-jax/README.md) | operators that are JAX pytrees | `JaxOperator`, `JaxModelOp`, `ScanCycle` |

**Steps that carry state.** A cycle, a training loop and a sweep remember where they are between calls, so pipekit-ops gives them a second form, the stateful operator,

```math
(x_t, s_t) = o(x_{t-1}, s_{t-1})
```

where $`x_t`$ is the carrier after call $`t`$ and $`s_t`$ is a `CarryState` (the clock and the cycle count of `DAState`, the step, the epoch and the optimiser state of `TrainerCarryState`), so a run that stops can resume where it stopped, and a stateful operator chains with `|` like any other.

**A name for a configured pipeline.** Every step records its class and its constructor arguments, so a pipeline has a content hash,

```math
h(p) = \mathrm{SHA256}(\mathrm{json}(s_p))
```

where $`s_p`$ is the state of $`p`$ (each step's class, module and arguments, with nested operators as their own records) written with sorted keys. Two pipelines share $`h`$ exactly when they are configured the same, so $`h`$ is the key of a cache entry and of a registered model. The [concepts page](https://jejjohnson.github.io/pipekit/concepts/) has the model behind both in full.

## 📥 Install

The packages are not on PyPI, so install the ones you need from the repository. Each is its own distribution, and all of them need Python 3.12 or newer.

```bash
pip install \
  "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git@pipekit-array-v0.3.1#subdirectory=packages/pipekit-ops" \
  "pipekit-array[numpy] @ git+https://github.com/jejjohnson/pipekit.git@pipekit-array-v0.3.1#subdirectory=packages/pipekit-array"
```

pipekit-ops depends on nothing. pipekit-array, pipekit-cycle, pipekit-train and pipekit-experiment depend on pipekit-ops alone, pipekit-evaluate adds NumPy and SciPy, and pipekit-jax adds pipekit-cycle, JAX, Equinox and jaxtyping. Everything else is an extra of the package that uses it, listed in each package's README and in [the installation guide](https://jejjohnson.github.io/pipekit/installation/). The `pipekit` umbrella installs all seven and holds no code, so a library should depend on the package it imports, usually pipekit-ops.

The packages ask for each other by name, and uv cannot find those names on PyPI, so a project that uses them names each one in `[tool.uv.sources]`, all on the same tag (uv treats two tags of one repository as two sources):

```toml
[project]
dependencies = ["pipekit-ops", "pipekit-array[numpy]", "pipekit-cycle"]

[tool.uv.sources]
pipekit-ops = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-ops", tag = "pipekit-array-v0.3.1" }
pipekit-array = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-array", tag = "pipekit-array-v0.3.1" }
pipekit-cycle = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-cycle", tag = "pipekit-array-v0.3.1" }
```

## 🛰️ Quickstart: a year of Sentinel-2 over one tile

Twelve monthly scenes of four bands over a tile of 64 by 64 pixels make a cube $`x_{t,b,i,j}`$, with $`t`$ the month, $`b`$ the band and $`(i, j)`$ the pixel. We check its signature, average each season of $`K = 3`$ months, and average each season's scene over the pixels, so the pipeline returns one value $`m_{q,b}`$ per season $`q`$ and band. pipekit-array gives the named cube and the two steps by dim name; pipekit-ops gives the check, the `|` and the JSON round-trip.

```python
import numpy as np
import pipekit_array as pa
import pipekit_ops as pk


# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels; K = 3 months per season

rng = np.random.default_rng(0)
# x_{t,b,i,j}, reflectances in [0, 0.4]
reflectance = rng.uniform(0.0, 0.4, size=(12, 4, 64, 64)).astype(np.float32)

# today (pipekit-array)
cube = pa.DimArray(
    reflectance,  # (T, B, Y, X)
    ("time", "band", "y", "x"),
    {"band": ("B02", "B03", "B04", "B08")},
)

# today (pipekit-ops)
# o_1: the dims and dtype, or a QCError
check = pk.AssertSignature("time band y x", dtype="float32")
# today (pipekit-array)
# s_{q,b,i,j} = (1/K) ∑ₖ x_{Kq+k,b,i,j}
seasonal = pa.Coarsen("mean", time=3)
# today (pipekit-array)
# m_{q,b} = (1/(Y X)) ∑ᵢ ∑ⱼ s_{q,b,i,j}
scene_mean = pa.Reduce("mean", ("y", "x"))
# today (pipekit-ops)
# p = o_3 ∘ o_2 ∘ o_1, one Sequential
pipeline = check | seasonal | scene_mean

# (T, B, Y, X) → (T/K, B)
table = pipeline(cube)

# today (pipekit-ops)
# p rebuilt from its JSON, and h(p) compared
again = pk.loads(pk.dumps(pipeline))
same_hash = pk.pipeline_hash(again) == pk.pipeline_hash(pipeline)

# the same p on an xarray carrier: (T, B, Y, X) → (T/K, B)
table_xr = pipeline(cube.to_xarray())

print(table.dims, table.shape, same_hash, table_xr.band.values)
```

This prints `('time', 'band') (4, 4) True ['B02' 'B03' 'B04' 'B08']`: four seasons by four bands, the same hash for the pipeline rebuilt from its JSON, and the band labels back as a coordinate when the carrier is an `xr.DataArray`. `pipeline.summary(pk.signature_of(cube))` gives the signature after each step before anything runs, `(time=12, band=4, y=64, x=64)`, then `(time=4, band=4, y=64, x=64)`, then `(time=4, band=4)`, and the same `pipeline` runs on JAX arrays under `jax.jit`, as the [pipekit-array README](packages/pipekit-array/README.md) shows.

The [getting-started guide](https://jejjohnson.github.io/pipekit/getting-started/) goes from here to a running pipeline of your own, and each package's README has a quickstart of its own kind of operator.

## 🔬 The packages

| | Package | Import | Holds |
|---|---|---|---|
| <img src="docs/assets/packages/pipekit-ops/icon.svg" width="28" alt=""> | [`pipekit-ops`](packages/pipekit-ops/README.md) | `pipekit_ops` | **Foundation.** `Operator`, `Sequential`, `Graph`, the wrappers, `Map`, observers, stores, `dumps` and `loads`, `pipeline_hash` |
| <img src="docs/assets/packages/pipekit-array/icon.svg" width="28" alt=""> | [`pipekit-array`](packages/pipekit-array/README.md) | `pipekit_array` | **Foundation.** Array API dispatch, `DimArray`, dim functions and their operators, `ApplyAlong`, `BatchAlong`, `KernelOp`, `Estimator` |
| <img src="docs/assets/packages/pipekit-cycle/icon.svg" width="28" alt=""> | [`pipekit-cycle`](packages/pipekit-cycle/README.md) | `pipekit_cycle` | time stepping and data assimilation: the cycles, their protocols, observation and forward components, adjoint specs |
| <img src="docs/assets/packages/pipekit-train/icon.svg" width="28" alt=""> | [`pipekit-train`](packages/pipekit-train/README.md) | `pipekit_train` | `TrainingLoop`, datasets, losses, callbacks, `HyperSweep`, and the Equinox, NumPyro and BlackJAX backends |
| <img src="docs/assets/packages/pipekit-evaluate/icon.svg" width="28" alt=""> | [`pipekit-evaluate`](packages/pipekit-evaluate/README.md) | `pipekit_evaluate` | the `Unit × Lens × Stage` scorers, `EvaluationReport` and the benchmark ladder |
| <img src="docs/assets/packages/pipekit-experiment/icon.svg" width="28" alt=""> | [`pipekit-experiment`](packages/pipekit-experiment/README.md) | `pipekit_experiment` | content-addressed model registries, tracker protocols, run records, and the DVC, Hydra and Metaflow bridges |
| <img src="docs/assets/packages/pipekit-jax/icon.svg" width="28" alt=""> | [`pipekit-jax`](packages/pipekit-jax/README.md) | `pipekit_jax` | `JaxOperator`, `JaxModelOp` with its registry round trip, `ScanCycle`, `Rollout` |
| <img src="docs/assets/icon.svg" width="28" alt=""> | [`pipekit`](packages/pipekit/README.md) | none | the umbrella: installs all seven and holds no code |

Each package has its own [version and release notes](https://github.com/jejjohnson/pipekit/releases). Dependencies point one way: every package imports pipekit-ops, pipekit-jax also imports pipekit-cycle, two imports go through an extra (pipekit-jax's `array` extra for pipekit-array, pipekit-train's `experiment` extra for pipekit-experiment), and neither foundation imports any of the other five.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_workspace_ecosystem/pipekit_workspace_ecosystem-dark.svg">
    <img alt="The seven pipekit packages and the umbrella, with an arrow for each import between them: every package imports pipekit-ops, pipekit-jax imports pipekit-cycle, and through extras pipekit-jax imports pipekit-array and pipekit-train imports pipekit-experiment; each card names the libraries it builds on, and a strip names the GeoStack libraries that import the workspace" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_workspace_ecosystem/pipekit_workspace_ecosystem-light.svg" width="100%">
  </picture>
</p>

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [Build a bigger graph](https://jejjohnson.github.io/pipekit/anomaly-graph/) | an anomaly detector as one `Graph`, with parallel reads, retries, a cache, checks, a `Branch` and a `Fanout` |
| 2 | [ETL over many regions](https://jejjohnson.github.io/pipekit/etl-regions/) | one pipeline mapped over many regions |
| 3 | [Train an emulator](https://jejjohnson.github.io/pipekit/train-emulator/) | a plume emulator trained by `TrainingLoop` and stored in a registry |
| 4 | [Train, register, forecast](https://jejjohnson.github.io/pipekit/train-register-forecast/) | the trained model loaded by its hash and run inside a forecast cycle |
| 5 | [Run a DA cycle](https://jejjohnson.github.io/pipekit/da-cycle/) and [ensemble data assimilation](https://jejjohnson.github.io/pipekit/ensemble-da/) | the forecast, observe and analyse loop, with one state and with an ensemble |
| 6 | [Run inference](https://jejjohnson.github.io/pipekit/inference/) | sensor records in, cleaned, scored by a registered model, decisions out |
| 7 | [Pipelines under jit](https://jejjohnson.github.io/pipekit/pipekit-under-jit/) | a `Graph` traced under `jax.jit`, with pipekit-jax for gradients and scans |

## 📖 Documentation

- **[Documentation](https://jejjohnson.github.io/pipekit/)**: the guides, tutorials, notebooks and design pages of the whole workspace, as one site.
- **[Getting started](https://jejjohnson.github.io/pipekit/getting-started/)** and **[Installation](https://jejjohnson.github.io/pipekit/installation/)**: from `git clone` to a running pipeline, and every extra of every package.
- **[Concepts](https://jejjohnson.github.io/pipekit/concepts/)**, **[Protocols](https://jejjohnson.github.io/pipekit/protocols/)** and **[Patterns](https://jejjohnson.github.io/pipekit/patterns/)**: the ideas behind the workspace, the seams a library satisfies without importing pipekit, and the shapes of a deployment (batch, service, orchestrator, audit).
- **[Tour of the core](https://jejjohnson.github.io/pipekit/core-tour/)**: every group of pipekit-ops names, with a run output for each.
- **[Background: toolz](https://jejjohnson.github.io/pipekit/toolz/)**: why operators are classes and not functions.
- **[API reference](https://jejjohnson.github.io/pipekit/reference/)**: one page per package, generated from the docstrings.

## 🧩 Ecosystem

pipekit is in the Core layer of the [GeoStack](https://github.com/jejjohnson/geostack), but nothing in the two foundations is about geoscience. It owns the operator model and the loops around a model, and composes the rest:

| Concern | Owner |
|---|---|
| Arrays, and the standard every backend meets | the [Array API standard](https://data-apis.org/array-api/latest/), [array-api-compat](https://github.com/data-apis/array-api-compat), [NumPy](https://numpy.org/), [JAX](https://github.com/jax-ml/jax), [CuPy](https://cupy.dev/), [PyTorch](https://pytorch.org/), [dask](https://www.dask.org/) |
| Labelled arrays, estimators and named contractions | [xarray](https://xarray.dev/), [scikit-learn](https://scikit-learn.org/), [einx](https://github.com/fferflo/einx) |
| Models, optimisers and training | [Equinox](https://github.com/patrick-kidger/equinox), [optax](https://github.com/google-deepmind/optax), [Grain](https://github.com/google/grain), [Orbax](https://github.com/google/orbax) |
| Bayesian inference | [NumPyro](https://github.com/pyro-ppl/numpyro), [BlackJAX](https://github.com/blackjax-devs/blackjax) |
| Run tracking and orchestration | [DVC](https://dvc.org/), [Hydra](https://hydra.cc/), [Metaflow](https://metaflow.org/) |
| Parallel execution | the standard library's [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) and [`asyncio`](https://docs.python.org/3/library/asyncio.html), and any executor that meets their protocol |

In the GeoStack, these libraries import pipekit on their `main` branches, and pipekit imports none of them:

| Library | Packages it imports |
|---|---|
| [geotoolz](https://github.com/jejjohnson/geotoolz) (geotoolz-ops, xrtoolz-ops, geotoolz-array, geotoolz-patcher) | pipekit-ops, pipekit-array, pipekit-train |
| [gridax](https://github.com/jejjohnson/gridax) | pipekit-ops, pipekit-array, and pipekit-jax through its `pipekit` extra |
| [geophysix](https://github.com/jejjohnson/geophysix), [radiomax](https://github.com/jejjohnson/radiomax), [tracerax](https://github.com/jejjohnson/tracerax) | pipekit-array |
| [diagnostix](https://github.com/jejjohnson/diagnostix) | pipekit-ops, pipekit-array |
| [plumax](https://github.com/jejjohnson/plumax) | pipekit-ops, pipekit-array, pipekit-train |
| [radtrax](https://github.com/jejjohnson/radtrax), through its `geo` extra | pipekit-ops |
| [sesmax](https://github.com/jejjohnson/sesmax) | pipekit-ops, pipekit-cycle, pipekit-jax |
| [somax](https://github.com/jejjohnson/somax) | pipekit-cycle, pipekit-jax |
| [vardax](https://github.com/jejjohnson/vardax) | pipekit-ops, pipekit-cycle |

[filterax](https://github.com/jejjohnson/filterax) satisfies the pipekit-cycle protocols without importing pipekit at all. Outside the Earth sciences, a genomics pipeline on AnnData or an astronomy reduction on astropy tables gets the same composition, caching, parallel maps and saved configuration by writing operators for its own carrier.

## 🌳 Repository layout

```
pipekit/
├── pyproject.toml          ← the uv workspace root, never published
├── packages/
│   ├── pipekit-ops/        ← src/pipekit_ops, its tests and its design docs
│   ├── pipekit-array/      ← and so on for each package
│   ├── pipekit-cycle/
│   ├── pipekit-train/
│   ├── pipekit-evaluate/
│   ├── pipekit-experiment/
│   ├── pipekit-jax/
│   └── pipekit/            ← the umbrella, no code
├── docs/                   ← one site: MyST guides, tutorials and notebooks, and the API reference
├── notebooks/              ← the scripts behind the docs notebooks
└── scripts/                ← build_docs.py and the README assets
```

## ⚙️ Development

With [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install      # uv sync of every group but train-extras, plus the pre-commit hooks
make test         # every package's tests
make lint         # ruff check on the whole repository
make typecheck    # ty on every package
make docs-serve   # build the docs site and serve it locally
```

`make help` lists every target. Conventional Commits drive the releases (release-please, one version per package). The rules for contributors and coding agents are in [`AGENTS.md`](AGENTS.md), review follows [`CODE_REVIEW.md`](CODE_REVIEW.md), and the local workflow is in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 📄 License

MIT. See [LICENSE](LICENSE).
