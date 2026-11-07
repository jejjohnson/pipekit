# pipekit

**A family of Python packages for composable scientific pipelines, built on two foundations you can use on their own: pipekit-ops for operators and pipekit-array for named-dim arrays.**

Every step of a scientific workflow (reading a scene, a QC check, a composite, a model, a forecast step, a training loop) becomes an `Operator` that takes a carrier and returns one.
Operators chain with `|`, wire into graphs, wrap in retries, caches and observers, map over many inputs, and save to JSON, and each of those is itself an Operator.
Install only what you need: each package is its own distribution, and the `pipekit` umbrella is there for when you want all of them.

## The layering

```
              your pipeline
                    │
   pipekit-cycle       time steps, DA cycles
   pipekit-train       training loops
   pipekit-experiment  registry, trackers
   pipekit-evaluate    benchmarks, scorers
   pipekit-jax         JAX / Equinox models
                    │
                    ▼  build on
   ┌─────────────────────────────────┐
   │ pipekit-array                   │
   │ named dims, any Array API       │
   │ backend                         │
   ├─────────────────────────────────┤
   │ pipekit-ops                     │
   │ Operator, Sequential, Graph;    │
   │ pure Python, no dependencies    │
   └─────────────────────────────────┘
     two foundations, each usable
     on its own
```

- **pipekit-ops** is a zero-dependency operator graph for any carrier: `Operator`, `|` and `Graph`, control flow, caching and stores, parallel `Map`, observers, QC and a JSON round-trip with a content hash. The [tour of the core](core-tour.md) covers every name.
- **pipekit-array** is named-dim array operators on any Array API backend: one operator runs on NumPy, JAX, CuPy, PyTorch or dask, and on a `DimArray`, an `xr.DataArray` or any carrier that registers an adapter.

The rest of the family builds on them.
pipekit-ops depends on nothing, and pipekit-array, pipekit-cycle, pipekit-experiment and pipekit-train depend on pipekit-ops only; pipekit-evaluate adds NumPy and SciPy, and pipekit-jax adds pipekit-cycle, JAX, Equinox and jaxtyping.

## Both foundations in one example

A year of Sentinel-2 over one tile: twelve monthly scenes, four bands and 64 by 64 pixels.
pipekit-array gives the named cube and the steps by dim name; pipekit-ops gives the shape check, the `|`, the summary and the JSON round-trip.

```python
import numpy as np

import pipekit_array as pa
import pipekit_ops as pk

# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels; K = 3 months per season

rng = np.random.default_rng(0)
reflectance = rng.uniform(0.0, 0.4, size=(12, 4, 64, 64)).astype(np.float32)

# today (pipekit-array)
cube = pa.DimArray(
    reflectance,                                       # (T, B, Y, X)
    ("time", "band", "y", "x"),
    {"band": ("B02", "B03", "B04", "B08")},
)

# today (pipekit-ops)
check = pk.AssertSignature("time band y x", dtype="float32")
# today (pipekit-array)
# s_{q,b,i,j} = (1/K) ∑ₖ x_{Kq+k,b,i,j}
seasonal = pa.Coarsen("mean", time=3)
# today (pipekit-array)
# m_{q,b} = (1/(Y X)) ∑ᵢ ∑ⱼ s_{q,b,i,j}
scene_mean = pa.Reduce("mean", ("y", "x"))
# today (pipekit-ops)
pipeline = check | seasonal | scene_mean

table = pipeline(cube)                                 # (T, B, Y, X) → (T/K, B)
# every intermediate shape, before or after running
print(pipeline.summary(pk.signature_of(cube)))

# today (pipekit-ops)
again = pk.loads(pk.dumps(pipeline))
assert pk.pipeline_hash(again) == pk.pipeline_hash(pipeline)
# the same pipeline on an xarray carrier: band labels come back as a coordinate
table_xr = pipeline(cube.to_xarray())                  # (T, B, Y, X) → (T/K, B)
```

To train an emulator, register it and forecast with it as the forward model, follow [Getting Started](getting-started.md).

## Three doors

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting Started](getting-started.md)**

    ---

    From `git clone` to your first pipeline, first DA cycle, and
    first registered model, in five minutes of copy and paste.

-   :material-book-open-page-variant: **[Concepts](concepts.md)**

    ---

    The 10-minute mental model. What's an `Operator`? Why
    carrier-agnostic? How does dispatch work?

-   :material-toolbox: **[Tutorials](tutorials/train-emulator.md)**

    ---

    End-to-end walkthroughs. Train an emulator. Run a DA cycle.
    Persist via the registry.

-   :material-package-variant: **[Installation](installation.md)**

    ---

    Per-package install matrix. Every extra spelled out.

-   :material-api: **[API Reference](api/pipekit-ops.md)**

    ---

    `mkdocstrings`-generated reference, kept in sync with source.

-   :material-notebook: **[Notebooks](notebooks/pipekit_train_quickstart.ipynb)**

    ---

    Executed Jupyter notebooks: an MLP quickstart, and the train → forecast
    handoff on a toy oscillator.

</div>

## The packages

Each package has its own README on GitHub, which is also its PyPI page.

| | Package | Purpose | Status | Version |
|---|---|---|---|---|
| ![](assets/packages/pipekit-ops/icon.svg){ width="28" } | [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) | **Foundation.** The operator core: `Operator`, `Sequential`, `Graph`, control and flow limits, dict and iterable operators, `Map`, `Cache` and stores, observe, QC. | ✅ implemented | 0.2.1 |
| ![](assets/packages/pipekit-array/icon.svg){ width="28" } | [pipekit-array](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array) | **Foundation.** Array API dispatch, `DimArray`, dim functions and their operators, `ApplyAlong`, `BatchAlong`, `KernelOp`, `Estimator`; the array QC asserts, the array `Histogram` and `ModelOp` are still to come. | 🚧 partial | 0.1.1 |
| ![](assets/packages/pipekit-cycle/icon.svg){ width="28" } | [pipekit-cycle](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) | Time-stepping, DA cycles, observation/forward protocols. | ✅ implemented | 0.1.0 |
| ![](assets/packages/pipekit-experiment/icon.svg){ width="28" } | [pipekit-experiment](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment) | Content-addressed model registry, tracker protocols, tool adapters. | ✅ implemented | 0.1.0 |
| ![](assets/packages/pipekit-evaluate/icon.svg){ width="28" } | [pipekit-evaluate](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-evaluate) | Benchmark ladder; `Unit × Lens × Stage` scorers; `EvaluationReport`. | 🚧 partial | 0.2.0 |
| ![](assets/packages/pipekit-train/icon.svg){ width="28" } | [pipekit-train](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train) | Training pipelines: datasets, losses, the loop, Equinox, NumPyro and BlackJAX adapters. | ✅ implemented | 0.2.0 |
| ![](assets/packages/pipekit-jax/icon.svg){ width="28" } | [pipekit-jax](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax) | `JaxModelOp` (an `eqx.Module` wrapper with registry weight round-trip), `JaxOperator`, `ScanCycle`. | ✅ implemented | 0.1.0 |

## Links

- [GitHub repository](https://github.com/jejjohnson/pipekit)
- [Changelog](https://github.com/jejjohnson/pipekit/blob/main/CHANGELOG.md)
- [Contributing](contributing.md)
- Master plan: [`research_journal_v2/notes/geotoolz/master_plan/`](https://github.com/jejjohnson/research_journal_v2/tree/main/notes/geotoolz/master_plan)
