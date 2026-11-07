<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-array/logo-dark.svg">
    <img alt="pipekit-array" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-array/logo-light.svg" width="386">
  </picture>
</p>

**pipekit-array is named-dim array operators on any Array API backend: one operator runs on NumPy, JAX, CuPy, PyTorch or dask, and on any carrier with named dims, with or without the rest of pipekit.**

<p align="center">
  <a href="https://jejjohnson.github.io/pipekit/api/pipekit-array/"><b>API</b></a> ·
  <a href="https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array/docs/design"><b>Design</b></a> ·
  <a href="https://github.com/jejjohnson/pipekit/blob/main/docs/concepts.md#5-backend-dispatch-pipekit-array"><b>Concepts</b></a> ·
  <a href="https://github.com/jejjohnson/pipekit"><b>pipekit family</b></a>
</p>

Version 0.1.1, imported as `pipekit_array`; partly built (see [Not built yet](#not-built-yet)).

## Why it exists

Take a year of Sentinel-2 over one tile: twelve monthly scenes, four bands (B02, B03, B04, B08) and a 64 by 64 pixel window, so a cube with dims `(time, band, y, x)`.
The questions we ask of it are always phrased by dim name: the NDVI per scene, a three-month running mean over time, the seasonal composites, the change from one season to the next, the 90th percentile per pixel.
In code they usually turn into `axis=1` and `axis=(2, 3)`, which silently go wrong the day someone transposes the cube, and into one version for NumPy on a laptop and another for JAX on a GPU.

pipekit-array writes each of those steps once, by dim name, against the [Python Array API standard](https://data-apis.org/array-api/).
The same function or operator then runs on the plain `DimArray` carrier, on an `xr.DataArray`, or on any carrier whose library registers a named-array adapter, and on whatever backend holds the data.
Every operator is a `pipekit_ops.Operator`, so it chains with `|`, round-trips through JSON and has a `pipeline_hash`.

## Use it on its own

You get real value from this package without the rest of pipekit.
It depends only on pipekit-ops (pure Python, no dependencies), and every backend is an opt-in extra, so `pipekit-array[numpy]` is a small install.
The dim functions (`window`, `coarsen`, `quantile`, `diff`, …) are plain functions: you can call them in a notebook on a `DimArray` or a DataArray and never write an operator.

Who builds on it:

- in this repository, `pipekit-cycle` and `pipekit-jax` reach it through their `[array]` extras (`pipekit-jax` turns a `KernelOp` into a JAX operator with `to_jax`);
- in the GeoStack, geoarray's carriers reach the dim functions through their own registered adapter, and `geotoolz` and `xrtoolz` are designed as the carrier-specific layers on top ([architecture](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/architecture.md)).

## Where it sits

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
   │ pipekit-array     ◀ this package│
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

pipekit-array depends on pipekit-ops only.
The other packages do not require it: `pipekit-cycle` and `pipekit-jax` add it with their `[array]` extras.
The full picture is in the [top-level README](https://github.com/jejjohnson/pipekit#the-layering).

## Install

Until the packages are on PyPI, install from the repository, with the backends you use as extras:

```bash
uv add "pipekit-array[numpy,xarray] @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-array"
```

Once published:

```bash
uv add "pipekit-array[numpy]"
```

| Extra | What it adds | When you need it |
|---|---|---|
| `numpy` | NumPy 2 | arrays on the CPU; the quickstart |
| `jax` | JAX | arrays on a GPU or TPU, and `jit` / `vmap` |
| `torch` | PyTorch | PyTorch tensors; add `compat` too |
| `cupy` | CuPy | NumPy-style arrays on a CUDA GPU |
| `dask` | `dask[array]` | lazy, chunked arrays larger than memory |
| `compat` | `array-api-compat` | backends without a native `__array_namespace__` (PyTorch) |
| `xarray` | xarray | the `xr.DataArray` adapter and `DimArray.to_xarray` |
| `sklearn` | scikit-learn | `Estimator` |
| `einx` | einx | `pipekit_array.einx`: einx patterns with dim names as axes |

## Quickstart

All three blocks use the same cube.
$T$ is the number of monthly scenes, $B$ the number of bands, $Y$ and $X$ the rows and columns of pixels, $S$ the length of a sliding window and $K$ the number of months per season.
The reflectance of band $b$ at pixel $(i, j)$ and time $t$ is $x_{t,b,i,j}$.
NDVI per scene and pixel is the normalised difference of the near-infrared and red bands,

```math
v_{t,i,j} = \frac{x_{t,\text{B08},i,j} - x_{t,\text{B04},i,j}}{x_{t,\text{B08},i,j} + x_{t,\text{B04},i,j} + \epsilon}
```

where $\epsilon = 10^{-6}$ keeps the ratio finite over dark water.

### 1. Dim functions on a named cube

```python
import numpy as np

import pipekit_array as pa

# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels;
#         S = 3 months per window; K = 3 months per season

rng = np.random.default_rng(0)
reflectance = rng.uniform(0.0, 0.4, size=(12, 4, 64, 64)).astype(np.float32)

# today (pipekit-array)
cube = pa.DimArray(
    reflectance,                                       # (T, B, Y, X)
    ("time", "band", "y", "x"),
    {"band": ("B02", "B03", "B04", "B08")},
)

# today (pipekit-array)
nir = pa.sel(cube, band="B08")                         # (T, B, Y, X) → (T, Y, X)
# today (pipekit-array)
red = pa.sel(cube, band="B04")                         # (T, B, Y, X) → (T, Y, X)


def normalized_difference(a, b, eps):
    """The L1 kernel: plain arrays in, a plain array out."""
    # v = (a − b) / (a + b + ε)
    return (a - b) / (a + b + eps)


# today (pipekit-array); normalized_difference is user
ndvi_op = pa.KernelOp(normalized_difference, "(),()->()", core={}, eps=1e-6)
ndvi = ndvi_op(nir, red)                               # (T, Y, X), (T, Y, X) → (T, Y, X)

# today (pipekit-array)
# w_{n,i,j,r} = v_{n+r,i,j}, 0 ≤ r < S
windows = pa.window(ndvi, time=3)                      # (T, Y, X) → (T−S+1, Y, X, S)
# today (pipekit-array)
# v̄_{n,i,j} = (1/S) ∑ᵣ w_{n,i,j,r}
smooth = pa.reduce(windows, "mean", "window")          # (T−S+1, Y, X, S) → (T−S+1, Y, X)

# today (pipekit-array)
# s_{q,i,j} = (1/K) ∑ₖ v_{Kq+k,i,j}
seasons = pa.coarsen(ndvi, "mean", time=3)             # (T, Y, X) → (T/K, Y, X)
# today (pipekit-array)
# Δs_{q,i,j} = s_{q+1,i,j} − s_{q,i,j}
change = pa.diff(seasons, "time")                      # (T/K, Y, X) → (T/K−1, Y, X)
# today (pipekit-array)
# p_{i,j} = quantile₀.₉ {v_{t,i,j} : t}
p90 = pa.quantile(ndvi, 0.9, "time")                   # (T, Y, X) → (Y, X)
```

Nothing above names an axis by number.
If the cube arrives as `(band, time, y, x)`, every line still does the same thing.

### 2. The operator twins, on four carriers and under `jit`

Each dim function has an operator twin with a JSON config, so the steps chain with `|` and round-trip.
Here the per-band seasonal change, averaged over the window, runs on NumPy, JAX, xarray and a jitted JAX call.

```python
import jax
import jax.numpy as jnp
import numpy as np

import pipekit_array as pa
import pipekit_ops as pk

# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels; K = 3 months per season

rng = np.random.default_rng(0)
reflectance = rng.uniform(0.0, 0.4, size=(12, 4, 64, 64)).astype(np.float32)
bands = {"band": ("B02", "B03", "B04", "B08")}
dims = ("time", "band", "y", "x")

# today (pipekit-array)
# s_{q,b,i,j} = (1/K) ∑ₖ x_{Kq+k,b,i,j}
seasonal = pa.Coarsen("mean", time=3)
# today (pipekit-array)
# Δs_{q,b,i,j} = s_{q+1,b,i,j} − s_{q,b,i,j}
change = pa.Diff("time")
# today (pipekit-array)
# m_{q,b} = (1/(Y X)) ∑ᵢ ∑ⱼ Δs_{q,b,i,j}
scene_mean = pa.Reduce("mean", ("y", "x"))
# today (pipekit-ops)
seasonal_change = seasonal | change | scene_mean

# today (pipekit-array)
on_numpy = pa.DimArray(reflectance, dims, bands)       # (T, B, Y, X)
# today (pipekit-array)
on_jax = pa.DimArray(jnp.asarray(reflectance), dims, bands)
# today (pipekit-array)
on_xarray = on_numpy.to_xarray()

a = seasonal_change(on_numpy)                          # (T, B, Y, X) → (T/K−1, B)
b = seasonal_change(on_jax)                            # (T, B, Y, X) → (T/K−1, B)
c = seasonal_change(on_xarray)                         # (T, B, Y, X) → (T/K−1, B)
d = jax.jit(seasonal_change)(on_jax)                   # (T, B, Y, X) → (T/K−1, B)

# today (pipekit-array)
# m_{t,b} = (1/(Y X)) ∑ᵢ ∑ⱼ x_{t,b,i,j}, one scene per call
per_scene = pa.ApplyAlong(scene_mean, dim="time")
# today (pipekit-array)
# the same, five scenes per call; the short last chunk is padded, then trimmed
per_chunk = pa.BatchAlong(scene_mean, size=5, dim="time", pad="last")
e = per_scene(on_numpy)                                # (T, B, Y, X) → (T, B)
f = per_chunk(on_numpy)                                # (T, B, Y, X) → (T, B)

# today (pipekit-ops)
again = pk.loads(pk.dumps(seasonal_change))
assert pk.pipeline_hash(again) == pk.pipeline_hash(seasonal_change)
```

`a` and `b` hold the same numbers on NumPy and on JAX, `c` is an `xr.DataArray` with the band labels as a coordinate, and `e` equals `f`.

### 3. A scikit-learn estimator over named dims

```python
import numpy as np
from sklearn.decomposition import PCA

import pipekit_array as pa

# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels; C = 2 components

rng = np.random.default_rng(0)
reflectance = rng.uniform(0.0, 0.4, size=(12, 4, 64, 64)).astype(np.float32)
# today (pipekit-array)
cube = pa.DimArray(reflectance, ("time", "band", "y", "x"))

# today (pipekit-array)
# every (t, i, j) is a sample and every band a feature: z = Wᵀ (x − μ)
pca = pa.Estimator(PCA(2), sample_dims=("time", "y", "x"), feature_dim="component")
scores = pca.fit(cube).transform(cube)                 # (T, B, Y, X) → (T, C, Y, X)
```

The estimator sees a `(T Y X, B)` table, and the result comes back with the band dim replaced by `component`, in place.

## What ships

| Module | Public names | Purpose |
| --- | --- | --- |
| `_namespace` | `array_namespace` | the Array API namespace shared by the inputs; the one point of backend dispatch |
| `reductions` | `MeanScalar` | a mean over positional axes, on raw arrays |
| `combinators` | `StackAlong`, `ConcatenateAlong` | join a list of arrays (`axis=`) or named carriers (`dim=`) |
| `dimarray` | `DimArray`, `DimArrayAdapter` | the plain named carrier (data, dims, static labels) and its reference adapter |
| `dims` | `isel`, `sel`, `transpose`, `reduce`, `argreduce`, `take_along`, `map_over`, `stack`, `stack_items`, `concat`, `combine`, `BackendError` | select, reduce, map and join by dim name |
| `dim_shapes` | `rename`, `expand_dims`, `squeeze`, `broadcast_like`, `merge_dims`, `split_dims` | change which dims a carrier has |
| `dim_shifts` | `pad`, `roll`, `shift`, `diff`, `cumreduce`, `where` | move, difference and mask along a dim |
| `dim_windows` | `window`, `coarsen` | sliding windows and non-overlapping blocks |
| `dim_algebra` | `quantile`, `dot` | quantiles and contractions over named dims |
| `dim_operators` | `ISel`, `Sel`, `Transpose`, `Reduce`, `ArgReduce`, `TakeAlong`, `MapOver`, `Combine`, `Rename`, `ExpandDims`, `Squeeze`, `BroadcastLike`, `MergeDims`, `SplitDims`, `Pad`, `Roll`, `Shift`, `Diff`, `CumReduce`, `Where`, `Window`, `Coarsen`, `Quantile`, `Dot`, `ApplyAlong` | the operator twin of each dim function, plus `ApplyAlong` |
| `batching` | `batch`, `Padded`, `BatchAlong` | fixed-size batches with a validity mask; an operator on chunks of a dim |
| `rows` | `to_rows`, `from_rows`, `RowLayout` | a carrier to a `(samples, features)` table by dim name, and back |
| `estimator` | `Estimator` | a scikit-learn-style estimator fitted and applied on a named carrier (`[sklearn]`) |
| `kernel` | `KernelSpec`, `KernelOp` | one array kernel, written once, run on raw arrays and every named carrier |
| `einx` | `rearrange`, `reduce`, `einsum`, `repeat`, `Einx` | einx with dim names as axes (`[einx]`; `from pipekit_array import einx`) |
| `xarray` | the `xr.DataArray` adapter | loaded the first time a DataArray reaches `pipekit_ops.get_adapter` (`[xarray]`) |
| `testing` | `check_adapter`, `check_kernel_op` | the test kit for carrier libraries that register their own adapter |

`pipekit_array.__all__` has 68 names; `einx`, `xarray` and `testing` are imported as submodules.

### Not built yet

From the v0.1 catalogue in the [API design](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/api/README.md):

- **QC:** `AssertClose`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction`;
- **Observation:** the array-aware `Histogram` controller;
- **Inference:** `ModelOp`, the framework-agnostic trained-model wrapper ([ADR A8](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/decisions.md#a8-modelop-split-pipekit-array-vs-pipekit-jax)).

Three planned names were retired instead of built.
`Subsample` is `isel` with a stepped slice, and `ApplyToBands` became `ApplyAlong` ([ADR A15](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/decisions.md#a11-to-a22-the-missing-dim-level-functions)).
The array-shaped `BatchedMap` became `BatchAlong`, so it no longer shares a name with `pipekit_ops.BatchedMap` (ADR A19, which supersedes A5).

## How it works

### Array API dispatch

Every function asks its input for its namespace, `array_namespace(x)`, and calls `xp.mean`, `xp.stack` and so on from it.
So one operator runs on NumPy, JAX, CuPy, PyTorch or dask without importing any of them, and the result stays on the backend it came from.
Inputs that span two backends raise a `TypeError` instead of being moved silently.
PyTorch tensors have no native `__array_namespace__`, so they need the `[compat]` extra.

### Named dims through the adapter protocol

pipekit-array never needs to know a carrier's class.
It asks pipekit-ops for the carrier's adapter (`pipekit_ops.get_adapter`), and the adapter answers a small protocol, `NamedArrayAdapter`: the dims, the raw data, the labels of a dim, `isel`, `rewrap` (new data and dims, same carrier type) and the `Signature`.
`LabelledAdapter` adds an optional `relabel`, which the functions that create a labelled dim need (`quantile` with a list of probabilities, for one).

- `DimArray` is the plain carrier we ship: a raw array, one name per axis and static labels on some dims, with no georeference, coordinates or alignment. `DimArray.to_xarray()` crosses over when an analysis needs those.
- `xr.DataArray` gets its adapter lazily, the first time one reaches `get_adapter`, so importing pipekit-array never imports xarray. Several functions take the adapter's fast path (a `window` on a DataArray is a view, not a copy).
- Your own carrier works once you call `pipekit_ops.register_adapter(MyCarrier, MyAdapter())`; `pipekit_array.testing.check_adapter` compares it against the `DimArray` reference.

### Dim functions and their operator twins

The functions come in four groups: shapes (`rename`, `expand_dims`, `squeeze`, `merge_dims`, `split_dims`, `broadcast_like`), shifts (`pad`, `roll`, `shift`, `diff`, `cumreduce`, `where`), windows (`window`, `coarsen`) and algebra (`quantile`, `dot`), on top of the core `isel`, `sel`, `reduce` and friends.
Each has an operator twin with the same arguments (`window` and `Window`, `coarsen` and `Coarsen`), which takes a reduction by name (`"mean"`, `"nanmax"`) so that its config is JSON.
The twins also implement `compute_output_signature`, so `Sequential.summary(signature)` prints every intermediate shape before anything runs.

**Note**: `pipekit_array.Reduce` reduces over named dims, while `pipekit_ops.Reduce` is a left fold over a list.
Import the packages under their own names (`pa.Reduce`, `pk.Reduce`) when you use both.

### `ApplyAlong` and `BatchAlong`

`ApplyAlong(inner, dim)` runs an operator on each item along a dim and stacks the results back along it, first.
It is `map_over` with an operator instead of a function, so it serialises whenever `inner` does.
`BatchAlong(inner, size, dim, pad=...)` cuts one carrier into chunks of `size` along a dim, which is what you want for a GPU model on a cube that does not fit in device memory.
With `pad="last"` the short last chunk is padded to `size` and trimmed afterwards, so a jitted `inner` compiles once.
`BatchAlong` is not `pipekit_ops.BatchedMap`, which batches a list of items.

### `KernelOp` and `Estimator`

`KernelOp(kernel, spec, core=...)` wraps a plain array function once and runs it on raw arrays and on every named carrier.
The `spec` is a generalised-ufunc signature (`"(),()->()"` element-wise, `"(c)->()"` consuming one dim), and `core` maps each core letter to a carrier dim, so one kernel serves `"band"` and `"variable"`.
Because it holds a user function, its config is a debug record only (`forbid_in_yaml`), like `pipekit_ops.Lambda`; a named operator built on it is what serialises.

`Estimator(estimator, sample_dims=..., feature_dim=...)` flattens a carrier to a sample table with `to_rows`, fits or applies any scikit-learn-style estimator, and rebuilds the carrier with `from_rows`.
It clones the estimator on `fit`, offers NaN policies (`"propagate"`, `"raise"`, `"mask_samples"`, `"mask_features"`, `"mask"`) and runs on NumPy and dask data.

### `jit` and `vmap` safety

`DimArray` is a JAX pytree whose only leaf is the data; dims and labels are static, so a new set of values never recompiles.
The dim functions are plain functions on that pytree, so they trace under `jax.jit` and `jax.vmap` (gates G4 and G5 of the [dim-function design](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/api/dim_functions.md)), and a `Sequential` of their operator twins runs under `jax.jit` too, as block 2 shows.
Arguments that set a shape (`n` in `diff`, `q` in `quantile`, a window size) are static under `jit`.
Two gotchas:

- a bare `jax.vmap` maps the first dim, drops it inside, and gives it back as `"batch"`; to map a named dim, use `map_over` or `ApplyAlong`;
- `Estimator.fit` runs scikit-learn on the host, so it raises a `TypeError` inside a JAX trace; fit it eagerly.

## Learn more

- The [design docs](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array/docs/design): vision, [architecture](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/architecture.md), [decisions](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/decisions.md), boundaries and the per-operator specs.
- The master plan, [Report 3, "Sister libraries"](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_3_pipekit_array.md).
- The [API reference](https://jejjohnson.github.io/pipekit/api/pipekit-array/).
- [Concepts: backend dispatch](https://github.com/jejjohnson/pipekit/blob/main/docs/concepts.md#5-backend-dispatch-pipekit-array) and the tutorial [Run pipekit pipelines under `jit`](https://github.com/jejjohnson/pipekit/blob/main/docs/tutorials/pipekit-under-jit.md).
- The sibling foundation, [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops), and the rest of the [family](https://github.com/jejjohnson/pipekit#the-packages).

## License

MIT; see [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/LICENSE).
