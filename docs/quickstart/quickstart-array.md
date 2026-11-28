# pipekit-array quickstart

The [README quickstart](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array#readme) computes the NDVI of a year of Sentinel-2 over one tile with the dim functions, and checks that a transposed cube gives the same answer. This page takes the same cube further: the operator twins chained on four carriers and under `jit`, the shapes of a chain before any data arrives, an operator along a dim and in chunks, a scikit-learn estimator over named dims, and a formula on plain arrays. Then it explains how the dispatch, the adapters and the twins work, what to install for each backend, and what is not built yet.

The blocks run in order, as one script, so a later block uses the names of an earlier one.

## The cube and its symbols

The cube has twelve monthly scenes, four bands (B02, B03, B04, B08) and a 64 by 64 pixel window. The reflectance of band $b$ at pixel $(i, j)$ and time $t$ is $x_{t,b,i,j}$.

| Symbol | Meaning | Dim |
|---|---|---|
| $T$ | the number of monthly scenes | `time` |
| $B$ | the number of bands | `band` |
| $Y$, $X$ | the rows and columns of pixels | `y`, `x` |
| $S$ | the months in a sliding window | `window` |
| $K$ | the months in a season | |
| $C$ | the components of a PCA | `component` |
| $x_{t,b,i,j}$ | the reflectance | |
| $v_{t,i,j}$ | the NDVI | |
| $s_{q,b,i,j}$ | the mean of season $q$ | |
| $\Delta s_{q,b,i,j}$ | its change from season $q$ to season $q + 1$ | |
| $m_{q,b}$ | a value averaged over the pixels | |

The NDVI per scene and pixel is the normalised difference of the near-infrared and red bands,

$$v_{t,i,j} = \frac{x_{t,\text{B08},i,j} - x_{t,\text{B04},i,j}}{x_{t,\text{B08},i,j} + x_{t,\text{B04},i,j} + \epsilon}$$ (eq-array-ndvi)

where $\epsilon = 10^{-6}$ keeps the ratio finite over dark water. The cube itself holds random reflectances between 0 and 0.4:

```python
import numpy as np
import pipekit_array as pa


# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels

# x_{t,b,i,j} ~ U(0, 0.4)
# → (T, B, Y, X)
rng = np.random.default_rng(0)
reflectance = rng.uniform(0.0, 0.4, size=(12, 4, 64, 64)).astype(np.float32)
bands = {"band": ("B02", "B03", "B04", "B08")}
dims = ("time", "band", "y", "x")

# today (pipekit-array)
# (T, B, Y, X), with one label per band
cube = pa.DimArray(reflectance, dims, bands)
print(cube.dims, cube.shape)
```

This prints `('time', 'band', 'y', 'x') (12, 4, 64, 64)`.

## The operator twins, on four carriers and under `jit`

Each dim function has an operator twin with a JSON config, so the steps chain with `|` and round-trip. Here the change of each band from one season to the next, $\Delta s_{q,b,i,j} = s_{q+1,b,i,j} - s_{q,b,i,j}$ with $s_{q,b,i,j} = \frac{1}{K} \sum_{k} x_{Kq+k,b,i,j}$, is averaged over the pixels into $m_{q,b}$, and the one chain runs on NumPy, on JAX, on an `xr.DataArray` and in a jitted JAX call.

```python
import jax
import jax.numpy as jnp
import pipekit_ops as pk


# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels;
#         K = 3 months per season

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

# (T, B, Y, X), the same cube on JAX
on_jax = pa.DimArray(jnp.asarray(reflectance), dims, bands)
# today (pipekit-array)
# (T, B, Y, X), the same cube as an xr.DataArray
on_xarray = cube.to_xarray()

# (T, B, Y, X) → (T/K−1, B)
a = seasonal_change(cube)
# (T, B, Y, X) → (T/K−1, B)
b = seasonal_change(on_jax)
# (T, B, Y, X) → (T/K−1, B)
c = seasonal_change(on_xarray)
# (T, B, Y, X) → (T/K−1, B)
d = jax.jit(seasonal_change)(on_jax)

# today (pipekit-ops)
again = pk.loads(pk.dumps(seasonal_change))
same_hash = pk.pipeline_hash(again) == pk.pipeline_hash(seasonal_change)

# (T/K−1, B) each, as NumPy arrays
others = [np.asarray(r) for r in (b.data, c.values, d.data)]
agree = all(np.allclose(a.data, r, atol=1e-6) for r in others)
labels = c.coords["band"].values.tolist()
print(a.dims, a.shape, type(c).__name__, labels, agree, same_hash)
```

This prints `('time', 'band') (3, 4) DataArray ['B02', 'B03', 'B04', 'B08'] True True`: three changes between the four seasons for each of the four bands, the same numbers on NumPy, on JAX, on xarray and under `jit` (to within 1e-6), the DataArray result with the band labels as a coordinate, and a chain that comes back from JSON with the same `pipeline_hash`.

## The shapes before the data

The twins implement `compute_output_signature`, so `Sequential.summary` prints every intermediate shape from a `Signature` alone, before anything runs.

```python
from pipekit_ops import Signature


# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels

# today (pipekit-ops)
# the cube's dims, sizes and dtype, without its data
sizes = (("time", 12), ("band", 4), ("y", 64), ("x", 64))
signature = Signature(dims=sizes, dtype="float32")
print(seasonal_change.summary(signature))
```

This prints

```text
Operator     Input sig                                         Output sig
--------------------------------------------------------------------------------------------------------------
[0] Coarsen  Signature((time=12, band=4, y=64, x=64) float32)  Signature((time=4, band=4, y=64, x=64) float32)
[1] Diff     Signature((time=4, band=4, y=64, x=64) float32)   Signature((time=3, band=4, y=64, x=64) float32)
[2] Reduce   Signature((time=3, band=4, y=64, x=64) float32)   Signature((time=3, band=4) float32)
```

that is, four seasons from twelve months, three changes from four seasons, and the pixels averaged away. [Concepts](../concepts.md) has more on shape inference, and on the `None` a step returns when it cannot track a shape.

## An operator along a dim, and in chunks

`ApplyAlong` runs an operator on each item along a dim and stacks the results back along it, and `BatchAlong` cuts the carrier into chunks of a given size along a dim, which is what a GPU model on a cube larger than device memory needs. Here both average each scene over its pixels, $m_{t,b} = \frac{1}{Y X} \sum_{i} \sum_{j} x_{t,b,i,j}$, one scene per call and five scenes per call.

```python
# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels

# today (pipekit-array)
# m_{t,b} = (1/(Y X)) ∑ᵢ ∑ⱼ x_{t,b,i,j}, one scene per call
per_scene = pa.ApplyAlong(scene_mean, dim="time")
# today (pipekit-array)
# the same, five scenes per call; the short last chunk is padded, then trimmed
per_chunk = pa.BatchAlong(scene_mean, size=5, dim="time", pad="last")
# (T, B, Y, X) → (T, B)
e = per_scene(cube)
# (T, B, Y, X) → (T, B)
f = per_chunk(cube)
print(e.dims, e.shape, bool(np.allclose(e.data, f.data)))
```

This prints `('time', 'band') (12, 4) True`: one mean per scene and band, the same from both. The twelve scenes make chunks of five, five and two, and with `pad="last"` the last chunk is padded to five and trimmed afterwards, so a jitted inner operator compiles once.

## A scikit-learn estimator over named dims

`Estimator` treats every combination of the sample dims as one row of a table and the remaining dim as its features, fits or applies a scikit-learn-style estimator on that table, and rebuilds the carrier. Here every pixel of every scene is a sample $\mathbf{x} \in \mathbb{R}^{B}$ (its four reflectances), and the PCA maps it to $\mathbf{z} = \mathbf{W}^\top (\mathbf{x} - \boldsymbol{\mu})$, with $\boldsymbol{\mu} \in \mathbb{R}^{B}$ the mean over the samples and $\mathbf{W} \in \mathbb{R}^{B \times C}$ the first $C$ principal directions.

```python
from sklearn.decomposition import PCA


# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels; C = 2 components

# today (pipekit-array)
# every (t, i, j) is a sample and every band a feature: z = Wᵀ (x − μ)
pca = pa.Estimator(PCA(2), sample_dims=("time", "y", "x"), feature_dim="component")
# (T, B, Y, X) → (T, C, Y, X)
scores = pca.fit(cube).transform(cube)
print(scores.dims, scores.shape)
```

This prints `('time', 'component', 'y', 'x') (12, 2, 64, 64)`. The estimator saw a table of $T Y X$ rows and $B$ columns, and the result comes back with the band dim replaced by `component`, in place.

## A formula on plain arrays

A library of formulas (geophysix, radiomax, diagnostix) has no dims to name, but it still wants one function for every array library. `pipekit_array.arrays` gives it the namespace of a call and inputs in one floating dtype (`floating_inputs`), and `elementwise` lets the same function take named carriers too. The wind speed $\lvert \mathbf{u} \rvert = \sqrt{u_e^2 + u_n^2}$, from the eastward and northward components $u_e$ and $u_n$, at three stations:

```python
import jax.numpy as jnp
import numpy as np
from pipekit_array import DimArray
from pipekit_array.arrays import floating_inputs, safe_sqrt
from pipekit_array.kernels import elementwise


# Shapes: N_s = 3 stations


# user
# |u| = √(u_e² + u_n²), the wind speed from its eastward and northward components
@elementwise()
def wind_speed(eastward, northward):
    # today (pipekit-array)
    # the namespace of the call, and both inputs in one floating dtype
    xp, (east, north) = floating_inputs(eastward, northward)
    # u_e² + u_n²
    squared = east * east + north * north
    # today (pipekit-array)
    # √(u_e² + u_n²), and 0 with a zero gradient where the sum rounds below 0
    return safe_sqrt(squared, xp)


# → (N_s,), on NumPy
on_numpy = wind_speed(np.asarray([3.0, 0.0, 6.0]), np.asarray([4.0, 0.0, 8.0]))
# → (N_s,), on JAX, with a Python list joining it
on_jax = wind_speed(jnp.asarray([3.0, 0.0, 6.0]), [4.0, 0.0, 8.0])
# today (pipekit-array)
# → (N_s,), on a carrier with dim names, with a constant northward component
on_carrier = wind_speed(DimArray(np.asarray([3.0, 0.0, 6.0]), ("station",)), 4.0)
print(on_numpy, type(on_jax).__name__, on_carrier.dims, on_carrier.data)
```

This prints `[ 5.  0. 10.] ArrayImpl ('station',) [5.         4.         7.21110255]`: the speeds on NumPy, a JAX array when one input is on JAX (the list joins it), and a `DimArray` with its `station` dim when one input is a carrier. `safe_sqrt` returns 0 with a zero gradient at 0, where $\sqrt{\cdot}$ has an infinite one, and NaN for NaN. The block needs the `numpy,compat` extras.

## How it works

### Array API dispatch

Every function asks its input for its namespace, `array_namespace(x)`, and calls `xp.mean`, `xp.stack` and so on from it. So one operator runs on NumPy, JAX, CuPy, PyTorch or dask without importing any of them, and the result stays on the backend it came from. A NumPy input next to a JAX one joins it (ADR A24), and `pipekit_ops.guard` reports each such join on request; any other mix of backends raises `BackendError`, which is both a `TypeError` and a `ValueError`. PyTorch tensors have no native `__array_namespace__`, so they need the `compat` extra.

The tests run the named layer on NumPy, JAX, dask and xarray (and on `array_api_strict` where it is installed); the CI job that installs PyTorch runs the plain-array functions and `elementwise` on it, and no CI job has CuPy, so CuPy goes through the standard untested.

### Named dims through the adapter protocol

pipekit-array never needs to know a carrier's class. It asks pipekit-ops for the carrier's adapter (`pipekit_ops.get_adapter`), and the adapter answers a small protocol, `NamedArrayAdapter`: the dims, the raw data, the labels of a dim, `isel`, `rewrap` (new data and dims, same carrier type) and the `Signature`. `LabelledAdapter` adds an optional `relabel`, which the functions that create a labelled dim need (`quantile` with a list of probabilities, for one).

- `DimArray` is the plain carrier we ship: a raw array, one name per axis and static labels on some dims, with no georeference, coordinates or alignment. `DimArray.to_xarray()` crosses over when an analysis needs those.
- `xr.DataArray` gets its adapter lazily, the first time one reaches `get_adapter`, so importing pipekit-array never imports xarray. Several functions take the adapter's fast path through xarray's own calls, which stay lazy on dask. A `window` on a DataArray is xarray's `rolling(...).construct`: the windows are a strided view of one padded copy of the input, not a copy per window.
- Your own carrier works once you call `pipekit_ops.register_adapter(MyCarrier, MyAdapter())`; `pipekit_array.testing.check_adapter` compares it against the `DimArray` reference.

### Dim functions and their operator twins

The functions come in four groups: shapes (`rename`, `expand_dims`, `squeeze`, `merge_dims`, `split_dims`, `broadcast_like`), shifts (`pad`, `roll`, `shift`, `diff`, `cumreduce`, `where`), windows (`window`, `coarsen`) and algebra (`quantile`, `dot`), on top of the core `isel`, `sel`, `reduce` and friends. Each has an operator twin with the same arguments (`window` and `Window`, `coarsen` and `Coarsen`), which takes a reduction by name (`"mean"`, `"nanmax"`) so that its config is JSON. The twins also implement `compute_output_signature`, as the summary above shows.

:::{note}
`pipekit_array.Reduce` reduces over named dims, while `pipekit_ops.Reduce` is a left fold over a list. Import the packages under their own names (`pa.Reduce`, `pk.Reduce`) when you use both.
:::

### `ApplyAlong` and `BatchAlong`

`ApplyAlong(inner, dim)` runs an operator on each item along a dim and stacks the results back along it, first. It is `map_over` with an operator instead of a function, so it serialises whenever `inner` does. `BatchAlong(inner, size, dim, pad=...)` cuts one carrier into chunks of `size` along a dim, and with `pad="last"` the short last chunk is padded to `size` and trimmed afterwards, so a jitted `inner` compiles once. `BatchAlong` is not `pipekit_ops.BatchedMap`, which batches a list of items.

### `KernelOp` and `Estimator`

`KernelOp(kernel, spec, core=...)` wraps a plain array function once and runs it on raw arrays and on every named carrier. The `spec` is a generalised-ufunc signature (`"(),()->()"` element-wise, `"(c)->()"` consuming one dim), and `core` maps each core letter to a carrier dim, so one kernel serves `"band"` and `"variable"`. Every output has the loop dims of all the inputs, in the first input's order, and on a DataArray the call is one `xr.apply_ufunc`, lazy on dask. Because it holds a user function, its config is a debug record only (`forbid_in_yaml`), like `pipekit_ops.Lambda`; a named operator built on it is what serialises.

`Estimator(estimator, sample_dims=..., feature_dim=...)` flattens a carrier to a sample table with `to_rows`, fits or applies any scikit-learn-style estimator, and rebuilds the carrier with `from_rows`. It clones the estimator on `fit` and offers NaN policies (`"propagate"`, `"raise"`, `"mask_samples"`, `"mask_features"`, `"mask"`). Fitting and applying run on the host, since scikit-learn computes on NumPy: a dask input is computed and a JAX input converted, eagerly.

### `jit` and `vmap` safety

`DimArray` is a JAX pytree whose only leaf is the data; dims and labels are static, so a new set of values never recompiles. The dim functions are plain functions on that pytree, so they trace under `jax.jit` and `jax.vmap` (gates G4 and G5 of the [dim-function design](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/api/dim_functions.md)), and a `Sequential` of their operator twins runs under `jax.jit` too, as the block on four carriers shows. Arguments that set a shape (`n` in `diff`, `q` in `quantile`, a window size) are static under `jit`. Two gotchas:

- a bare `jax.vmap` maps the first dim, drops it inside, and gives it back as `"batch"`; to map a named dim, use `map_over` or `ApplyAlong`;
- `Estimator.fit` runs scikit-learn on the host, so it raises a `TypeError` inside a JAX trace; fit it eagerly.

The tutorial [Pipelines under jit](../tutorials/pipekit-under-jit.md) covers JAX operators in a pipeline more generally.

## Install and extras

pipekit-array is useful without the rest of pipekit. It depends only on pipekit-ops (pure Python, no dependencies), and every backend is an opt-in extra, so `pipekit-array[numpy]` is a small install. The dim functions (`window`, `coarsen`, `quantile`, `diff`, …) are plain functions: you can call them in a notebook on a `DimArray` or a DataArray and never write an operator. The [README](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array#readme) has the install commands from git and the `[tool.uv.sources]` block.

| Extra | What it adds | When you need it |
|---|---|---|
| `numpy` | NumPy 2 | arrays on the CPU; the quickstart |
| `jax` | JAX | arrays on a GPU or TPU, and `jit` / `vmap` |
| `torch` | PyTorch | PyTorch tensors; add `compat` too |
| `cupy` | CuPy | NumPy-style arrays on a CUDA GPU |
| `dask` | `dask[array]` | lazy, chunked arrays larger than memory |
| `compat` | array-api-compat | backends without a native `__array_namespace__` (PyTorch); with `numpy`, `pipekit_array.arrays` and `elementwise` |
| `xarray` | xarray | the `xr.DataArray` adapter and `DimArray.to_xarray` |
| `sklearn` | scikit-learn | `Estimator` |
| `einx` | einx | `pipekit_array.bridges.einx`: einx patterns with dim names as axes |

`import pipekit_array` imports neither NumPy nor xarray, and its `__all__` has 68 names: those of `named`, `kernels` and `operators` that a pipeline uses, `Estimator` and `BackendError`. `pipekit_array.arrays` and `pipekit_array.kernels.elementwise` need `numpy,compat`, and `pipekit_array.testing` needs NumPy. In this workspace, `pipekit-cycle` and `pipekit-jax` add pipekit-array with their `array` extras, and `pipekit-jax` turns a `KernelOp` into a JAX operator with `to_jax`.

## Not built yet

From the v0.1 catalogue in the [API design](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/api/README.md):

- **QC:** `AssertClose`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction`;
- **Observation:** the array-aware `Histogram` controller;
- **Inference:** `ModelOp`, the framework-agnostic trained-model wrapper ([ADR A8](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/decisions.md#a8-modelop-split-pipekit-array-vs-pipekit-jax)).

Three planned names were retired instead of built. `Subsample` is `isel` with a stepped slice, and `ApplyToBands` became `ApplyAlong` ([ADR A15](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/decisions.md#a11-to-a22-the-missing-dim-level-functions)). The array-shaped `BatchedMap` became `BatchAlong`, so it no longer shares a name with `pipekit_ops.BatchedMap` (ADR A19, which supersedes A5).

A pipeline saved before the package was split into areas still loads: an operator is recorded under `pipekit_array` wherever its file is, and `__former_modules__` maps the states saved under the old modules (ADR A30).

## Learn more

- The [tour of the six areas](../tour-array.md): every area of the package, with the main names at work.
- The [design docs](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array/docs/design): vision, [architecture](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/architecture.md), [decisions](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/decisions.md), boundaries and the per-operator specs.
- The master plan, [Report 3, "Sister libraries"](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_3_pipekit_array.md).
- The [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-array/).
- [Concepts](../concepts.md), section 5 (backend dispatch).
- The sibling foundation, [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops), and the rest of the [family](https://github.com/jejjohnson/pipekit#the-packages).
