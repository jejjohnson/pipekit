<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-array/logo-dark.svg">
  <img alt="pipekit-array" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-array/logo-light.svg" width="386">
</picture>

**Named-dim array operators on any Array API backend, written once by dim name and run on NumPy, JAX or dask data and on an xarray DataArray: on pipekit-ops, the Python Array API standard and array-api-compat, with bridges to xarray, scikit-learn and einx.**

[![Tests](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pipekit/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pipekit/)
[![codecov](https://codecov.io/gh/jejjohnson/pipekit/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pipekit)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jejjohnson/pipekit/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

📖 [Documentation](https://jejjohnson.github.io/pipekit/) ·
🚀 [Quickstart](https://jejjohnson.github.io/pipekit/quickstart-array/) ·
📚 [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-array/) ·
🧭 [Design](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array/docs/design)

</div>

---

Take a year of Sentinel-2 over one tile: twelve monthly scenes, four bands (B02, B03, B04, B08) and a 64 by 64 pixel window, so a cube with dims `(time, band, y, x)`. The questions we ask of it are phrased by dim name (the NDVI per scene, a three-month running mean, the seasonal composites, the change from one season to the next, the 90th percentile per pixel). In code they usually turn into `axis=1` and `axis=(2, 3)`, which silently go wrong the day someone transposes the cube, and into one version for NumPy on a laptop and another for JAX on a GPU. So pipekit-array writes each step once, by dim name, against the [Python Array API standard](https://data-apis.org/array-api/), and treats an operation as a function of the dims it consumes, applied the same way at every index of the others,

```math
\mathbf{y}_\ell = f\left(\mathbf{x}^{(1)}_\ell, \dots, \mathbf{x}^{(N_\text{in})}_\ell\right)
```

where the $`N_\text{in}`$ inputs $`\mathbf{x}^{(k)}`$ are named carriers and $`\ell`$ runs over every index of their loop dims (the dims the operation does not consume, matched across the inputs by name). At each $`\ell`$, $`\mathbf{x}^{(k)}_\ell \in \mathbb{R}^{N_{\mathcal{C}_k}}`$ holds the values of input $`k`$ over its core dims $`\mathcal{C}_k`$ (a scalar when it has none), and $`\mathbf{y}_\ell`$ is the output over the output's core dims; $`f`$ is a function of plain arrays, written once. No index in it is a position, so transposing an input changes nothing, and $`f`$ takes its array functions from the namespace of its inputs, so the same $`f`$ runs on whichever library holds the data. The reductions, windows and differences of `pipekit_array.named` are this form with $`f`$ fixed (`reduce` consumes its dims, `window` and `diff` change the length of one, `dot` contracts one across two inputs), and `KernelOp` takes $`f`$ from the caller.

The sketch chains three operator twins (the seasonal mean, its change from one season to the next, and the mean of that change over the pixels) and runs the one chain on three carriers.

```python
# Shapes: T monthly scenes; B bands; Y, X pixels; K months per season
# (pa, jax, jnp and the cube as in the quickstart below)

# today (pipekit-array)
# the seasonal change of each band, averaged over the pixels
seasonal = pa.Coarsen("mean", time=3) | pa.Diff("time") | pa.Reduce("mean", ("y", "x"))
# (T, B, Y, X) → (T/K−1, B), the data on NumPy
on_numpy = seasonal(cube)
# (T, B, Y, X) → (T/K−1, B), the data on JAX, under jit
on_jax = jax.jit(seasonal)(pa.DimArray(jnp.asarray(cube.data), cube.dims, cube.labels))
# (T, B, Y, X) → (T/K−1, B), an xr.DataArray with the band labels as a coordinate
on_xarray = seasonal(cube.to_xarray())
```

None of it is a new array library. The arithmetic is the namespace's own (`xp.mean`, `xp.stack`), read from the data through the Python Array API standard. [array-api-compat](https://github.com/data-apis/array-api-compat) supplies the namespace of a library that has no native one (PyTorch), and it is the base of the plain-array functions of `pipekit_array.arrays`. [NumPy](https://numpy.org) and [JAX](https://github.com/jax-ml/jax) arrays pass through as they are, and a `DimArray` is a JAX pytree whose only leaf is the data, so `jax.jit` and `jax.vmap` trace through it. Every operator is a [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) `Operator`, so it chains with `|`, round-trips through JSON, has a `pipeline_hash` and reports its output `Signature`, and a carrier is found through the pipekit-ops adapter registry. The bridges hand each job to the library that owns it: [xarray](https://github.com/pydata/xarray) for a DataArray (its own `rolling`, `apply_ufunc` and dask laziness), [einx](https://github.com/fferflo/einx) for patterns whose axes are dim names, and [scikit-learn](https://scikit-learn.org) for any estimator over named dims.

pipekit-array is one of the two foundations of the pipekit workspace, next to pipekit-ops; the [workspace README](https://github.com/jejjohnson/pipekit#readme) introduces the other packages.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_array_overview/pipekit_array_overview.png" alt="pipekit-array in one picture: you supply data on a backend, a carrier with dim names, the dims to consume or your own function; one rule y_ℓ = f(x⁽¹⁾_ℓ, …) by dim name; six areas (named, kernels, operators, bridges, arrays and testing); out comes the same carrier on the same backend, its signature, and a pipekit pipeline; on pipekit-ops, the Array API and its extras" width="100%"></p>

## 🌟 Highlights

- 🏷️ **Dims by name, never by position**: `sel`, `isel`, `reduce`, `quantile`, `window`, `coarsen`, `diff`, `shift`, `merge_dims`, `stack`, `map_over` and the rest of `pipekit_array.named` take dim names, on a `DimArray`, an `xr.DataArray` or any carrier whose library registers an adapter with pipekit-ops.
- 🔁 **The backend the data is on**: every call reads its namespace from its inputs (`array_namespace`), so the result stays on NumPy, JAX or dask; a NumPy constant next to JAX data joins it, and any other mix raises `BackendError`. The tests run on NumPy, JAX, dask and xarray, and the plain-array functions also on PyTorch.
- 🧱 **Operator twins**: each dim function has an `Operator` with the same arguments (`window` and `Window`), a JSON config and a `compute_output_signature`, so a chain prints every intermediate shape before any data arrives, and the chain runs under `jax.jit`.
- ⚙️ **One kernel on every carrier**: `KernelOp` wraps a function of plain arrays once, with a generalized-ufunc signature (`"(),()->()"`, `"(c)->()"`), and runs it on raw arrays, on named carriers and, as one lazy `apply_ufunc`, on DataArrays; `elementwise` does the same as a decorator.
- 🧮 **The array backend of a library of formulas**: `pipekit_array.arrays` gives plain functions the namespace, dtype and device of a call (`floating_inputs`, `like`), indexed updates and loops that are JAX's own calls on JAX data (`set_at`, `scan`), safe roots with finite gradients, masked and weighted reductions, checks that work under `jit`, and frozen dataclasses as pytrees.
- 🌉 **Bridges**: the DataArray adapter (loaded the first time a DataArray arrives), einx patterns with dim names as axes (`rearrange`, `reduce`, `einsum`, `repeat`), and `Estimator`, which fits and applies any scikit-learn-style estimator over named sample and feature dims.
- ✅ **A test kit in the wheel**: `assert_backends_agree`, `assert_traces`, `assert_grad_matches_fd`, `check_adapter` and the rest of `pipekit_array.testing`, so a library checks its functions on every installed array library, both precisions and JAX's transforms without copying the checks into its own `tests/`.

## 📐 One rule, six areas

The package is six areas, and a name is imported from its area. They split by what they take: plain arrays, named carriers, one function, the functions as operators, other libraries, and the test suites of other libraries.

| Area | Takes | Gives | Extra |
|---|---|---|---|
| `arrays` | plain arrays of any library | the namespace, dtype and device of a call; updates, loops, safe roots, checks, reductions, records | `numpy,compat` |
| `named` | carriers with dim names | the rule by dim name: select, reduce, reshape, shift, window, join, map, batch, rows | a backend |
| `kernels` | one function of plain arrays | that function on raw arrays and every carrier | a backend; `numpy,compat` for `elementwise` |
| `operators` | the named functions | their `Operator` twins, and operators along a dim | a backend |
| `bridges` | xarray, einx, scikit-learn | the DataArray adapter, einx by dim name, `Estimator` | `xarray`, `einx`, `sklearn` |
| `testing` | another library's functions | checks over backends, transforms, gradients, samples and adapters | `numpy` |

An area imports only from the ones before it, and `testing` from all of them ([ADR A30](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/docs/design/decisions.md#a30-six-areas-one-import-direction-and-a-recorded-module)). `import pipekit_array` re-exports the names of `named`, `kernels` and `operators` that a pipeline uses, with `Estimator` and `BackendError`, and imports neither NumPy nor xarray. `arrays` and `testing` are imported by name, because `arrays` has names of its own (its `pad` is for plain arrays).

Each twin takes its reduction by name (`"mean"`, `"nanmax"`), which is what keeps its config JSON. A twin that holds a user function cannot round-trip, so `KernelOp`, like `pipekit_ops.Lambda`, keeps a debug record of its config only; a named operator built on it is what serialises.

## 📥 Install

pipekit-array is not on PyPI; the stack installs it from git and pins a tag. Its one dependency is pipekit-ops, which is pure Python with no dependencies of its own, and every array library is an extra, so the base install brings in no array library at all. It needs Python 3.12 or newer. pip cannot find pipekit-ops on PyPI either, so give it both packages:

```bash
pip install "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops" "pipekit-array[numpy] @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-array"
```

The extras are `numpy` (NumPy 2) for arrays on the CPU, `jax` (JAX) for `jit`, `vmap` and accelerators, `dask` (`dask[array]`) for lazy, chunked arrays, `torch` (PyTorch, which also needs `compat`) and `cupy` (CuPy), `compat` (array-api-compat) for a library without a native namespace and, with `numpy`, for `pipekit_array.arrays` and `elementwise`, `xarray` for the DataArray adapter and `DimArray.to_xarray`, `sklearn` for `Estimator`, and `einx` for `pipekit_array.bridges.einx`. A library of formulas takes `numpy,compat`:

```bash
pip install "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops" "pipekit-array[numpy,compat] @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-array"
```

A uv project points `[tool.uv.sources]` at the repository with one tag for every pipekit package it names, as the libraries of the stack do, because uv refuses two refs of one repository; naming pipekit-ops as well keeps it at the same tag. The workspace tags all its packages at the same commit, so take the newest `pipekit-array-v*` tag:

```toml
[project]
dependencies = ["pipekit-ops", "pipekit-array[numpy,compat]"]"]

[tool.uv.sources]
pipekit-ops = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-ops", tag = "pipekit-array-v0.3.1" }
pipekit-array = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-array", tag = "pipekit-array-v0.3.1" }
```

uv applies `[tool.uv.sources]` only to a project's direct dependencies, so every pipekit package named there is listed in `dependencies` too.

## 🛰️ Quickstart: a year of NDVI over one tile

The cube is the one of the opening, with random reflectances. $`T`$ is the number of monthly scenes, $`B`$ the number of bands, $`Y`$ and $`X`$ the rows and columns of pixels, $`S`$ the length of a sliding window and $`K`$ the number of months per season. The reflectance of band $`b`$ at pixel $`(i, j)`$ and time $`t`$ is $`x_{t,b,i,j}`$, and the NDVI per scene and pixel is the normalised difference of the near-infrared and red bands,

```math
v_{t,i,j} = \frac{x_{t,\text{B08},i,j} - x_{t,\text{B04},i,j}}{x_{t,\text{B08},i,j} + x_{t,\text{B04},i,j} + \epsilon}
```

where $`\epsilon = 10^{-6}`$ keeps the ratio finite over dark water. From it we take the windows $`w_{n,i,j,r} = v_{n+r,i,j}`$ for $`0 \le r < S`$ and their means $`\bar{v}_{n,i,j}`$, the seasonal means $`s_{q,i,j}`$ of season $`q`$ and their change $`\Delta s_{q,i,j}`$, and the 90th percentile $`p_{i,j}`$ over the year. The last step runs the NDVI and the percentile again on the cube with its dims in another order.

```python
import numpy as np
import pipekit_array as pa


# Shapes: T = 12 monthly scenes; B = 4 bands; Y = X = 64 pixels;
#         S = 3 months per window; K = 3 months per season

# x_{t,b,i,j} ~ U(0, 0.4)
# → (T, B, Y, X)
rng = np.random.default_rng(0)
reflectance = rng.uniform(0.0, 0.4, size=(12, 4, 64, 64)).astype(np.float32)

# today (pipekit-array)
# (T, B, Y, X), with one label per band
cube = pa.DimArray(
    reflectance,
    ("time", "band", "y", "x"),
    {"band": ("B02", "B03", "B04", "B08")},
)


# user
# v = (a − b) / (a + b + ε), on plain arrays of any library
def normalized_difference(a, b, eps):
    return (a - b) / (a + b + eps)


# today (pipekit-array)
# the kernel, element by element, on raw arrays and on every carrier
ndvi_op = pa.KernelOp(normalized_difference, "(),()->()", core={}, eps=1e-6)


# user
# v_{t,i,j} from the bands labelled B08 and B04
def ndvi_of(cube):
    # today (pipekit-array)
    # (T, B, Y, X) → (T, Y, X)
    nir = pa.sel(cube, band="B08")
    # today (pipekit-array)
    # (T, B, Y, X) → (T, Y, X)
    red = pa.sel(cube, band="B04")
    # today (pipekit-array)
    # (T, Y, X), (T, Y, X) → (T, Y, X)
    return ndvi_op(nir, red)


# (T, B, Y, X) → (T, Y, X)
ndvi = ndvi_of(cube)

# today (pipekit-array)
# w_{n,i,j,r} = v_{n+r,i,j}, 0 ≤ r < S
# (T, Y, X) → (T−S+1, Y, X, S)
windows = pa.window(ndvi, time=3)
# today (pipekit-array)
# v̄_{n,i,j} = (1/S) ∑ᵣ w_{n,i,j,r}
# (T−S+1, Y, X, S) → (T−S+1, Y, X)
smooth = pa.reduce(windows, "mean", "window")

# today (pipekit-array)
# s_{q,i,j} = (1/K) ∑ₖ v_{Kq+k,i,j}
# (T, Y, X) → (T/K, Y, X)
seasons = pa.coarsen(ndvi, "mean", time=3)
# today (pipekit-array)
# Δs_{q,i,j} = s_{q+1,i,j} − s_{q,i,j}
# (T/K, Y, X) → (T/K−1, Y, X)
change = pa.diff(seasons, "time")
# today (pipekit-array)
# p_{i,j} = quantile₀.₉ {v_{t,i,j} : t}
# (T, Y, X) → (Y, X)
p90 = pa.quantile(ndvi, 0.9, "time")

# today (pipekit-array)
# the same cube with its dims in another order
# (T, B, Y, X) → (X, B, Y, T)
shuffled = pa.transpose(cube, ("x", "band", "y", "time"))
# (X, B, Y, T) → (X, Y, T)
ndvi_shuffled = ndvi_of(shuffled)
# today (pipekit-array)
# (X, Y, T) → (X, Y)
p90_shuffled = pa.quantile(ndvi_shuffled, 0.9, "time")
# today (pipekit-array)
# (X, Y) → (Y, X)
p90_back = pa.transpose(p90_shuffled, p90.dims)
same = bool(np.allclose(p90.data, p90_back.data))

print(smooth.dims, smooth.shape, change.shape, p90.shape, same)
```

This prints `('time', 'y', 'x') (10, 64, 64) (3, 64, 64) (64, 64) True`: ten three-month running means of NDVI per pixel, three changes between the four seasons, one map of the 90th percentile, and `True` because the cube with its dims shuffled gives the same map. Nothing above names an axis by number, and no step names NumPy: a `DimArray` of JAX data, or the cube as an `xr.DataArray`, goes through the same steps and gives the same numbers.

The [docs quickstart](https://jejjohnson.github.io/pipekit/quickstart-array/) takes this cube further: the operator twins chained on four carriers and under `jit`, an operator along a dim and in chunks, a scikit-learn estimator over named dims, a formula on plain arrays, and how the dispatch, the adapters and the twins work.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_array_roles/pipekit_array_roles-dark.svg">
    <img alt="pipekit-array on the GeoStack role ladder: ApplyAlong and BatchAlong as recipes, the operator twins, KernelOp, Einx and Estimator as steps, no algorithms or models of its own, kernel specs, policies and option sets as strategies, the named and plain-array functions as primitives, and DimArray, its adapters and layouts as structures, with the test kit and the registration functions beside it" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_array_roles/pipekit_array_roles-light.svg" width="100%">
  </picture>
</p>

Each public name sits on one rung of the GeoStack role ladder, by the job it does. `DimArray`, `Padded`, `RowLayout`, `Groups` and the adapters are structures (⓪), the named functions and the functions of plain arrays are primitives (①), and `KernelSpec`, which holds only the settings of a kernel, is a strategy (②). Each operator twin, `KernelOp`, `Einx` and `Estimator` is a step (⑤), and `ApplyAlong` and `BatchAlong`, which run another operator along a dim, are recipes (⑥). The table lists them by area.

| Area | Symbols | Notes |
|---|---|---|
| `arrays` | `namespace_of`, `floating_inputs`, `as_inexact`, `concrete`, `is_dataarray`, `like`, `constants_like`, `table_like`, `boolean_like`, `BackendError` | the library, dtype and device of a call |
| `arrays` | `set_at`, `add_at`, `multiply_at`, `pad`, `vmap`, `scan`, `stop_gradient`, `optimization_barrier` | JAX's own call on JAX data, an equivalent elsewhere |
| `arrays` | `divide_or_nan`, `safe_sqrt`, `safe_power`, `safe_atan2`, `squared_magnitude`, `stand_in`, `check_values`, `require_floating` | finite gradients; checks that also work under `jit` |
| `arrays` | `weighted_sum`, `weighted_mean`, `Groups`, `NO_GROUP`, `reduction_inputs`, `used_weights`, `register_record` | masked, weighted reductions; a frozen dataclass as a pytree |
| `named` | `DimArray`, `DimArrayAdapter` | the plain carrier and its reference adapter |
| `named` | `isel`, `sel`, `take_along`, `reduce`, `argreduce`, `cumreduce`, `quantile`, `dot` | select; reduce |
| `named` | `transpose`, `rename`, `expand_dims`, `squeeze`, `broadcast_like`, `merge_dims`, `split_dims`, `pad`, `roll`, `shift`, `diff`, `where`, `window`, `coarsen` | shape; shift; windows |
| `named` | `stack`, `stack_items`, `concat`, `combine`, `map_over`, `batch`, `Padded`, `PadPolicy`, `to_rows`, `from_rows`, `RowLayout` | join; map; padded batches; sample tables |
| `named` | `array_namespace`, `join_backends`, `core_dims`, `vmapped_dims`, `BOUNDARIES`, `METHODS`, `PAD_MODES`, `PAD_POLICIES` | dispatch; dims under `vmap`; allowed options |
| `kernels` | `KernelSpec`, `KernelOp`, `run_kernel`, `elementwise`, `derive_geometry`, `kernel_output_dims`, `kernel_output_signature` | one function on raw arrays and every carrier |
| `operators` | `ISel`, `Sel`, `TakeAlong`, `Reduce`, `ArgReduce`, `CumReduce`, `Quantile`, `Dot`, `Transpose`, `Rename`, `ExpandDims`, `Squeeze`, `BroadcastLike`, `MergeDims`, `SplitDims`, `Pad`, `Roll`, `Shift`, `Diff`, `Where`, `Window`, `Coarsen`, `Combine`, `MapOver` | the twin of each named function |
| `operators` | `ApplyAlong`, `BatchAlong`, `MeanScalar`, `StackAlong`, `ConcatenateAlong` | an operator along a dim; operators of raw arrays |
| `bridges.xarray` | `XarrayAdapter`, `from_xarray`, `register` | the DataArray adapter, loaded lazily (`xarray`) |
| `bridges.einx` | `rearrange`, `reduce`, `einsum`, `repeat`, `Einx`, `PatternError` | einx with dim names as axes (`einx`) |
| `bridges.sklearn` | `Estimator`, `NanPolicy` | an estimator over named dims (`sklearn`) |
| `testing` | `ArrayKind`, `array_kinds`, `values_of`, `assert_backends_agree`, `assert_refuses_dataarrays`, `check_adapter`, `check_kernel_op`, `DIMS`, `LABELS` | every array library installed; carriers |
| `testing` | `assert_matches_oracle`, `assert_adjoint`, `assert_traces`, `assert_finite_gradient`, `assert_grad_matches_fd`, `assert_mean`, `assert_variance`, `assert_uniform`, `ks_statistic`, `ks_critical_value` | oracles; transforms and gradients; samples |

## 📖 Documentation

- **[Quickstart](https://jejjohnson.github.io/pipekit/quickstart-array/)**: the operator twins on four carriers and under `jit`, `ApplyAlong` and `BatchAlong`, `Estimator`, a formula on plain arrays, how dispatch and the adapters work, and what is not built yet.
- **[API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-array/)**: every public name of the six areas, generated from the docstrings.
- **[Concepts: backend dispatch](https://jejjohnson.github.io/pipekit/concepts/#id-5-backend-dispatch-pipekit-array)**: where pipekit-array sits among the ideas of the workspace, and shape inference with `Signature`.
- **[Design docs](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array/docs/design)**: the vision, the architecture, the decisions (ADRs A1 to A30), the boundaries and the specs of each operator and dim function.
- **[Documentation](https://jejjohnson.github.io/pipekit/)**: the whole workspace, with the tutorials of every package.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_array_ecosystem/pipekit_array_ecosystem-dark.svg">
    <img alt="The 11 packages that import pipekit-array on their main branches, by layer, each with the areas it imports: pipekit-jax through its array extra, geotoolz's four packages, gridax, geophysix, radiomax, diagnostix and tracerax, and plumax; pipekit-array builds on pipekit-ops and the Array API" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_array_ecosystem/pipekit_array_ecosystem-light.svg" width="100%">
  </picture>
</p>

pipekit-array is a package of the pipekit [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), at tier 0 of the [GeoStack](https://github.com/jejjohnson/geostack). It owns array operations by dim name and on any backend, and composes the rest in:

| Concern | Owner |
|---|---|
| `Operator`, `\|`, JSON, `pipeline_hash`, `Signature`, the adapter registry, `guard` | [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) |
| The array functions every backend shares | [Python Array API standard](https://data-apis.org/array-api/) |
| The namespace of a library without a native one; the base of `arrays` | [array-api-compat](https://github.com/data-apis/array-api-compat) |
| Arrays on the CPU | [NumPy](https://github.com/numpy/numpy) |
| `jit`, `vmap`, pytrees, and the calls the `arrays` hooks stand for | [JAX](https://github.com/jax-ml/jax) |
| Lazy, chunked arrays | [dask](https://github.com/dask/dask) |
| DataArrays with coordinates, lazy through dask | [xarray](https://github.com/pydata/xarray) |
| Patterns over named axes | [einx](https://github.com/fferflo/einx) |
| Estimators fitted on a sample table | [scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| A check under a JAX trace (`check_values`; imported only there) | [equinox](https://github.com/patrick-kidger/equinox) |

Dependencies point one way. In this repository pipekit-jax imports `kernels` through its `array` extra (`to_jax` turns a `KernelOp` into a JAX operator), and pipekit-cycle and the `pipekit` umbrella install pipekit-array without importing it. In the GeoStack, geophysix, radiomax and diagnostix import `arrays` and `kernels` (geophysix and diagnostix also `testing`, in their tests), plumax and tracerax `arrays`, gridax `arrays`, `named` and the top level, and geotoolz the top level, `named` and `bridges` across geotoolz-array, geotoolz-ops, geotoolz-patcher and xrtoolz-ops. None of them is imported by pipekit-array.

## 🌳 Repository layout

```
packages/pipekit-array/
├── src/pipekit_array/
│   ├── __init__.py      ← re-exports named, kernels and operators; registers the lazy xarray adapter
│   ├── arrays/          ← functions of plain arrays ([numpy,compat])
│   ├── named/           ← DimArray and the dim functions
│   ├── kernels/         ← KernelSpec, KernelOp, elementwise
│   ├── operators/       ← the operator twins, ApplyAlong, BatchAlong
│   ├── bridges/         ← xarray.py, einx.py, sklearn.py
│   └── testing/         ← the test kit for other libraries
├── tests/               ← one folder per area, plus the area import rules
└── docs/design/         ← vision, architecture, decisions, boundaries, api/
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install                                   # uv sync of every group but train-extras + pre-commit hooks
uv run pytest packages/pipekit-array/tests     # this package's tests
make lint                                      # ruff check on the whole repo
make typecheck                                 # ty check on every package
make docs                                      # the MyST prose and the API reference, with a link check
```

`make help` lists every task. [`AGENTS.md`](https://github.com/jejjohnson/pipekit/blob/main/AGENTS.md) holds the rules for contributors and coding agents across the workspace: the definition of done, the commit and PR conventions, and the docs build.

## 📄 License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-array/LICENSE).
