# A tour of pipekit-array

pipekit-array is six areas, each with one job. This page visits them in the order they import each other: `arrays` (formulas on plain arrays), `named` (functions of carriers with dim names), `kernels` (one array function on every carrier), `operators` (the named functions as pipeline steps), `bridges` (xarray, einx and scikit-learn) and `testing` (the checks a library built on pipekit-array runs). Each section shows the main names at work and lists the rest. The [quickstart](quickstart/quickstart-array.md) is the shorter path through the same ground; the [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-array/) has every signature.

The blocks run in order, as one script, so a later block uses the names of an earlier one. They need the `numpy`, `jax`, `compat`, `xarray`, `einx` and `sklearn` extras.

## The map

| Area | Names | Reach for it when |
|---|---|---|
| `arrays` | `namespace_of`, `floating_inputs`, `like`, `set_at`, `add_at`, `scan`, `vmap`, `divide_or_nan`, `safe_sqrt`, `weighted_mean`, `Groups`, `check_values`, `register_record` and 20 more | a formula on raw arrays that must run on NumPy, JAX, dask or PyTorch |
| `named` | `DimArray`, `isel`, `sel`, `reduce`, `rename`, `merge_dims`, `shift`, `diff`, `window`, `coarsen`, `quantile`, `dot`, `where`, `map_over`, `batch`, `Padded`, `to_rows`, `from_rows` and 25 more | arrays whose axes have names: a `DimArray`, an `xr.DataArray` or a carrier of your own |
| `kernels` | `KernelSpec`, `KernelOp`, `elementwise`, `run_kernel`, `kernel_output_dims` and 2 more | one array function, written once, on raw arrays and on every carrier |
| `operators` | `ISel`, `Reduce`, `Coarsen`, `Diff`, `Window`, … (one twin per named function), `ApplyAlong`, `BatchAlong`, `MapOver`, `MeanScalar`, `StackAlong`, `ConcatenateAlong` | a named function as a step of a `Sequential` or a `Graph`, saved as JSON |
| `bridges` | `bridges.xarray` (`XarrayAdapter`, `from_xarray`), `bridges.einx` (`rearrange`, `reduce`, `einsum`, `repeat`, `Einx`), `bridges.sklearn` (`Estimator`) | xarray, einx patterns or a scikit-learn model on named dims |
| `testing` | `assert_backends_agree`, `assert_traces`, `check_kernel_op`, `check_adapter`, `assert_matches_oracle`, `assert_adjoint`, `assert_mean` and 12 more | the test suite of a library built on pipekit-array |

An area imports only from the ones before it, and `testing` from all of them ([ADR A30](design-array/decisions.md)). Import a name from its area (`from pipekit_array.arrays import scan`), never from a file in it. `import pipekit_array` re-exports the names of `named`, `kernels` and `operators` that a pipeline uses, `Estimator` and `BackendError`, 68 names in all; `arrays` and `testing` are imported by name.

## `arrays`: formulas on plain arrays

A formula library (geophysix, radiomax, diagnostix) has no dims to name, but it wants one function for every array library. The functions of `pipekit_array.arrays` find the namespace of a call, put Python numbers and NumPy arrays on the library and device of the other inputs, and fill the gaps of the array API standard: indexed updates, loops and a few safe versions of functions whose gradient is infinite somewhere.

### The namespace of a call

`floating_inputs` returns the namespace of its inputs and the inputs in one floating dtype; `like` puts a number in the dtype and on the device of a reference array. Here the dewpoint depression $\Delta T = T - T_d$ at three stations, in kelvin, with the temperature on JAX and the dewpoint as a Python list:

```python
import jax.numpy as jnp
import numpy as np
from pipekit_array.arrays import floating_inputs, like


# Shapes: N_s = 3 stations


# user
# ΔT = T − T_d, clipped below at 0 K
def depression(temperature, dewpoint):
    # today (pipekit-array)
    # the namespace of the call, and both inputs in one floating dtype
    xp, (t, td) = floating_inputs(temperature, dewpoint)
    # today (pipekit-array)
    # 0 K in t's dtype, on its device
    zero = like(0.0, t, xp)
    # (N_s,)
    return xp.maximum(t - td, zero)


# → (N_s,), on JAX: the list joins the JAX array
on_jax = depression(jnp.asarray([288.0, 281.5, 300.0]), [283.0, 282.0, 290.0])
# → (N_s,), on NumPy
on_numpy = depression(np.asarray([288.0, 281.5, 300.0]), np.asarray([283.0, 282.0, 290.0]))
print(type(on_jax).__name__, on_jax.dtype, np.asarray(on_jax).tolist(), on_numpy.tolist())
```

This prints `ArrayImpl float32 [5.0, 0.0, 10.0] [5.0, 0.0, 10.0]`: a JAX array in JAX's default float32, and the same numbers on NumPy. At the second station the dewpoint is above the temperature, so the depression is clipped to 0.

### Indexed updates and loops

The standard has no `x[i] = v` that works everywhere (JAX arrays are immutable) and no loop that JAX can compile. `set_at`, `add_at` and `multiply_at` return an updated copy, through `x.at[i]` on JAX and a copy and an assignment elsewhere; `scan` is `jax.lax.scan` on JAX data and a Python loop elsewhere. Here a running total of rainfall $c_k = c_{k-1} + r_k$, after a faulty gauge reading is set to zero:

```python
from pipekit_array.arrays import scan, set_at


# Shapes: K = 5 hours


# today (pipekit-array)
# r with r_2 = 0: a copy, the input unchanged
rain = np.asarray([1.0, 0.5, 99.0, 2.0, 0.0])
fixed = set_at(rain, 2, 0.0)


# user
# (c_k, y_k) = (c_{k-1} + r_k, c_{k-1} + r_k)
def accumulate(total, r):
    total = total + r
    return total, total


# today (pipekit-array)
# the last carry c_{K-1}, and every c_k stacked: (K,)
last, totals = scan(accumulate, 0.0, fixed)
# today (pipekit-array)
# the same on JAX, as one jax.lax.scan
last_jax, totals_jax = scan(accumulate, jnp.asarray(0.0), jnp.asarray(fixed))
print(rain[2], totals.tolist(), float(last_jax))
```

This prints `99.0 [1.0, 1.5, 1.5, 3.5, 3.5] 3.5`: the input keeps its faulty 99.0, the running totals skip it, and the compiled JAX scan ends on the same total.

### Safe functions and weighted reductions

`safe_sqrt`, `safe_power` and `safe_atan2` return 0 with a zero gradient where the plain function's gradient is infinite, and `divide_or_nan` gives NaN where a denominator is 0 instead of a warning or an infinity. `weighted_mean` and `weighted_sum` take weights and a validity mask, and `Groups` reduces samples by group label, built once on the host so JAX sees it as a constant. Here the mean of five gauges in two catchments, with the third gauge invalid:

```python
from pipekit_array.arrays import Groups, divide_or_nan, weighted_mean


# Shapes: N = 5 gauges; G = 2 catchments


# today (pipekit-array)
# r̄ = ∑ₙ mₙ wₙ rₙ / ∑ₙ mₙ wₙ, over the valid gauges
rain = np.asarray([2.0, 4.0, 99.0, 1.0, 3.0])
valid = np.asarray([True, True, False, True, True])
area = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0])
mean = weighted_mean(rain, weights=area, valid=valid)
# today (pipekit-array)
# the mean per catchment; the invalid gauge has label −1, so it is in none
catchments = Groups.from_labels([0, 0, -1, 1, 1], n_groups=2)
per_catchment = catchments.mean(rain)
# today (pipekit-array)
# NaN where the denominator is 0
ratio = divide_or_nan(np.asarray([1.0, 1.0]), np.asarray([2.0, 0.0]), np)
print(float(mean), per_catchment.tolist(), ratio.tolist())
```

This prints `2.3333333333333335 [3.0, 2.0] [0.5, nan]`: the area-weighted mean $(2 + 4 + 2 \cdot 1 + 2 \cdot 3) / 6$ of the four valid gauges, the mean of each catchment without the invalid gauge, and NaN for $1 / 0$.

The rest of the area: `namespace_of` (the namespace alone), `as_inexact`, `constants_like`, `table_like` and `boolean_like` (a constant, a NumPy table or a mask in the library of a reference array), `concrete` (the values on the host, or `None` while JAX traces), `pad`, `vmap`, `stop_gradient` and `optimization_barrier`, `squared_magnitude`, `stand_in`, `reduction_inputs` and `used_weights` (the pieces of the weighted reductions), `check_values` (a check of values that also holds under `jit`, through Equinox's errors) and `require_floating` (a check of the dtype), `is_dataarray`, and `register_record` (a frozen dataclass as a JAX pytree, registered once JAX loads). The functions that find the namespace themselves (`namespace_of`, `floating_inputs`, the weighted reductions) raise `BackendError` for a DataArray: plain-array code takes plain arrays, and `elementwise` is how it accepts carriers.

## `named`: carriers with dim names

### `DimArray`

`DimArray` is the carrier pipekit-array ships: a raw array, one name per axis and static labels on some dims. It has no coordinates, no alignment and no georeference; `to_xarray()` crosses over when an analysis needs them. The running example of the rest of the page is a small cube of six monthly scenes, three bands and a 4 by 4 pixel window:

```python
import pipekit_array as pa


# Shapes: T = 6 monthly scenes; B = 3 bands; Y = X = 4 pixels

# x_{t,b,i,j} ~ U(0, 0.4)
rng = np.random.default_rng(0)
data = rng.uniform(0.0, 0.4, size=(6, 3, 4, 4))
# today (pipekit-array)
# (T, B, Y, X), with one label per band
cube = pa.DimArray(data, ("time", "band", "y", "x"), {"band": ("B04", "B08", "B11")})
print(cube.dims, cube.shape, cube.labels["band"])
```

This prints `('time', 'band', 'y', 'x') (6, 3, 4, 4) ('B04', 'B08', 'B11')`.

### Selecting and reducing

`isel` selects by position and `sel` by exact label; an integer or a single label drops its dim. `reduce` takes a reduction by its name in the array namespace (`"mean"`, `"nanmax"`, `"std"`) and the dims to reduce over:

```python
# Shapes: T = 6 monthly scenes; B = 3 bands; Y = X = 4 pixels


# today (pipekit-array)
# x_{t,B08,i,j}: (T, Y, X)
nir = pa.sel(cube, band="B08")
# today (pipekit-array)
# the first three scenes: (3, B, Y, X)
first = pa.isel(cube, time=slice(0, 3))
# today (pipekit-array)
# m_{t,b} = (1/(Y X)) ∑ᵢ ∑ⱼ x_{t,b,i,j}: (T, B)
scene_mean = pa.reduce(cube, "mean", ("y", "x"))
print(nir.dims, first.shape, scene_mean.dims, scene_mean.labels["band"])
```

This prints `('time', 'y', 'x') (3, 3, 4, 4) ('time', 'band') ('B04', 'B08', 'B11')`: the band dim dropped by a single label, three scenes, and a mean per scene and band that keeps the band labels.

### Shapes, shifts, windows and algebra

The dim functions follow xarray's names and arguments ([dim-level functions](design-array/api/dim_functions.md) has the parity table). Four groups: shapes (`rename`, `transpose`, `expand_dims`, `squeeze`, `merge_dims`, `split_dims`, `broadcast_like`), shifts (`pad`, `roll`, `shift`, `diff`, `cumreduce`, `where`), windows (`window`, `coarsen`) and algebra (`quantile`, `dot`, `take_along`, `argreduce`).

```python
# Shapes: T = 6 monthly scenes; B = 3 bands; Y = X = 4 pixels; P = Y X pixels


# today (pipekit-array)
# the pixels as one dim, last: (T, B, P)
flat = pa.merge_dims(cube, pixel=("y", "x"))
# today (pipekit-array)
# Δx_{t,b,i,j} = x_{t+1,b,i,j} − x_{t,b,i,j}: (T−1, B, Y, X)
change = pa.diff(cube, "time")
# today (pipekit-array)
# the mean of each three-month season: (T/3, B, Y, X)
seasons = pa.coarsen(cube, "mean", time=3)
# today (pipekit-array)
# windows of three months, sliding by one, as a new last dim: (T−2, B, Y, X, 3)
windows = pa.window(cube, time=3)
# today (pipekit-array)
# the 10th and 90th percentiles over time: (2, B, Y, X)
spread = pa.quantile(cube, [0.1, 0.9], "time")
# today (pipekit-array)
# the month of the highest value per band and pixel: (B, Y, X)
peak = pa.argreduce(cube, "argmax", "time")
for result in (flat, change, seasons, windows, spread, peak):
    print(result.dims, result.shape)
```

This prints

```text
('time', 'band', 'pixel') (6, 3, 16)
('time', 'band', 'y', 'x') (5, 3, 4, 4)
('time', 'band', 'y', 'x') (2, 3, 4, 4)
('time', 'band', 'y', 'x', 'window') (4, 3, 4, 4, 3)
('quantile', 'band', 'y', 'x') (2, 3, 4, 4)
('band', 'y', 'x') (3, 4, 4)
```

that is, the sixteen pixels as one dim, five changes from six months, two seasons, four full windows of three months, two percentiles and one month index per band and pixel.

### Many carriers, and a function per item

`stack` and `concat` join carriers with the same dims and never align them; `combine` applies a function to the data of several carriers; `map_over` runs a function on each item along a dim and stacks the results back. `join_backends` is what each of them runs first: NumPy data next to JAX data joins it, and any other mix raises `BackendError`.

```python
# Shapes: T = 6 monthly scenes; B = 3 bands; Y = X = 4 pixels


# today (pipekit-array)
# v_{t,i,j} = (x_{t,B08,i,j} − x_{t,B04,i,j}) / (x_{t,B08,i,j} + x_{t,B04,i,j}): (T, Y, X)
red = pa.sel(cube, band="B04")
ndvi = pa.combine(lambda a, b: (a - b) / (a + b), nir, red)
# today (pipekit-array)
# the two years side by side, as a new leading dim: (2, T/2, B, Y, X)
years = pa.stack([pa.isel(cube, time=slice(0, 3)), pa.isel(cube, time=slice(3, 6))], "year")
# today (pipekit-array)
# the brightest pixel of each scene, one scene per call: (T, B)
brightest = pa.map_over(cube, lambda scene: pa.reduce(scene, "max", ("y", "x")), "time")
print(ndvi.dims, years.dims, years.shape, brightest.dims)
```

This prints `('time', 'y', 'x') ('year', 'time', 'band', 'y', 'x') (2, 3, 3, 4, 4) ('time', 'band')`.

### Batches and sample tables

`batch` groups carriers into `Padded` batches of a fixed size, the last one padded with copies and a mask that says which items are real, so a jitted step compiles once. `to_rows` flattens a carrier to a `(n_samples, n_features)` table and returns the layout that `from_rows` needs to rebuild it.

```python
# Shapes: T = 6 monthly scenes; B = 3 bands; Y = X = 4 pixels; N = T Y X samples


# today (pipekit-array)
# the scenes in batches of four: (4, B, Y, X), then two real ones and two copies
scenes = [pa.isel(cube, time=t) for t in range(6)]
batches = list(pa.batch(scenes, size=4, dim="scene"))
masks = [np.asarray(b.valid).tolist() for b in batches]
# today (pipekit-array)
# reduce(where=…) leaves the copies out of a mean over the batch: (B, Y, X)
last_mean = pa.reduce(batches[1].value, "mean", "scene", where=batches[1].valid)
# today (pipekit-array)
# every (t, i, j) a row, every band a column: (N, B)
rows, layout = pa.to_rows(cube, sample_dims=("time", "y", "x"))
# today (pipekit-array)
# and back: (T, B, Y, X)
again = pa.from_rows(rows, layout)
print(batches[0].value.dims, masks[1], last_mean.dims, rows.shape, again.dims)
```

This prints `('scene', 'band', 'y', 'x') [True, True, False, False] ('band', 'y', 'x') (96, 3) ('time', 'band', 'y', 'x')`: a batch of four scenes, a last batch of two real scenes and two copies, a mean over that batch that leaves the copies out, and a table of $6 \cdot 4 \cdot 4 = 96$ rows and three columns that rebuilds the cube.

The rest of the area: `DimArrayAdapter` (the reference adapter, and the one a carrier of your own is checked against), `array_namespace`, `stack_items`, `vmapped_dims` and `core_dims` (dim names around a bare `jax.vmap`), `RowLayout`, and the tuples of accepted arguments `METHODS`, `BOUNDARIES`, `PAD_MODES` and `PAD_POLICIES`.

## `kernels`: one array function on every carrier

A kernel is a plain array function whose core dims come last, described by a generalised-ufunc signature: `"(),()->()"` is element-wise, `"(c)->()"` consumes one dim. `KernelOp` wraps it once, and `core` says which carrier dim each letter stands for, so the same kernel serves `"band"` on one carrier and `"variable"` on another. Every output has the loop dims of all the inputs, in the first input's order; on a DataArray the call is one `xr.apply_ufunc`, lazy on dask. Here the root-mean-square over the bands, $r_{t,i,j} = \sqrt{\frac{1}{B}\sum_b x_{t,b,i,j}^2}$:

```python
from pipekit_array.kernels import KernelOp, kernel_output_dims


# Shapes: T = 6 monthly scenes; B = 3 bands; Y = X = 4 pixels


# user
# r = √((1/C) ∑_c x_c²), over the last axis
def rms(x):
    return np.sqrt(np.mean(x * x, axis=-1))


# today (pipekit-array)
# "c" is the band dim of the carrier
op = KernelOp(rms, "(c)->()", core={"c": "band"})
# (T, B, Y, X) → (T, Y, X)
on_cube = op(cube)
# (T, B, Y, X) → (T, Y, X), on the DataArray of the same cube
on_xarray = op(cube.to_xarray())
# today (pipekit-array)
# the output dims, from the input dims alone
dims = kernel_output_dims(op.spec, {"c": "band"}, [cube.dims])
print(on_cube.dims, type(on_xarray).__name__, bool(np.allclose(on_cube.data, on_xarray.values)), dims)
```

This prints `('time', 'y', 'x') DataArray True (('time', 'y', 'x'),)`: the band dim consumed, the same numbers from the DataArray, and the output dims known before the call.

`elementwise` is the shortcut for the element-wise case: a decorator that lets a function of plain arrays take named carriers too, as the [quickstart](quickstart/quickstart-array.md) shows with a wind speed. `run_kernel` is the dispatch behind `KernelOp`, for a library that writes its own operator class; `derive_geometry` and `kernel_output_signature` are its other pieces. A `KernelOp` holds a user function, so its config is a debug record only (`forbid_in_yaml`); a named operator built on one is what serialises.

## `operators`: the named functions as pipeline steps

Each named function has an operator twin with the same arguments (`coarsen` and `Coarsen`, `window` and `Window`), which takes a reduction by name, so its config is JSON. The twins chain with `|`, save with `pipekit_ops.dumps`, and implement `compute_output_signature`, so `summary` gives every shape before any data arrives.

```python
import pipekit_ops as pk


# Shapes: T = 6 monthly scenes; B = 3 bands; Y = X = 4 pixels


# today (pipekit-array)
# the seasonal means, their change, averaged over the pixels: (T/3 − 1, B)
seasonal_change = pa.Coarsen("mean", time=3) | pa.Diff("time") | pa.Reduce("mean", ("y", "x"))
# today (pipekit-ops)
# the same chain from its JSON
again = pk.loads(pk.dumps(seasonal_change))
result = again(cube)
print(result.dims, result.shape, pk.pipeline_hash(again) == pk.pipeline_hash(seasonal_change))
```

This prints `('time', 'band') (1, 3) True`: one change between the two seasons for each band, from a chain that came back from JSON with the same `pipeline_hash`.

Four operators have no function twin:

- `ApplyAlong(inner, dim)` runs an operator on each item along a dim, and serialises when `inner` does; `MapOver(fn, dim)` does the same with a function and is `forbid_in_yaml`.
- `BatchAlong(inner, size, dim, pad=...)` runs an operator on chunks of one carrier, for a model on a cube larger than device memory ([Patch inference in padded batches](tutorials/padded-batches.md)).
- `MeanScalar`, `StackAlong` and `ConcatenateAlong` work on raw arrays (`StackAlong` and `ConcatenateAlong` take named carriers too, with `dim=`).

:::{note}
`pipekit_array.Reduce` reduces over named dims, while `pipekit_ops.Reduce` is a left fold over a list. Import the packages under their own names (`pa.Reduce`, `pk.Reduce`) when you use both.
:::

## `bridges`: xarray, einx and scikit-learn

Each bridge is one module behind one extra, and imports its library only when used.

- **`bridges.xarray`** (`[xarray]`): `XarrayAdapter`, registered lazily the first time a DataArray reaches pipekit, so every named function and operator above takes a DataArray; several take xarray's own fast path, which stays lazy on dask. `from_xarray` turns a DataArray into a `DimArray`, its 1-D index coordinates becoming labels.
- **`bridges.einx`** (`[einx]`): einx patterns with dim names as axes, `rearrange`, `reduce`, `einsum` and `repeat`, and `Einx`, one pattern as a serialisable operator.
- **`bridges.sklearn`** (`[sklearn]`): `Estimator`, a scikit-learn-style estimator fitted and applied on a carrier through `to_rows` and `from_rows` ([A scikit-learn model over named dims](tutorials/sklearn-named-dims.md)).

```python
from pipekit_array.bridges import einx as pe
from pipekit_array.bridges.xarray import from_xarray


# Shapes: T = 6 monthly scenes; B = 3 bands; Y = X = 4 pixels


# today (pipekit-array)
# the DataArray of the cube, and back: band labels as a coordinate, then as labels
da = cube.to_xarray()
back = from_xarray(da)
# today (pipekit-array)
# m_{t,b} = (1/(Y X)) ∑ᵢ ∑ⱼ x_{t,b,i,j}, as an einx pattern on dim names: (T, B)
means = pe.reduce("time band y x -> time band", cube, op="mean")
# today (pipekit-array)
# the same pattern as an operator, from its JSON
op = pk.loads(pk.dumps(pe.Einx("reduce", "time band y x -> time band", op="mean")))
print(back.labels["band"], means.dims, bool(np.allclose(means.data, op(da).values)))
```

This prints `('B04', 'B08', 'B11') ('time', 'band') True`: the band labels back from the coordinate, and the same means from the function on a `DimArray` and from the operator on the DataArray.

## `testing`: the checks a library runs

`pipekit_array.testing` is what geophysix, radiomax and the other libraries on pipekit-array test their functions with. Here are the three most used, on the depression function from the first section. They compare float64 results, so a test suite turns on JAX's 64-bit mode once, in its `conftest.py`.

```python
import jax
from pipekit_array.testing import array_kinds, assert_backends_agree, assert_traces


# Shapes: N_s = 3 stations


# the checks compare float64 results, so a suite turns on JAX's 64-bit mode,
# once, in its conftest.py
jax.config.update("jax_enable_x64", True)
t = np.asarray([288.0, 281.5, 300.0])
td = np.asarray([283.0, 282.0, 290.0])
# today (pipekit-array)
# NumPy's numbers on every installed kind of array, float32 and float64, and under jit
assert_backends_agree(depression, t, td)
# today (pipekit-array)
# jit, vmap and an eager loop agree over a batch of 4 calls: (4, N_s) each
assert_traces(depression, np.stack([t] * 4), np.stack([td] * 4))
# today (pipekit-array)
# the kinds of array installed here
kinds = [kind.name for kind in array_kinds()]
```

Neither check raises. `kinds` names the kinds of array installed: NumPy, `array_api_strict` on two devices, JAX, dask and xarray here, and PyTorch where it is installed. The first check ran on all but xarray, which it adds with `labelled=True` for an element-wise function, and it ran JAX both eagerly and under `jit`.

The rest of the area:

- **Carriers and kernels:** `check_adapter` (an adapter of your own against `DimArrayAdapter`), `check_kernel_op` (one `KernelOp` across carrier families), `assert_refuses_dataarrays`, `values_of`, and the `DIMS` and `LABELS` of the test carriers.
- **Numbers:** `assert_matches_oracle` (against a reference to `atol + rtol·|expected|`, where a mask holds), `assert_adjoint` (the dot-product test of an operator and its adjoint), `assert_finite_gradient` and `assert_grad_matches_fd`.
- **Random draws:** `assert_mean`, `assert_variance` and `assert_uniform`, within the sampling error, and the `ks_statistic` and `ks_critical_value` behind them.

## Where next

- [Patch inference in padded batches](tutorials/padded-batches.md) and [A scikit-learn model over named dims](tutorials/sklearn-named-dims.md): two tasks from start to finish.
- The [design pages](design-array/README.md): the vision, the architecture, the decisions and the operator catalogue.
