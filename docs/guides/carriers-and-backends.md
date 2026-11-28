# Carriers and backends

pipekit-array runs one function on NumPy, JAX, dask and xarray data, and some mixes of those in one call are fine while others are errors. This page says which is which, and why: what joins in the plain-array functions and in the named layer, how to see a move, what the functions do with NaN, when dask stays lazy, and what a DataArray keeps that a `DimArray` does not. Each rule has a block that runs it.

The blocks run in order, as one script, so a later block uses the names of an earlier one. They need the `numpy`, `jax`, `compat`, `dask` and `xarray` extras, and the last one the `sklearn` and `einx` extras. For the names themselves read the [tour](../tour-array.md); for the decisions, [ADR A6, A7, A23 and A24](../design-array/decisions.md).

## What joins in the plain-array functions

The functions of `pipekit_array.arrays` follow the host rule of [ADR A23](../design-array/decisions.md): Python numbers, sequences and NumPy arrays live on the host, and join the library and the device of the other arrays. Two libraries other than NumPy in one call raise `BackendError`, and so does a value that is no array of a known library. `BackendError` derives from `TypeError` and from `ValueError`, so a caller that catches either one catches it.

```python
import dask.array as da
import jax.numpy as jnp
import numpy as np
from pipekit_array.arrays import namespace_of


# Shapes: N = 4 stations


# today (pipekit-array)
# a number and a NumPy array alone: NumPy
alone = namespace_of(np.ones(4), 2.0)
# today (pipekit-array)
# a NumPy array next to JAX: JAX
with_jax = namespace_of(np.ones(4), jnp.ones(4))
# today (pipekit-array)
# a NumPy array next to dask: dask
with_dask = namespace_of(np.ones(4), da.ones(4))
print(alone.__name__, with_jax.__name__, with_dask.__name__)
```

This prints `array_api_compat.numpy jax.numpy array_api_compat.dask.array`: NumPy data takes the library of the other array, whichever it is.

```python
from pipekit_array import BackendError


# Shapes: N = 4 stations


# today (pipekit-array)
# JAX next to dask: two libraries other than NumPy
try:
    namespace_of(jnp.ones(4), da.ones(4))
except BackendError as error:
    first = str(error).split(":")[0]
    print(type(error).__name__, isinstance(error, TypeError), isinstance(error, ValueError), first)
```

This prints `BackendError True True arrays of several libraries in one call`: one error class that is both a `TypeError` and a `ValueError`, naming the mix. Put one of the arrays on the other's library yourself, as no function moves an array between two libraries for you.

## What joins in the named layer

The dim functions (`stack`, `concat`, `combine`, `reduce(where=)` and the rest) and `KernelOp` follow [ADR A24](../design-array/decisions.md): a NumPy array next to a JAX one joins it, in either order, and any other mix raises `BackendError`. The named layer is stricter than the plain functions on purpose, because it would otherwise move a carrier's data from NumPy to JAX, or compute a dask array, without being asked. `join_backends` is the function each of them runs first on the raw data.

```python
import pipekit_array as pa


# Shapes: Y = 2 rows; X = 3 columns


# today (pipekit-array)
# the same field on NumPy, on JAX and on dask: (Y, X)
on_numpy = pa.DimArray(np.ones((2, 3)), ("y", "x"))
on_jax = pa.DimArray(jnp.ones((2, 3)), ("y", "x"))
on_dask = pa.DimArray(da.ones((2, 3)), ("y", "x"))
# today (pipekit-array)
# a new leading dim "k" from two carriers: (2, Y, X)
joined = pa.stack([on_numpy, on_jax], "k")
print(type(joined.data).__name__, joined.dims)

for other in (on_jax, on_numpy):
    try:
        pa.stack([on_dask, other], "k")
    except BackendError as error:
        print(type(error).__name__, str(error).split(";")[0])
```

This prints

```text
ArrayImpl ('k', 'y', 'x')
BackendError stack: inputs on ['dask', 'jax']
BackendError stack: inputs on ['dask', 'numpy']
```

that is, a stack on JAX (`ArrayImpl` is JAX's array) from NumPy and JAX, and two refusals that name the call and the libraries: dask with JAX, and dask with NumPy, which joins in the plain functions above but not here.

## Seeing a move

NumPy and JAX join without a warning. When you build an app, above all one on a GPU, you may want to know what moved: `pipekit_ops.guard` reports each eager join of an array of one dimension or more, and a dtype it widens. `"warn"` gives a `BackendWarning` and `"raise"` a `BackendError`; Python numbers, NumPy scalars and 0-d arrays are never reported. The [guards guide](../guards.md) has the rest, and `jit_report` for the steps that do not trace.

```python
import warnings

import pipekit_ops as pk
from pipekit_array.arrays import floating_inputs


# Shapes: N = 4 stations


# today (pipekit-ops)
# "raise" makes the first move an error, which suits a test
with pk.guard("raise"):
    # today (pipekit-array)
    # a number joins JAX: no move
    xp, _ = floating_inputs(jnp.ones(4), 2.0)
    try:
        # today (pipekit-array)
        # an array of 4 NumPy values joins JAX: a move
        floating_inputs(jnp.ones(4), np.ones(4))
    except BackendError as error:
        print(xp.__name__, "|", error)

# today (pipekit-ops)
# "warn" reports each move as a BackendWarning and carries on
with pk.guard("warn"), warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    pa.stack([on_jax, on_numpy], "k")
print([(w.category.__name__, str(w.message)) for w in caught])
```

This prints

```text
jax.numpy | floating_inputs: NumPy data joined jax
[('BackendWarning', 'stack: NumPy data joined jax')]
```

that is, no report for the number, and one report for each array, naming the function that moved it.

## NaN is the caller's choice

[ADR A6](../design-array/decisions.md) documents NaN and does not hide it: NumPy and JAX differ at the edges, so no function turns NaN into a number behind your back. Where a function takes NaN into account, you say so.

- `reduce` takes the reduction by its name in the namespace, so `"mean"` propagates a NaN and `"nanmean"` skips it.
- `quantile` has `skipna=False`, and `skipna=True` means `nanquantile`.
- `weighted_mean` and the other weighted reductions give NaN for a NaN in a valid sample; a NaN in a sample that `valid` marks out of the mean does not count.
- `divide_or_nan` gives NaN where the denominator is 0.

```python
from pipekit_array.arrays import weighted_mean


# Shapes: S = 2 sites; T = 3 times


# today (pipekit-array)
# (S, T), one gap at site 0, time 1
gappy = pa.DimArray(np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 6.0]]), ("site", "t"))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # today (pipekit-array)
    # the mean over time, with and without skipping the gap: (S,)
    plain = pa.reduce(gappy, "mean", "t").data
    skipping = pa.reduce(gappy, "nanmean", "t").data
    # today (pipekit-array)
    # the median over time, then the median that skips NaN: (S,)
    median = pa.quantile(gappy, 0.5, "t").data
    nanmedian = pa.quantile(gappy, 0.5, "t", skipna=True).data
# today (pipekit-array)
# the mean of site 0 with the gap marked invalid, and with it left valid
row = gappy.data[0]
marked = weighted_mean(row, weights=np.ones(3), valid=~np.isnan(row))
unmarked = weighted_mean(row, weights=np.ones(3))
print(plain, skipping, median, nanmedian)
print(float(marked), float(unmarked))
```

This prints

```text
[nan  4.] [2. 4.] [nan  4.] [2. 4.]
2.0 nan
```

that is, a NaN in the mean and the median of site 0 until you ask to skip it, and the mean 2.0 when the gap is marked invalid and NaN when it is not.

`Estimator` has its own `nan_policy`, because scikit-learn estimators do not take NaN: `"propagate"` (the default), `"raise"`, `"mask_samples"` (rows with a non-finite entry are left out and come back as NaN), `"mask_features"` (columns non-finite in every fitted row are dropped, and come back as NaN from `inverse_transform`) and `"mask"` (both, for a land-masked field that also has gaps).

## Dask stays lazy if you keep it so

[ADR A7](../design-array/decisions.md): pipekit-array never calls `.compute()` for you. The dim functions keep dask lazy, and you call `.compute()` or `.persist()` at the end, where you decide that the data is needed. Two pieces cannot stay lazy, and the ADR names both: `Estimator` computes a dask input on the host, because scikit-learn computes on NumPy, and the einx bridge raises `BackendError` naming `.compute()`, because einx has no dask backend. `concrete` computes too, since its job is the values on the host.

```python
from pipekit_array.bridges import einx as pe
from pipekit_array.bridges.sklearn import Estimator
from sklearn.decomposition import PCA


# Shapes: B = 4 bands; Y = 6 rows; X = 5 columns


# today (pipekit-array)
# (B, Y, X), in chunks of 3 rows
rng = np.random.default_rng(0)
lazy = pa.DimArray(da.from_array(rng.normal(size=(4, 6, 5)), chunks=(4, 3, 5)), ("band", "y", "x"))
# today (pipekit-array)
# three dim functions on dask data: still lazy
steps = [pa.reduce(lazy, "mean", "x"), pa.diff(lazy, "y"), pa.quantile(lazy, 0.5, "x")]
lazy_types = [type(step.data).__module__ for step in steps]
# today (pipekit-array)
# the final compute is yours: (B, Y)
values = pa.reduce(lazy, "mean", "x").data.compute()
# today (pipekit-array)
# a PCA over the pixels computes the input: (C, Y, X)
pca = Estimator(PCA(2), sample_dims=("y", "x"), feature_dim="component")
fitted = pca.fit(lazy).transform(lazy)
print(lazy_types, values.shape, type(fitted.data).__module__, fitted.dims)
try:
    pe.reduce("band y x -> band", lazy, op="mean")
except BackendError as error:
    print(error)
```

This prints

```text
['dask.array.core', 'dask.array.core', 'dask.array.core'] (4, 6) numpy ('component', 'y', 'x')
reduce: einx has no dask backend; call .compute() (or .load()) on the input first.
```

that is, three lazy results, one array you computed, a PCA output that is NumPy, and the einx refusal that tells you what to do.

## A DataArray and a `DimArray`

The named functions take a `DimArray` and an `xr.DataArray` alike. They differ in what a result keeps. A `DimArray` has one name per axis and static labels on some dims, and no coordinates, no alignment and no georeference. A DataArray has all of xarray's, and several dim functions take xarray's own path for it, which stays lazy on dask. A coordinate stays while its dim does, and goes with a dim that a reduction removes. `to_xarray` makes the labels coordinates, and `from_xarray` makes 1-D index coordinates labels and drops the rest.

```python
import xarray as xr
from pipekit_array.bridges.xarray import from_xarray


# Shapes: T = 3 times; B = 2 bands


# today (pipekit-array)
# (T, B), with a label for each band
labelled = pa.DimArray(np.ones((3, 2)), ("time", "band"), {"band": ("a", "b")})
# today (pipekit-array)
# the same, as a DataArray with a band coordinate
to_da = labelled.to_xarray()
# user
# (T, B), with an index and a latitude along time
rich = xr.DataArray(
    np.ones((3, 2)),
    dims=("time", "band"),
    coords={"time": [10, 20, 30], "band": ["a", "b"], "lat": ("time", [1.0, 2.0, 3.0])},
)
# today (pipekit-array)
# the mean over time of each: (B,)
mean_dim = pa.reduce(labelled, "mean", "time")
mean_da = pa.reduce(rich, "mean", "time")
# today (pipekit-array)
# the mean over band of each: (T,)
by_time = pa.reduce(rich, "mean", "band")
# today (pipekit-array)
# the DataArray as a DimArray: the index coordinates are labels, "lat" is dropped
back = from_xarray(rich)
print(list(to_da.coords), mean_dim.labels, list(mean_da.coords), list(by_time.coords), back.labels)
```

This prints

```text
['band'] {'band': ('a', 'b')} ['band'] ['time', 'lat'] {'time': (10, 20, 30), 'band': ('a', 'b')}
```

that is, labels that became a coordinate, the band labels kept by a mean over time on both carriers, a mean over band that keeps `time` and its `lat`, and a `DimArray` that has the two index coordinates as labels and no `lat`.

`elementwise` and `KernelOp` give a DataArray no name and no attributes, because the quantity and its unit have changed; its coordinates stay. A DataArray next to a `DimArray` in one `elementwise` call raises `BackendError`, and so does a DataArray given to a plain-array function such as `namespace_of`: plain-array code takes plain arrays, and `elementwise` is how it takes carriers.

## PyTorch needs `compat`

A PyTorch tensor has no `__array_namespace__`, so `array_namespace` cannot find its namespace on its own. [array-api-compat](https://github.com/data-apis/array-api-compat) supplies it, and the `compat` extra installs it: install `pipekit-array[torch,compat]`, not `torch` alone. A value with no namespace at all gets an error that says so:

```python
# today (pipekit-array)
# an object that is no array of any library
try:
    pa.array_namespace(object())
except TypeError as error:
    sentences = str(error).split(". ")
    print(sentences[0])
    print(sentences[-1])
```

This prints

```text
No input implements the Array API
PyTorch additionally needs pipekit-array[compat] for namespace dispatch.
```

the first and the last sentence of the message, which names `compat`.
