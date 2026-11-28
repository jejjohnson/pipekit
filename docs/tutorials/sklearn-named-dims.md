# A scikit-learn model over named dims

A land-cover map labels every pixel of a satellite scene as water, forest or bare ground from its reflectances. A scikit-learn classifier does this from a table with one row per pixel and one column per band, but a cube has dims, not rows, and clouds leave gaps in it. This tutorial builds the table by hand with `to_rows` and `from_rows`, then lets `pipekit_array.Estimator` do the same job on any carrier, fitted on some scenes and applied to others. It then follows what the five NaN policies do to the cloudy pixels and ends with a two-step pipeline that gives the area of each class per scene.

**Time:** about 15 minutes.

**Prerequisites:**

```bash
uv add pipekit-array pipekit-ops numpy jax scikit-learn xarray
```

The blocks run in order, as one script, so a later block uses the names of an earlier one. They need the `numpy`, `jax`, `compat`, `xarray` and `sklearn` extras. The [quickstart](../quickstart/quickstart-array.md) fits a PCA with `Estimator` in one block and the [tour](../tour-array.md) lists the bridges; this page uses a supervised classifier and the NaN policies.

## The cube and its symbols

The cube has six scenes of the same 16 by 16 window, four bands (B02, B03, B04, B08) and three land-cover classes with fixed mean reflectances. The classes are the same in every scene, the reflectances carry noise, and a random 15 percent of the pixels of each scene are under cloud, which makes all four bands NaN there.

| Symbol | Meaning | Dim |
|---|---|---|
| $T$ | the number of scenes | `time` |
| $B$ | the number of bands | `band` |
| $Y$, $X$ | the rows and columns of pixels | `y`, `x` |
| $K$ | the number of classes: 0 water, 1 forest, 2 bare ground | `class` |
| $N$ | the number of samples, $T Y X$ | |
| $x_{t,b,i,j}$ | the reflectance, NaN under cloud | |
| $\mathbf{x}_n$ | the $B$ reflectances of sample $n = (t, i, j)$ | |
| $c_{t,i,j}$ | the true class | |
| $\hat{c}_{t,i,j}$ | the predicted class | |
| $p_{t,k,i,j}$ | the predicted probability of class $k$ | |

Each pixel is a draw $\mathbf{x} = \boldsymbol{\mu}_c + \boldsymbol{\epsilon}$ around the mean reflectance $\boldsymbol{\mu}_c$ of its class,

$$x_{t,b,i,j} = \mu_{c_{t,i,j},b} + \epsilon_{t,b,i,j}, \quad \epsilon_{t,b,i,j} \sim \mathcal{N}(0, 0.03^2)$$ (eq-sklearn-cube)

and the classifier learns the map $\mathbf{x}_n \mapsto c_n$ from the samples of the first four scenes.

```python
import jax
import jax.numpy as jnp
import numpy as np
import pipekit_array as pa
from sklearn.ensemble import RandomForestClassifier


# Shapes: T = 6 scenes; B = 4 bands; Y = X = 16 pixels; K = 3 classes

# μ_{c,b}: the mean reflectance of water, forest and bare ground in B02, B03, B04, B08
# → (K, B)
means = np.array(
    [[0.05, 0.08, 0.05, 0.04], [0.04, 0.08, 0.05, 0.45], [0.20, 0.22, 0.25, 0.30]],
    dtype=np.float32,
)
rng = np.random.default_rng(0)
# c_{i,j} ~ U{0, 1, 2}, the same map in every scene
# → (T, Y, X)
classes = np.broadcast_to(rng.integers(0, 3, size=(1, 16, 16)), (6, 16, 16))
# x_{t,i,j,b} = μ_{c,b} + ε, ε ~ 𝒩(0, 0.03²)
# → (T, Y, X, B), then (T, B, Y, X)
noise = rng.normal(0.0, 0.03, size=(6, 16, 16, 4)).astype(np.float32)
reflectance = (means[classes] + noise).transpose(0, 3, 1, 2).astype(np.float32)
# each pixel is under cloud with probability 0.15: NaN in every band
# → (T, Y, X)
cloud = rng.random((6, 16, 16)) < 0.15
reflectance[np.broadcast_to(cloud[:, None], reflectance.shape)] = np.nan

# today (pipekit-array)
# (T, B, Y, X), with one label per band
cube = pa.DimArray(reflectance, ("time", "band", "y", "x"), {"band": ("B02", "B03", "B04", "B08")})
# today (pipekit-array)
# c: (T, Y, X), the target
truth = pa.DimArray(classes.astype(np.int64), ("time", "y", "x"))
# scenes 0–3 to fit, scenes 4–5 held out
train, y_train = pa.isel(cube, time=slice(0, 4)), pa.isel(truth, time=slice(0, 4))
test, y_test = pa.isel(cube, time=slice(4, 6)), pa.isel(truth, time=slice(4, 6))
print(cube.dims, cube.shape, round(float(cloud.mean()), 3))
```

This prints `('time', 'band', 'y', 'x') (6, 4, 16, 16) 0.148`: the cube, and the share of cloudy pixels over all scenes.

## The table by hand

`to_rows(x, sample_dims)` makes every combination of the sample dims one row and every other dim one column. It returns the table and a `RowLayout` that says how to undo it. The rows are the C-order product of the sample dims, in the carrier's dim order, so a row is a pixel of one scene.

```python
# Shapes: N = T Y X samples; B = 4 bands

# today (pipekit-array)
# (T, B, Y, X) → (N, B), with (t, i, j) as the sample and b as the feature
rows, layout = pa.to_rows(cube, ("time", "y", "x"))
# the rows with a NaN band: the cloudy pixels
cloudy = np.isnan(rows).any(axis=1)
print(rows.shape, layout.sample_dims, layout.feature_dims, int(cloudy.sum()))
```

This prints `(1536, 4) ('time', 'y', 'x') ('band',) 228`: 1536 samples of four features, in the sample dims `time`, `y` and `x` and the feature dim `band`, and 228 cloudy rows. A classifier cannot take them, so fit on the clear rows of the four training scenes and predict the clear rows of the held-out ones. The predictions are one value per sample, which `from_rows` turns back into a carrier with the sample dims only, and the cloudy pixels stay NaN.

```python
# Shapes: N_tr = 4 Y X training samples; N_te = 2 Y X test samples; B = 4 bands

# today (pipekit-array)
# (N_tr, B) and (N_tr, 1): the training table and its targets
x_rows, _ = pa.to_rows(train, ("time", "y", "x"))
y_rows, _ = pa.to_rows(y_train, ("time", "y", "x"))
# today (pipekit-array)
# (N_te, B) and its layout, to rebuild the map from
t_rows, t_layout = pa.to_rows(test, ("time", "y", "x"))

# user
# keep the clear rows: m_n = 1 when all B bands are finite
clear_train = np.isfinite(x_rows).all(axis=1)
clear_test = np.isfinite(t_rows).all(axis=1)
# user
# ĉ = f(x), the forest fitted on the clear training rows
forest = RandomForestClassifier(n_estimators=20, random_state=0)
forest.fit(x_rows[clear_train], y_rows[clear_train, 0])
# user
# ĉ_n for the clear test rows, NaN for the cloudy ones: (N_te,)
by_hand = np.full(t_rows.shape[0], np.nan)
by_hand[clear_test] = forest.predict(t_rows[clear_test])

# today (pipekit-array)
# (N_te,) → (T, Y, X), the sample dims in place
class_map = pa.from_rows(by_hand, t_layout)
correct = class_map.data[clear_test.reshape(2, 16, 16)] == y_test.data[clear_test.reshape(2, 16, 16)]
print(class_map.dims, class_map.shape, int(np.isnan(class_map.data).sum()), float(correct.mean()))
```

This prints `('time', 'y', 'x') (2, 16, 16) 68 1.0`: a class map over `time`, `y` and `x` with the 68 cloudy pixels of the two held-out scenes left as NaN, and every clear pixel classified correctly, because the classes are well separated.

## The same with `Estimator`

`Estimator(estimator, sample_dims=..., method=...)` does the table, the fit, the apply and the rebuild. `fit(x, y)` takes the targets as a carrier whose dims are the sample dims, `method="predict"` makes calling the operator run `predict`, and `nan_policy="mask_samples"` leaves the cloudy rows out of fitting and applying and returns them as NaN, which is what the hand loop did. The `estimator` is cloned on `fit`, so the forest above is never touched, and the fitted copy is `estimator_`.

```python
# Shapes: T = 2 held-out scenes; Y = X = 16 pixels

# today (pipekit-array)
# ĉ = f(x) per pixel; cloudy pixels are left out and come back as NaN
classify = pa.Estimator(
    RandomForestClassifier(n_estimators=20, random_state=0),
    sample_dims=("time", "y", "x"),
    nan_policy="mask_samples",
    method="predict",
)
# today (pipekit-array)
# fit on the four training scenes with their classes, then apply to the held-out ones
classify.fit(train, y_train)
# (T, B, Y, X) → (T, Y, X)
predicted = classify(test)
print(predicted.dims, predicted.shape, bool(np.array_equal(predicted.data, class_map.data, equal_nan=True)))
```

This prints `('time', 'y', 'x') (2, 16, 16) True`: the band dim is gone and the map is the one built by hand, NaN for NaN.

## The class probabilities

`predict_proba` returns one column per class, so the result needs a dim for the columns. `Estimator` calls it `class` unless `feature_dim` names it, and puts it where the band dim was.

```python
# Shapes: T = 2 held-out scenes; K = 3 classes; Y = X = 16 pixels

# today (pipekit-array)
# p_{t,k,i,j} = P(c = k | x), from the fitted forest
proba = classify.predict_proba(test)
# the order of the classes in the K columns
print(proba.dims, proba.shape, classify.estimator_.classes_.tolist())
```

This prints `('time', 'class', 'y', 'x') (2, 3, 16, 16) [0, 1, 2]`: the band dim replaced by `class` in place, with the classes in the order 0, 1, 2.

## What a NaN costs: the five policies

A band that is NaN at every pixel, as when a sensor fails, makes the policies differ, so add a fifth band to the cube that is NaN everywhere. The loop below fits and applies one `Estimator` per policy and prints the share of NaN pixels in the result, the accuracy over the pixels that got a class, and the accuracy over the cloudy pixels that got one (the pixels with `nan` in the cube).

```python
# Shapes: T = 6 scenes; B = 5 bands, one of them dead; Y = X = 16 pixels

# today (pipekit-array)
# (T, B+1, Y, X): the cube with a band that is NaN everywhere (concat drops labels)
dead = pa.DimArray(np.full((6, 1, 16, 16), np.nan, np.float32), ("time", "band", "y", "x"))
broken = pa.concat([cube, dead], "band")
train5, test5 = pa.isel(broken, time=slice(0, 4)), pa.isel(broken, time=slice(4, 6))
# the cloudy pixels of the held-out scenes: (T, Y, X)
cloudy_test = np.isnan(test.data).any(axis=1)

for policy in ("propagate", "raise", "mask_samples", "mask_features", "mask"):
    # today (pipekit-array)
    model = pa.Estimator(
        RandomForestClassifier(n_estimators=20, random_state=0),
        sample_dims=("time", "y", "x"),
        nan_policy=policy,
        method="predict",
    )
    try:
        model.fit(train5, y_train)
        out = model.predict(test5).data
    except ValueError:
        print(f"{policy:14s} ValueError")
        continue
    got = ~np.isnan(out)
    share = float(np.isnan(out).mean())
    clear_acc = float((out[got & ~cloudy_test] == y_test.data[got & ~cloudy_test]).mean())
    cloud_acc = (
        float((out[got & cloudy_test] == y_test.data[got & cloudy_test]).mean())
        if (got & cloudy_test).any()
        else None
    )
    kept = None if model.kept_features_ is None else model.kept_features_.tolist()
    print(f"{policy:14s} nan {share:.3f}  clear {clear_acc:.3f}  cloudy {cloud_acc if cloud_acc is None else round(cloud_acc, 3)}  kept {kept}")
```

This prints

```text
propagate      nan 0.000  clear 1.000  cloudy 0.338  kept None
raise          ValueError
mask_samples   ValueError
mask_features  nan 0.000  clear 1.000  cloudy 0.338  kept [True, True, True, True, False]
mask           nan 0.133  clear 1.000  cloudy None  kept [True, True, True, True, False]
```

Each policy fails or succeeds for its own reason. `"propagate"` hands every row to the forest as it is: this scikit-learn forest accepts NaN, so it returns a class for every pixel, and for the cloudy pixels, which have no information, the class is a guess that is right a third of the time, which is chance for three classes. Whether a model accepts NaN depends on the model and the scikit-learn version, so `Estimator` does not decide it. `"raise"` stops at the first non-finite entry, and the dead band makes that every fit. `"mask_samples"` drops the rows with any NaN, which is every row here, so the table is empty and scikit-learn raises. `"mask_features"` drops the column that is non-finite in every fitted row, the last of `kept`, and still sends the cloudy rows to the forest, so it matches `"propagate"`. `"mask"` drops the dead column first and then the cloudy rows, so the clear pixels are right and the cloudy ones are NaN, which is what a map should show.

## A carrier of your own: JAX and xarray

The result is a carrier like the input. A `DimArray` gives a `DimArray`, an `xr.DataArray` gives a `DataArray`, and a JAX array is converted to NumPy for scikit-learn, because fit and apply run on the host. Inside `jax.jit` the data is a tracer that cannot be converted, so the call raises a `TypeError` that says to call it eagerly.

```python
# Shapes: T = 2 held-out scenes; B = 4 bands; Y = X = 16 pixels

# today (pipekit-array)
# (T, B, Y, X), the same scenes as an xr.DataArray with band labels, and as JAX
as_xarray = test.to_xarray()
on_jax = pa.DimArray(jnp.asarray(test.data), test.dims, test.labels)
# (T, B, Y, X) → (T, Y, X), on each carrier
from_xarray = classify(as_xarray)
from_jax = classify(on_jax)
print(type(from_xarray).__name__, from_xarray.dims, type(from_jax).__name__, type(from_jax.data).__name__)

# the fit of the same operator, inside jit
try:
    jax.jit(pa.Estimator(RandomForestClassifier(), sample_dims=("time", "y", "x")).fit)(on_jax)
except TypeError as error:
    print("TypeError:", str(error).split(";")[0])
```

This prints

```text
DataArray ('time', 'y', 'x') DimArray ArrayImpl
TypeError: Estimator.fit runs scikit-learn on the host and cannot run inside a JAX trace
```

The `DataArray` comes back with the dims `time`, `y` and `x`, and the `DimArray` built on JAX data comes back on JAX, though scikit-learn computed on NumPy in between.

## A two-step pipeline: area per class

A fitted `Estimator` is an operator, so it chains with the dim functions. `predict_proba` followed by a mean over the pixels gives the share of each class per scene, $a_{t,k} = \frac{1}{\lvert C_t \rvert} \sum_{(i,j) \in C_t} p_{t,k,i,j}$, with $C_t$ the clear pixels of scene $t$. `nanmean` leaves the cloudy pixels out, because `"mask_samples"` gave them NaN probabilities.

```python
from pipekit_ops import Sequential


# Shapes: T = 2 held-out scenes; K = 3 classes; Y = X = 16 pixels

# today (pipekit-array)
# the class probabilities as the operator's output, with NaN under cloud
probabilities = pa.Estimator(
    RandomForestClassifier(n_estimators=20, random_state=0),
    sample_dims=("time", "y", "x"),
    nan_policy="mask_samples",
    method="predict_proba",
).fit(train, y_train)
# today (pipekit-array)
# a_{t,k} = mean of p_{t,k,i,j} over the clear pixels
clear_mean = pa.Reduce("nanmean", ("y", "x"))
# today (pipekit-ops)
area = Sequential([probabilities, clear_mean])

# (T, B, Y, X) → (T, K)
shares = area(test)
true_share = np.stack([(y_test.data[t][~cloudy_test[t]][:, None] == np.arange(3)).mean(axis=0) for t in range(2)])
print(shares.dims, shares.shape, np.round(shares.data, 2).tolist(), np.round(true_share, 2).tolist())
```

This prints `('time', 'class') (2, 3) [[0.3, 0.32, 0.38], [0.31, 0.3, 0.39]] [[0.3, 0.32, 0.38], [0.31, 0.31, 0.39]]`: a share per class and scene, which sums to 1 over the classes, next to the true shares of the clear pixels.

## Make it yours

- **Another model.** Any object with `fit` and the method you call works: `LogisticRegression`, a `Pipeline` of a scaler and a classifier, `KMeans` with `method="predict"` for unlabelled clusters.
- **Other sample dims.** Choose the samples by name: `sample_dims=("y", "x")` leaves `time` and `band` as features, so one row is a pixel's whole record of all scenes and bands.
- **Fit on another carrier.** A fitted `Estimator` applies to any carrier with the same feature dims and sizes, and raises a `ValueError` naming the fitted and the given features if they differ.

## Where next

- [The quickstart](../quickstart/quickstart-array.md): the PCA `Estimator` with `feature_dim`, and the other operator twins.
- [The tour](../tour-array.md): `to_rows`, `from_rows` and the bridges in one place.
- [Padded batches](padded-batches.md): running a jitted model on a scene too large for one call.
