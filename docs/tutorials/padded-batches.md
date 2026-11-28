# Patch inference in padded batches

A scene is too large to give a model in one call, so it is cut into patches, the patches go through the model in batches, and the results are stitched back into a scene. The model is jitted, and a jitted function compiles again for every new input shape. Sixteen patches in batches of six leave a short last batch of four, which would be a second shape and a second compilation. `batch` pads the short batch to the full size with copies and returns a mask of the real entries, so the model compiles once. This tutorial builds the loop by hand with the named functions, then writes the same job as one operator, `BatchAlong`, and saves it as JSON.

**Time:** about 15 minutes.

**Prerequisites:**

```bash
uv add pipekit-array pipekit-ops jax numpy
```

The blocks run in order, as one script, so a later block uses the names of an earlier one. They need the `numpy`, `jax` and `compat` extras. The [tour](../tour-array.md) shows every named function and the [quickstart](../quickstart/quickstart-array.md) shows `BatchAlong` on a time series; this page uses them on one task.

## The scene and its symbols

The scene has three bands (B02, B03, B04) over 32 by 32 pixels. It is cut into a 4 by 4 grid of patches of 8 by 8 pixels, and a toy model maps the three bands of every pixel to two features.

| Symbol | Meaning | Dim |
|---|---|---|
| $B$ | the number of bands | `band` |
| $Y$, $X$ | the rows and columns of pixels in the scene | `y`, `x` |
| $T_y$, $T_x$ | the patches down and across the scene | `ty`, `tx` |
| $P_y$, $P_x$ | the rows and columns of pixels in a patch | `py`, `px` |
| $N$ | the number of patches, $T_y T_x$ | `tile` or `patch` |
| $M$ | the patches in a batch | `patch` |
| $F$ | the features the model returns | `feature` |
| $x_{b,i,j}$ | the scene | |
| $\mathbf{W}$ | the weights of the model, $F \times B$ | |
| $z_{f,i,j}$ | the model's output | |
| $v_n$ | 1 for a real patch, 0 for a copy | |

The model is a linear map over the bands followed by a tanh, applied at every pixel independently,

$$z_{f,i,j} = \tanh\Big(\sum_{b} W_{f,b}\, x_{b,i,j}\Big)$$ (eq-padded-model)

so the answer on the whole scene is known and the patched run can be checked against it.

## Cut the scene into patches

`split_dims` splits `y` into `ty` and `py` and `x` into `tx` and `px`, in C order, and `merge_dims` joins the two patch dims into one `tile` dim. Taking one entry of `tile` with `isel` gives a patch with the dims `band`, `py` and `px`.

```python
import jax
import jax.numpy as jnp
import numpy as np
import pipekit_array as pa


# Shapes: B = 3 bands; Y = X = 32 pixels; T_y = T_x = 4 patches per side;
#         P_y = P_x = 8 pixels per patch; N = 16 patches

# x_{b,i,j} ~ 𝒩(0, 1), fixed seed
# → (B, Y, X)
rng = np.random.default_rng(0)
data = rng.normal(size=(3, 32, 32)).astype(np.float32)

# today (pipekit-array)
# (B, Y, X), with one label per band
scene = pa.DimArray(data, ("band", "y", "x"), {"band": ("B02", "B03", "B04")})
# today (pipekit-array)
# y → (T_y, P_y), x → (T_x, P_x), in place
split = pa.split_dims(scene, y={"ty": 4, "py": 8}, x={"tx": 4, "px": 8})
# today (pipekit-array)
# (T_y, T_x) → N, in C order: patch n is row n // T_x, column n % T_x
merged = pa.merge_dims(split, tile=("ty", "tx"))
# today (pipekit-array)
# → (N, B, P_y, P_x)
tiles = pa.transpose(merged, ("tile", "band", "py", "px"))
# today (pipekit-array)
# patch n: (B, P_y, P_x)
patches = [pa.isel(tiles, tile=n) for n in range(16)]
print(tiles.dims, tiles.shape, patches[0].dims, patches[0].labels)
```

This prints `('tile', 'band', 'py', 'px') (16, 3, 8, 8) ('band', 'py', 'px') {'band': ('B02', 'B03', 'B04')}`: sixteen patches of three bands, and each patch keeps the band labels.

## Group the patches into padded batches

`batch` takes the patches one at a time and yields `Padded` batches of `size` along a new dim, here `patch`. A `Padded` holds the stacked patches as `value` and a boolean `valid` with one entry per patch. With `pad="last"` the short last batch is filled with copies of its last real patch, so every batch has the same shape.

```python
# Shapes: M = 6 patches per batch; N = 16, so batches of 6, 6 and 4 real patches

# today (pipekit-array)
# (M, B, P_y, P_x) per batch; v_n = 1 for the first 6, 6 and 4 patches, else 0
batches = list(pa.batch(patches, size=6, dim="patch", pad="last"))
for padded in batches:
    print(padded.value.dims, padded.value.shape, padded.valid.tolist())
```

This prints

```text
('patch', 'band', 'py', 'px') (6, 3, 8, 8) [True, True, True, True, True, True]
('patch', 'band', 'py', 'px') (6, 3, 8, 8) [True, True, True, True, True, True]
('patch', 'band', 'py', 'px') (6, 3, 8, 8) [True, True, True, True, False, False]
```

The last batch holds the four real patches 12 to 15 and two copies of patch 15. The copies are real data, so the model sees finite values in distribution, but their `valid` is false so that nothing downstream counts them. `pad="none"` would yield the short batch as it is, which is a second shape, and `pad="drop"` would lose its patches.

## Run a jitted model, compiled once

`forward` is the model of {eq}`eq-padded-model` on a batch, jitted. Its first line is a Python side effect, which runs only while JAX traces the function, so the counter `traces` counts compilations. `Padded` is a `NamedTuple` of a carrier and an array, so the whole batch could go through `jit`; here the model takes the raw data of `value`, and the dims are put back by hand.

```python
# Shapes: F = 2 features; M = 6 patches per batch

# W_{f,b} ~ 𝒩(0, 1), fixed seed
# → (F, B)
weights = rng.normal(size=(2, 3)).astype(np.float32)
traces = []


# user
# z_{m,f,i,j} = tanh(∑ᵦ W_{f,b} x_{m,b,i,j}), jitted; the append runs once per trace
@jax.jit
def forward(w, x):
    traces.append(x.shape)
    # (F, B) · (M, B, P_y, P_x) → (M, F, P_y, P_x)
    return jnp.tanh(jnp.einsum("fb,mbyx->mfyx", w, x))


# user
# the model on one batch, as a carrier again: (M, F, P_y, P_x)
def run(padded):
    z = forward(jnp.asarray(weights), jnp.asarray(padded.value.data))
    return pa.DimArray(z, ("patch", "feature", "py", "px"))


# (M, F, P_y, P_x) for each of the three batches
outputs = [run(padded) for padded in batches]
print(len(outputs), outputs[0].shape, traces)
```

This prints `3 (6, 2, 8, 8) [(6, 3, 8, 8)]`: three batches went through the model, and `forward` was traced once, for the shape `(6, 3, 8, 8)`. Without the padding the last batch would have been `(4, 3, 8, 8)` and a second trace.

## Reduce over the real patches only

Every output has six patches, but the last has two copies. `reduce` takes a mask `where`, and a plain array covers the leading dims of the data: the `valid` of a batch has one entry per `patch`, and `patch` is the first dim of the output. Here the mean of each feature over the scene, $\bar z_f = \frac{1}{N P_y P_x} \sum_{n} v_n \sum_{i,j} z_{n,f,i,j}$, takes each batch's masked sum, then adds the three sums and divides by the number of real pixels.

```python
# Shapes: F = 2 features; N = 16 patches; P_y = P_x = 8 pixels per patch

# today (pipekit-array)
# s_f = ∑ₙ vₙ ∑ᵢⱼ z_{n,f,i,j}, per batch: (F,)
sums = [
    pa.reduce(z, "sum", ("patch", "py", "px"), where=padded.valid)
    for z, padded in zip(outputs, batches, strict=True)
]
# z̄_f = (1/(N P_y P_x)) ∑ s_f: (F,)
mean = sum(np.asarray(s.data) for s in sums) / (16 * 8 * 8)

# user
# the same on the whole scene, without patches: z̄_f = mean over i, j of z_{f,i,j}
direct = np.tanh(np.einsum("fb,byx->fyx", weights, data)).mean(axis=(1, 2))
# today (pipekit-array)
# the same sums with the copies counted, for comparison: (F,)
unmasked = sum(np.asarray(pa.reduce(z, "sum", ("patch", "py", "px")).data) for z in outputs)
print(sums[0].dims, bool(np.allclose(mean, direct, atol=1e-5)), (unmasked / 1024 - direct).round(4))
```

This prints `('feature',) True [ 0.0247 -0.0297]`: each batch's sum is over the patch and pixel dims and leaves only `feature`, the masked mean equals the mean over the whole scene, and counting the two copies of patch 15 would have shifted the mean by the amounts in the last array.

## Stitch the scene back

The real patches of each batch are picked with an index array, the batches are joined along `patch`, and the steps of the cut run backwards: `split_dims` the patch dim into `ty` and `tx`, put each patch row next to its pixel rows with `transpose`, and `merge_dims` them into `y` and `x`.

```python
# Shapes: F = 2 features; Y = X = 32 pixels; T_y = T_x = 4; P_y = P_x = 8

# today (pipekit-array)
# the real patches of each batch, dropping the copies: (valid count, F, P_y, P_x)
real = [
    pa.isel(z, patch=np.flatnonzero(np.asarray(padded.valid)))
    for z, padded in zip(outputs, batches, strict=True)
]
# today (pipekit-array)
# the batches joined: (N, F, P_y, P_x)
joined = pa.concat(real, "patch")
# today (pipekit-array)
# N → (T_y, T_x), in C order
grid = pa.split_dims(joined, patch={"ty": 4, "tx": 4})
# today (pipekit-array)
# (T_y, T_x, F, P_y, P_x) → (F, T_y, P_y, T_x, P_x)
interleaved = pa.transpose(grid, ("feature", "ty", "py", "tx", "px"))
# today (pipekit-array)
# (T_y, P_y) → Y and (T_x, P_x) → X: (F, Y, X)
stitched = pa.merge_dims(interleaved, y=("ty", "py"), x=("tx", "px"))

# user
# the model on the whole scene: (F, Y, X)
whole = np.tanh(np.einsum("fb,byx->fyx", weights, data))
print(joined.shape, stitched.dims, stitched.shape, bool(np.allclose(stitched.data, whole, atol=1e-5)))
```

This prints `(16, 2, 8, 8) ('feature', 'y', 'x') (2, 32, 32) True`: sixteen patches again, a scene with the dims `feature`, `y` and `x`, and the same numbers as the model run on the whole scene.

## The same loop as one operator

`BatchAlong(inner, size, dim, pad)` is the loop of the last four sections for one carrier: it cuts `dim` into chunks, pads a short last chunk with copies of its last entry, runs `inner` on each chunk, trims the padding from the output and joins the chunks. Its `inner` must be a pipekit operator and must keep `dim`, one output entry per input entry. So the model becomes a small `Operator`. It keeps its weights as lists, so its configuration is JSON, and its `_apply` calls the jitted `forward` of above.

The operator also implements `compute_output_signature`, which is what lets `summary` print its output shape from a `Signature` without running it.

```python
from pipekit_ops import Operator, Sequential, Signature


# Shapes: N = 16 patches; B = 3 bands; F = 2 features; P_y = P_x = 8 pixels per patch

# user
# z = tanh(W x), over the band dim, on a carrier with a `tile` dim
class LinearTanh(Operator):
    def __init__(self, *, weights):
        self.weights = [[float(v) for v in row] for row in weights]

    def _apply(self, x):
        w = jnp.asarray(self.weights, dtype=jnp.float32)
        # (M, B, P_y, P_x) → (M, F, P_y, P_x)
        z = forward(w, jnp.asarray(x.data))
        return pa.DimArray(z, ("tile", "feature", "py", "px"))

    def get_config(self):
        return {"weights": self.weights}

    def compute_output_signature(self, signature):
        sizes = dict(signature.dims)
        dims = (("tile", sizes["tile"]), ("feature", len(self.weights)), ("py", sizes["py"]), ("px", sizes["px"]))
        return Signature(dims=dims, dtype="float32")


# today (pipekit-array)
# z_n for every patch, six patches per call, the short last call padded then trimmed
model = pa.BatchAlong(LinearTanh(weights=weights), size=6, dim="tile", pad="last")
# today (pipekit-array)
# z̄_{n,f} = mean of z over the pixels of patch n
patch_mean = pa.Reduce("mean", ("py", "px"))
# today (pipekit-ops)
pipeline = Sequential([model, patch_mean])

# (N, B, P_y, P_x) → (N, F)
means = pipeline(tiles)
# user
# the same from the hand loop: (N, F)
expected = np.concatenate([np.asarray(z.data)[: int(np.asarray(p.valid).sum())] for z, p in zip(outputs, batches, strict=True)]).mean(axis=(2, 3))
print(means.dims, means.shape, bool(np.allclose(means.data, expected, atol=1e-5)), len(traces))
```

This prints `('tile', 'feature') (16, 2) True 1`: one mean per patch and feature, the same as the hand loop, and still one trace, because the operator called the same jitted `forward` on the same shape.

## The shapes before the data, and the JSON round trip

`summary` runs `compute_output_signature` through the chain from a `Signature`, so the shapes of every step print before anything runs. `pipekit_ops.dumps` writes the pipeline as JSON, `loads` rebuilds it, and `pipeline_hash` says the two are the same pipeline.

```python
import json

import pipekit_ops as pk


# Shapes: N = 16 patches; B = 3 bands; P_y = P_x = 8 pixels per patch

# today (pipekit-ops)
# the patches' dims, sizes and dtype, without their data
signature = Signature(dims=(("tile", 16), ("band", 3), ("py", 8), ("px", 8)), dtype="float32")
print(pipeline.summary(signature))

# today (pipekit-ops)
# the pipeline as JSON, and back
text = pk.dumps(pipeline)
again = pk.loads(text)
steps = json.loads(text)["config"]["operators"]
print([s["class"] for s in steps], sorted(steps[0]["config"]))
print(bool(np.allclose(again(tiles).data, means.data)), pk.pipeline_hash(again) == pk.pipeline_hash(pipeline))
```

This prints

```text
Operator        Input sig                                            Output sig
------------------------------------------------------------------------------------------------------------------------
[0] BatchAlong  Signature((tile=16, band=3, py=8, px=8) float32)     Signature((tile=16, feature=2, py=8, px=8) float32)
[1] Reduce      Signature((tile=16, feature=2, py=8, px=8) float32)  Signature((tile=16, feature=2) float32)
['BatchAlong', 'Reduce'] ['dim', 'inner', 'pad', 'size']
True True
```

The summary shows the 16 patches of three bands becoming 16 patches of two features, and the pixels averaged away. The JSON holds the two steps, and the `BatchAlong` config holds its `dim`, `inner`, `pad` and `size`, with the weights inside `inner`. The rebuilt pipeline gives the same numbers and the same hash. `loads` looks a class up among the modules already imported, so the module that defines `LinearTanh` must be imported where the JSON is read.

## Make it yours

- **A real model.** Replace `forward` with a trained network and keep the jit at the top: the operator needs only an `_apply` that returns a carrier with `tile` kept.
- **A different batch size.** `size` is the one number to tune against device memory, and the loop and the operator both follow it.
- **Overlapping patches.** `window` cuts overlapping patches along one dim, and blending the overlaps when stitching is a step this tutorial does not build.

## Where next

- [The tour of pipekit-array](../tour-array.md): every named function and bridge.
- [The quickstart](../quickstart/quickstart-array.md): `ApplyAlong`, `BatchAlong` and `Estimator` in a page.
- [Design decisions](../design-array/decisions.md): why `BatchAlong` is not `BatchedMap` (A19) and what the pad policies are (A10).
