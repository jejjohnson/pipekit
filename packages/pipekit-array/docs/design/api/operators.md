---
status: draft
version: 0.1.0
---

# Operators: data flow

The seven operators that move arrays through a pipeline: the combinators, the reducer, the carrier-shaped `BatchAlong`, the `Histogram` controller and the `Subsample` geometry op.

| Operator | Status |
| --- | --- |
| `ApplyAlong` (was `ApplyToBands`) | today, `pipekit_array.dim_operators` (#206) |
| `StackAlong`, `ConcatenateAlong` | today, `pipekit_array.combinators` |
| `Subsample` | retired: `isel` with a stepped slice (A15) |
| `Histogram` | not built |
| `MeanScalar` | today, `pipekit_array.reductions` |
| `BatchAlong` (was the array-shaped `BatchedMap`) | today, `pipekit_array.batching` (#206) |

## `ApplyToBands(inner: Operator, axis: int = 0)`

Superseded by `ApplyAlong(inner: Operator, dim: str)` in `pipekit_array.dim_operators` (ADR A15, phase 3 of [dim_functions.md](dim_functions.md)), which ships today.
It does the same split, apply and stack by dim name, and it serialises whenever `inner` does.
`ApplyToBands` itself was never built.

The design: split the input along `axis`, apply `inner` to each slice, and stack the results back along `axis`.
It is the canonical "apply a band-wise operation" combinator.

```python
# proposed (phase B; superseded by ApplyAlong, A15)
from pipekit_array.combinators import ApplyToBands

# proposed (phase C)
from pipekit_array.qc import AssertValueRange

# Apply a value-range assertion per band; axis 0 is the band dim.
checked = ApplyToBands(AssertValueRange(0.0, 1.0), axis=0)
out = checked(reflectance_arr)  # (B, H, W) → (B, H, W)
```

The shipped form, on a named carrier, runs as written:

```python
import numpy as np
from pipekit_array import ApplyAlong, DimArray, Reduce

# Shapes: T = 3 scenes (time); B = 2 bands (band); X = 4 pixels (x)

data = np.ones((3, 2, 4))                                 # → (T, B, X)
# today (pipekit-array)
cube = DimArray(data, ("time", "band", "x"))              # (T, B, X)

# today (pipekit-array)
# s_{tj} = ∑_b x_{tbj}, for each t on its own
per_scene = ApplyAlong(Reduce("sum", dim="band"), dim="time")
out = per_scene(cube)                                     # (T, B, X) → (T, X)
assert out.dims == ("time", "x")
```

**Contract (`ApplyToBands`, as designed):**

- `_apply(x)`: `xp = array_namespace(x)`; for each `i`, `chunks.append(inner(xp.take(x, i, axis=axis)))`; return `xp.stack(chunks, axis=axis)`.
- `inner` must return arrays of the same shape *minus* the split axis (`xp.stack` re-adds it).
  If `inner` returns scalars, `axis=0` and the output is a 1-D array.
  If `inner` reduces ambiguously, you get a `ValueError` from `xp.stack`.
- `inner` must return arrays of the same dtype (`xp.stack` enforces it).

**As built (`ApplyAlong`):** `_apply` is `map_over(x, inner, dim)`, so `inner` runs on `isel(x, dim=i)` and the results are stacked along `dim`, placed first.
`inner` returns named carriers with the same dims.

**`get_config()`:**

```python
# As designed (ApplyToBands):
{
    "inner": {"class": type(self.inner).__name__, "config": self.inner.get_config()},
    "axis": self.axis,
}
# As built (ApplyAlong):
{"inner": nested_config(self.inner), "dim": self.dim}
```

`nested_config(op)` is `pipekit_ops`' shared helper; it returns the child's full `{"module", "class", "config"}` record.

**`compute_output_signature(sig)`:** delegates to `inner` on the per-item signature and re-adds the split dim, first, if `inner` returns a signature.

**Why this is in pipekit-array, not pipekit core:** the split and stack mechanics use Array API functions (`xp.take`, `xp.stack`), or, as built, the named-carrier adapter; neither is carrier-agnostic.
The closest pipekit core combinator is `Fanout`, which takes a list of operators rather than slicing one carrier.

---

## `StackAlong(axis: int | None = None, dim: str | None = None)`

Stack a list of arrays along a new axis.
It is a thin wrapper over `xp.stack`, so the operation composes into a pipeline.

```python
out = StackAlong(axis=0)([arr_a, arr_b, arr_c])  # 3 × (H, W) → (3, H, W)
```

**Contract:**

- Input: a `list` or `tuple` of arrays, all the same shape and all on the same backend (`array_namespace(*xs)` asserts it).
- Output: a single array, `xp.stack(xs, axis=axis)`.
- With no arguments, `axis` defaults to 0.
- **As built**, `dim=` takes named carriers instead: the call becomes `pipekit_array.stack(xs, dim)`, and the new dim leads.
  `axis` and `dim` are mutually exclusive; passing both raises `ValueError`.

**`get_config()`:** `{"axis": self.axis}`, or `{"dim": self.dim}` for the named form.

**Failure mode:** an empty input list raises `ValueError`; a mixed-backend list raises `TypeError` via `array_namespace`; mismatched shapes raise the backend's stack error.

---

## `ConcatenateAlong(axis: int | None = None, dim: str | None = None)`

Concatenate a list of arrays along an existing axis.
It is a thin wrapper over `xp.concat`, the [Array API standard][api-concat] name (NumPy also exposes it as `concat` from 2.0, next to the legacy `concatenate` alias).

[api-concat]: https://data-apis.org/array-api/latest/API_specification/generated/array_api.concat.html

```python
out = ConcatenateAlong(axis=0)([arr_a, arr_b])  # 2 × (N, H, W) → (2N, H, W)
```

**Contract:**

- Input: a `list` or `tuple` of arrays, all the same shape *except* along `axis`, and all on the same backend.
- Output: a single array, `xp.concat(xs, axis=axis)`.
- With no arguments, `axis` defaults to 0.
- **As built**, `dim=` takes named carriers instead (`pipekit_array.concat(xs, dim)`); `axis` and `dim` are mutually exclusive.

**`get_config()`:** `{"axis": self.axis}`, or `{"dim": self.dim}` for the named form.

**Distinction from `StackAlong`:** `StackAlong` *adds* a new axis, while `ConcatenateAlong` joins along an *existing* axis, as NumPy's `stack` and `concat` do.

---

## `Subsample(stride: int | tuple[int, ...] = 10, axes: tuple[int, ...] = (-2, -1))`

**Retired (ADR A15):** `Subsample` is not built, because on a named carrier it is `isel` with a stepped slice, `isel(x, y=slice(None, None, 2), x=slice(None, None, 2))` (xarray's `thin`), and the operator form is `ISel(dim, index)`, one dim per operator.
See [dim_functions.md](dim_functions.md) §2.

The design: stride-decimate the input along `axes`.
It defaults to the last two axes (H, W in the remote-sensing convention) with stride 10.

```python
# proposed (phase B; retired by A15)
out = Subsample(stride=2)(image)                       # (B, H, W) → (B, H/2, W/2)
out = Subsample(stride=(4, 2), axes=(-2, -1))(image)   # (B, H, W) → (B, H/4, W/2)
```

**Contract:**

- `_apply(x)`: build a slice tuple of `slice(None)` everywhere except `axes`, where `slice(None, None, stride[i])` applies, and return `x[tuple(slices)]`.
- `stride` may be an int (broadcast across `axes`) or a tuple of ints (one per axis).

**`get_config()`:** `{"stride": self.stride, "axes": self.axes}`.

**`compute_output_signature(sig)`:** returns a `Signature` with the per-axis sizes divided by their strides (with `ceil` division), and the other axes unchanged.

**Per-backend note:** all five backends support `__getitem__` with slice tuples per the Array API, so there is no special handling.

---

## `Histogram(bins: int | array | "auto" = 10, range: tuple[float, float] | None = None)`

**Not built yet** (phase B).

A controller that captures distributions at named tap sites.
It follows the pattern of `pipekit_ops.observe.Histogram`, but with array-aware defaults (`to_array = xp.reshape(x, (-1,))` is implicit).

```python
# proposed (phase B)
hist = Histogram(bins=50, range=(0.0, 1.0))
# today (pipekit-array): ISel, MeanScalar
pipe = ISel("y", slice(None, None, 2)) | hist.at("post_thin") | MeanScalar()
pipe(image)
hist.report()  # {"post_thin": (counts, edges)}
```

**Contract:**

- The class itself is not an `Operator`.
  `Histogram.at(key)` returns a small `_HistAt(Operator)` that:
  1. calls `array_namespace(x)` and gets `xp`;
  2. flattens via `xp.reshape(x, (-1,))`;
  3. computes bin counts via `xp.searchsorted(edges, flat) - 1`, followed by a per-bin `xp.sum(indices == i)`.
     Edges are computed once per `Histogram` controller from `bins` and `range`.
     Both `xp.searchsorted` and `xp.sum` are in the [Array API standard][api-search], so no histogram primitive is needed;
  4. stores `(counts, edges)` under `key` in the controller's `captures` dict.
     Counts and edges are stored as whatever the backend produced (for example a `jax.Array` from a JAX input), and the user calls `np.asarray(counts)` at inspection time if they want plain NumPy;
  5. returns `x` unchanged (it is a `Tap`-shaped operator).
- `Histogram.report()` returns the captures dict (a shallow copy).

[api-search]: https://data-apis.org/array-api/latest/API_specification/generated/array_api.searchsorted.html

**`get_config()`** (on `_HistAt`): `{"key": self.key}`.
The controller itself is not serialised, because it is an interactive object, and `_HistAt` carries `forbid_in_yaml = True` because it closes over the controller (`architecture.md` §5).

**Per-backend note:** `xp.histogram` is **not** in the Array API standard, so the implementation uses `xp.searchsorted` plus `xp.sum` per bin (both conform).
That costs one pass over the array per bin, instead of one optimised `np.histogram` call, which is acceptable for the inspection and debugging use case the controller serves.
If the use case grows beyond inspection (large online statistics), Q4 in `boundaries.md` has the streaming-counters upgrade path.

---

## `MeanScalar(axis: int | tuple[int, ...] | None = None, keepdims: bool = False)`

Reduce via `xp.mean`.
It defaults to a full reduction (`axis=None`), giving a scalar (`ndim == 0`).
The output dtype matches the input's floating dtype, per the Array API `mean` contract (see the [spec][api-mean]).

[api-mean]: https://data-apis.org/array-api/latest/API_specification/generated/array_api.mean.html

```python
import numpy as np
from pipekit_array import MeanScalar

# Shapes: B = 3 bands; H = 4 rows; W = 5 columns

image = np.ones((3, 4, 5), dtype=np.float32)              # → (B, H, W)

# today (pipekit-array)
# m = (1/(BHW)) ∑_{b,i,j} x_{bij}
total = MeanScalar()(image)                               # (B, H, W) → ()
# today (pipekit-array)
# m_b = (1/(HW)) ∑_{i,j} x_{bij}
per_band = MeanScalar(axis=(-2, -1))(image)               # (B, H, W) → (B,)
# today (pipekit-array)
# m_{ij} = (1/B) ∑_b x_{bij}
per_pixel = MeanScalar(axis=0, keepdims=True)(image)      # (B, H, W) → (1, H, W)
assert total.dtype == np.float32
```

**Contract:**

- `_apply(x)`: `xp = array_namespace(x); return xp.mean(x, axis=self.axis, keepdims=self.keepdims)`.
  There is no `dtype=` keyword, because the Array API `mean` signature accepts only `axis` and `keepdims`.
  To upcast before the reduction, compose with an `xp.astype` step in user code.
- It returns the backend's scalar or reduced-shape array.
- A list `axis` (as it comes back from JSON) is turned into a tuple.

**`get_config()`:** `{"axis": self.axis, "keepdims": self.keepdims}`.

**`compute_output_signature(sig)`:** drops the reduced axes from `sig.dims` (or sets their size to 1 if `keepdims=True`); if `axis=None` and `keepdims=False`, it returns `Signature(dims=(), dtype=sig.dtype)`.
An `axis` out of range for the signature's rank raises `ValueError`.

---

## `BatchAlong(inner: Operator, size: int, dim: str, pad: str = "none")`

Ships today in `pipekit_array.batching` (phase 3 of [dim_functions.md](dim_functions.md), #206).
It was renamed from the planned array-shaped `BatchedMap(inner, batch_size, axis)` by ADR A19, which supersedes A5: `pipekit_ops.parallel.BatchedMap` takes an iterable and calls `inner` on lists, so one name for both would fail in confusing ways when both are imported.

Split the input carrier along the named `dim` into chunks of `size`, apply `inner` to each chunk, and concatenate the results along `dim`.
It is the carrier-shaped analogue of `pipekit_ops.parallel.BatchedMap` (which takes an iterable).

```python
import numpy as np
from pipekit_array import BatchAlong, DimArray, Reduce

# Shapes: N = 10 samples (sample); B = 2 bands (band); X = 4 pixels (x); m = 4

data = np.ones((10, 2, 4))                                # → (N, B, X)
# today (pipekit-array)
cube = DimArray(data, ("sample", "band", "x"))            # (N, B, X)

# today (pipekit-array)
# y = concat(inner(x_{0:m}), inner(x_{m:2m}), …) along sample
per_chunk = BatchAlong(Reduce("sum", "band"), size=4, dim="sample")
out = per_chunk(cube)                                     # (N, B, X) → (N, X)
# today (pipekit-array)
dropped = BatchAlong(Reduce("sum", "band"), 4, "sample", pad="drop")(cube)
assert out.shape == (10, 4) and dropped.shape == (8, 4)   # drop: (N, B, X) → (8, X)
```

**Contract:**

- `_apply(x)`: compute `n`, the length of `dim`; for each chunk `start, end` in `range(0, n, size)`, `chunks.append(inner(isel(x, **{dim: slice(start, end)})))`; return `concat(chunks, dim)`.
- `inner` must return carriers with the same dims, `dim` included; `concat` re-aligns the chunks.
- For inputs where `n <= size`, this is a single `inner(x)` call (no overhead).
- `pad` takes ADR A10's policy strings:
  - `"none"` leaves the last chunk short;
  - `"last"` pads it with copies of its last item, so `jit` compiles once, and drops the padding from the output;
  - `"drop"` drops the short last chunk.

**As built:**

- `size` must be positive and `pad` one of `"none"`, `"last"`, `"drop"`; otherwise the constructor raises `ValueError`, and a non-`Operator` `inner` raises `TypeError`.
- `concat` drops `dim`'s labels, so `BatchAlong` sets them back when `inner` keeps one entry per input entry; on a labelled `dim` with an adapter that has no `relabel`, it raises `TypeError`.
- `pad="drop"` raises `ValueError` when every entry would be dropped (the length of `dim` is less than `size`).

**`get_config()`:**

```python
{
    "inner": nested_config(self.inner),  # {"module", "class", "config"}
    "size": self.size,
    "dim": self.dim,
    "pad": self.pad,
}
```

**`compute_output_signature(sig)`:** delegates to `inner` on a single-chunk signature; if `inner` keeps `dim`, the output signature is `inner`'s with the original length of `dim` (trimmed to whole chunks for `pad="drop"`).
It returns None when `dim`'s length is unknown or `inner` does not keep `dim`.

**Use case:** running a GPU-bound model on a large array that doesn't fit in device memory.
The split, apply and concatenate is the inference-time equivalent of a training data loader's batching.
