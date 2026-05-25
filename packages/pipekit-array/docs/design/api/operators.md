---
status: draft
version: 0.1.0
---

# Operators — data flow

The seven operators that move arrays through a pipeline:
combinators, the reducer, the array-shaped `BatchedMap`, the
`Histogram` controller, and the `Subsample` geometry op.

## `ApplyToBands(inner: Operator, axis: int = 0)`

Split the input along `axis`, apply `inner` to each slice, stack the
results back along `axis`. The canonical "apply a band-wise
operation" combinator.

```python
from pipekit_array.combinators import ApplyToBands
from pipekit_array.qc import AssertValueRange


# Apply value-range assertion per band; axis 0 == band dim.
checked = ApplyToBands(AssertValueRange(0.0, 1.0), axis=0)
out = checked(reflectance_arr)  # shape (B, H, W)
```

**Contract:**

- `_apply(x)`: `xp = array_namespace(x)`; for each `i`,
  `chunks.append(inner(xp.take(x, i, axis=axis)))`; return
  `xp.stack(chunks, axis=axis)`.
- `inner` must return arrays of the same shape *minus* the split
  axis. (`xp.stack` re-adds it.) If `inner` returns scalars,
  `axis=0` and the output is a 1-D array. If `inner` reduces
  ambiguously, you get a `ValueError` from `xp.stack`.
- `inner` must return arrays of the same dtype. (`xp.stack`
  enforces.)

**`get_config()`:**

```python
{
    "inner": {"class": type(self.inner).__name__, "config": self.inner.get_config()},
    "axis": self.axis,
}
```

**`compute_output_signature(sig)`** : delegates to `inner` on the
per-band slice and re-adds the split axis if `inner` returns a
signature.

**Why this is in pipekit-array, not pipekit core:** the
split / stack mechanics use Array API functions
(`xp.take`, `xp.stack`); they're not carrier-agnostic. The pipekit
core combinator that's closest is `Fanout`, which takes a list of
operators rather than slicing one carrier.

---

## `StackAlong(axis: int = 0)`

Stack a list-of-arrays along a new axis. Thin wrapper over `xp.stack`
so the operation composes into a pipeline.

```python
out = StackAlong(axis=0)([arr_a, arr_b, arr_c])  # shape (3, H, W)
```

**Contract:**

- Input: a `list`/`tuple` of arrays, all the same shape, all the
  same backend (`array_namespace(*xs)` asserts).
- Output: a single array; `xp.stack(xs, axis=axis)`.

**`get_config()`:** `{"axis": self.axis}`.

**Failure mode:** mixed-backend input list raises `TypeError` via
`array_namespace`. Mismatched shapes raise the backend's stack error.

---

## `ConcatenateAlong(axis: int = 0)`

Concatenate a list-of-arrays along an existing axis. Thin wrapper
over `xp.concat` (the [Array API standard][api-concat] name; numpy
also exposes it as `concat` from 2.0+ alongside the legacy
`concatenate` alias).

[api-concat]: https://data-apis.org/array-api/latest/API_specification/generated/array_api.concat.html

```python
out = ConcatenateAlong(axis=0)([arr_a, arr_b])  # shape (2*N, H, W) if axis=0
```

**Contract:**

- Input: a `list`/`tuple` of arrays; all the same shape *except*
  along `axis`; all the same backend.
- Output: a single array; `xp.concat(xs, axis=axis)`.

**`get_config()`:** `{"axis": self.axis}`.

**Distinction from `StackAlong`:** `StackAlong` *adds* a new axis;
`ConcatenateAlong` joins along an *existing* axis. Same as numpy's
`stack` vs `concat`.

---

## `Subsample(stride: int | tuple[int, ...] = 10, axes: tuple[int, ...] = (-2, -1))`

Stride-decimate the input along `axes`. Defaults to the last two
axes (H, W in remote-sensing convention) with stride 10.

```python
out = Subsample(stride=2)(image)  # half-res via slice [..., ::2, ::2]
out = Subsample(stride=(4, 2), axes=(-2, -1))(image)  # different strides
```

**Contract:**

- `_apply(x)`: build a slice tuple of `slice(None)` everywhere
  except `axes`, where `slice(None, None, stride[i])` applies.
  Return `x[tuple(slices)]`.
- `stride` may be an int (broadcast across `axes`) or a tuple of
  ints (one per axis).

**`get_config()`:** `{"stride": self.stride, "axes": self.axes}`.

**`compute_output_signature(sig)`** : returns a `Signature` with
the per-axis sizes divided by their strides (using `ceil`-div).
Other axes unchanged.

**Per-backend note:** All five backends support `__getitem__` with
slice tuples per the Array API. No special handling.

---

## `Histogram(bins: int | array | "auto" = 10, range: tuple[float, float] | None = None)`

Controller that captures distributions at named tap sites. Same
pattern as `pipekit.observe.Histogram` but with array-aware defaults
(`to_array = xp.reshape(x, (-1,))` is implicit).

```python
hist = Histogram(bins=50, range=(0.0, 1.0))
pipe = Subsample(stride=2) | hist.at("post_subsample") | MeanScalar()
pipe(image)
hist.report()  # {"post_subsample": (counts, edges)}
```

**Contract:**

- The class itself is not an `Operator`. `Histogram.at(key)`
  returns a small `_HistAt(Operator)` that:
  1. Calls `array_namespace(x)`, gets `xp`.
  2. Flattens via `xp.reshape(x, (-1,))`.
  3. Computes bin counts via `xp.searchsorted(edges, flat) - 1`
     followed by a per-bin `xp.sum(indices == i)`. Edges are
     pre-computed once per `Histogram` controller from `bins` and
     `range`. Both `xp.searchsorted` and `xp.sum` are in the
     [Array API standard][api-search]; no histogram primitive is
     needed.
  4. Stores `(counts, edges)` under `key` in the controller's
     `captures` dict. Counts and edges are stored as whatever the
     backend produced (e.g. `jax.Array` from a JAX input); the user
     calls `np.asarray(counts)` at inspection time if they want
     plain numpy.
  5. Returns `x` unchanged (it's a `Tap`-shaped operator).
- `Histogram.report()` returns the captures dict (a shallow copy).

[api-search]: https://data-apis.org/array-api/latest/API_specification/generated/array_api.searchsorted.html

**`get_config()`** (on `_HistAt`): `{"key": self.key}`. The
controller itself isn't serialised — it's an interactive object,
and `_HistAt` carries `forbid_in_yaml = True` because it closes over
the controller (see `architecture.md` §5).

**Per-backend note:** `xp.histogram` is **not** in the Array API
spec, so the implementation uses `xp.searchsorted` + `xp.sum` per
bin (both spec-conformant). The performance is O(n_bins) bin passes
over the array instead of one optimised numpy `np.histogram` call;
acceptable for the inspection / debugging use case the controller
serves. If the use case grows beyond inspection (large-N online
statistics), see Q4 in `boundaries.md` for the streaming-counters
upgrade path.

---

## `MeanScalar(axis: int | tuple[int, ...] | None = None, keepdims: bool = False)`

Reduce via `xp.mean`. Defaults to a full reduction (`axis=None`)
yielding a scalar (`ndim == 0`). Output dtype matches the input's
floating dtype per the Array API `mean` contract — see the
[spec][api-mean].

[api-mean]: https://data-apis.org/array-api/latest/API_specification/generated/array_api.mean.html

```python
out = MeanScalar()(image)               # scalar
out = MeanScalar(axis=(-2, -1))(image)  # per-band mean: shape (B,)
out = MeanScalar(axis=0, keepdims=True)(image)  # shape (1, H, W)
```

**Contract:**

- `_apply(x)`: `xp = array_namespace(x); return xp.mean(x,
  axis=self.axis, keepdims=self.keepdims)`. No `dtype=` kwarg — the
  Array API `mean` signature only accepts `axis` and `keepdims`. To
  upcast before reduction, compose with an `xp.astype` step in user
  code or wait for a v0.2 `Astype` operator (Q7 of `boundaries.md`).
- Returns the backend's scalar / reduced-shape array.

**`get_config()`:** `{"axis": self.axis, "keepdims": self.keepdims}`.

**`compute_output_signature(sig)`** : drops the reduced axes from
`sig.dims` (or sets their size to 1 if `keepdims=True`); if
`axis=None` and `keepdims=False`, returns `Signature(dims=(),
dtype=sig.dtype)`.

---

## `BatchedMap(inner: Operator, batch_size: int = 8, axis: int = 0)`

Split the input array along `axis` into chunks of `batch_size`,
apply `inner` to each chunk, concatenate the results along `axis`.
The array-shaped analogue of `pipekit.parallel.BatchedMap` (which
takes an iterable). See ADR A5 for the boundary.

```python
out = BatchedMap(model_op, batch_size=32)(x)  # x.shape = (N, ...)
```

**Contract:**

- `_apply(x)`: compute `n = x.shape[axis]`; for each chunk
  `start, end` in `range(0, n, batch_size)`:
  `chunks.append(inner(xp.take(x, range(start, end), axis=axis)))`.
  Return `xp.concat(chunks, axis=axis)` (the [Array API standard][api-concat]
  name; see `ConcatenateAlong` above).
- `inner` must return arrays of the same shape minus the split-axis
  size (i.e. `inner` may change axis-0 length per chunk only via
  the chunk slice; `concatenate` re-aligns).
- For inputs where `x.shape[axis] <= batch_size`, this is a single
  `inner(x)` call (no overhead).

**`get_config()`:**

```python
{
    "inner": {"class": type(self.inner).__name__, "config": self.inner.get_config()},
    "batch_size": self.batch_size,
    "axis": self.axis,
}
```

**`compute_output_signature(sig)`** : delegates to `inner` on a
single-chunk signature; if `inner` preserves the split axis, the
output signature matches `inner`'s with the original `axis` size.

**Use case:** running a GPU-bound `ModelOp` on a large array that
doesn't fit in device memory. The split / apply / concatenate is the
inference-time equivalent of a training data loader's batching.
