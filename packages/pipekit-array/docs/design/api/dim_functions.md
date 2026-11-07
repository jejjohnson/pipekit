---
status: draft
version: 0.1.0
---

# Dim-level functions — the missing ones

The dim-level functions in [`dims.py`](../../../src/pipekit_array/dims.py) select, reduce, stack and map by dim name on any carrier with a `pipekit_ops.NamedArrayAdapter`.
They cannot yet rename a dim, add or drop a length-1 dim, broadcast one carrier against another, merge `y x` into `pixel`, shift or difference along time, cut windows, coarsen, take a quantile or contract two carriers.
This page adds those as free functions in `dims` with operator twins in `dim_operators`, and checks every name and argument against xarray.

## 1. Decision

**Added: 16 functions, each with an operator twin, plus two operators (`ApplyAlong`, `BatchAlong`):**

- dims: `rename`, `expand_dims`, `squeeze`, `broadcast_like`, `merge_dims`, `split_dims`;
- index-only transforms: `pad`, `roll`, `shift`, `diff`, `cumreduce`, `where`;
- windows: `window`, `coarsen`;
- statistics and algebra: `quantile`, `dot`.

**Renamed or retired in the catalogue (none of them ships today):**

- the planned `Subsample` is `ISel` with a stepped slice, which ships today;
- the planned `ApplyToBands(inner, axis)` becomes `ApplyAlong(inner, dim)`, the serialisable twin of `MapOver`;
- the planned array-shaped `BatchedMap(inner, batch_size, axis)` becomes `BatchAlong(inner, size, dim)`, which supersedes ADR A5.

**Not changed:**

- The joining rule (GA-Q3, in the module docstring of [`dims.py`](../../../src/pipekit_array/dims.py)): a function never reads coordinate values or geometry.
  Every new function works on positions; labels only follow the data.
- `NamedArrayAdapter` keeps its seven required methods ([`adapters.py`](../../../../pipekit-ops/src/pipekit_ops/adapters.py)).
  Every new function has a generic path through `data`, `isel` and `rewrap`; an adapter may add a fast path of the same name, as today.
  One optional method is added, `relabel`, for the six functions that put labels on a dim `rewrap` cannot label (section 3.0, decision A22).
- `DimArray` gets no methods ([`dimarray.py`](../../../src/pipekit_array/dimarray.py)).
- Element-wise functions (`fillna`, `clip`, `isin`, `isnull`, `round`, `astype`) stay `combine` calls: they touch no dim.
- `groupby`, `resample`, `interp`, `reindex` and anything else that selects by label value stay in xarray (and coordax).

## 2. The running example

A Sentinel-2-like cube: $T = 12$ scenes, $B = 13$ bands, $Y = X = 256$ pixels, dims `time band y x`.
Write $x_{tbij}$ for the value of scene $t$, band $b$, row $i$ and column $j$, with $0 \le t < T$, $0 \le b < B$, $0 \le i < Y$ and $0 \le j < X$.
The `band` dim has labels `B1 … B12, B8A`; the other dims have none.
A cloud mask $v_{tij}$ (True = valid) has dims `time y x`.

What works today:

```python
import numpy as np
from pipekit_array import DimArray, isel, map_over, reduce

# Shapes: T = 12 scenes; B = 13 bands; Y = X = 256 pixels

rng = np.random.default_rng(0)
data = rng.random((12, 13, 256, 256), dtype=np.float32)
bands = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11")
bands = (*bands, "B12")
dims = ("time", "band", "y", "x")
cube: DimArray["time band y x"] = DimArray(data, dims, {"band": bands})

# today (pipekit-array)
# x̃_{tbij} = x_{tb,2i,2j}
# (T, B, Y, X) → (T, B, Y/2, X/2)
every_other = slice(None, None, 2)
thinned: DimArray["time band y x"] = isel(cube, y=every_other, x=every_other)

# today (pipekit-array)
# m_{bij} = (1/T) ∑ₜ x_{tbij}
# (T, B, Y, X) → (B, Y, X)
mean: DimArray["band y x"] = reduce(cube, "mean", "time")
```

`thinned` is the whole of the planned `Subsample` (xarray's `thin`), so `Subsample` is not added.

## 3. The function catalogue

Columns: the signature (keyword dims, `x` positional-only as in `isel`); the semantics; what the adapter must provide; JAX behaviour; status.
"Generic" means the function runs on any adapter through `data`, `isel` and `rewrap`, with the array namespace from `namespace_of` ([`dims.py`](../../../src/pipekit_array/dims.py)).
"Static" lists the arguments that fix output shapes, so they must be Python values under `jax.jit`.
Every function in the tables is `proposed` unless marked `today`; phase 1 (section 3.1) ships in [`dim_shapes.py`](../../../src/pipekit_array/dim_shapes.py), phase 2 (section 3.2) in [`dim_shifts.py`](../../../src/pipekit_array/dim_shifts.py), phase 3 (section 3.3) in [`dim_windows.py`](../../../src/pipekit_array/dim_windows.py), phase 4 (section 3.4) in [`dim_algebra.py`](../../../src/pipekit_array/dim_algebra.py).

### 3.0 Labels on new or renamed dims

`rewrap(x, data, dims)` keeps labels by dim name: `DimArrayAdapter.rewrap` copies `x.labels[dim]` for each output dim whose size is unchanged ([`dimarray.py`](../../../src/pipekit_array/dimarray.py)), and the xarray adapter keeps a coordinate when all its dims survive with their sizes ([`xarray.py`](../../../src/pipekit_array/xarray.py)).
So `rewrap` alone cannot label a dim whose name or labels are new, and six functions need exactly that:

| Function | Output dim | Its labels |
| --- | --- | --- |
| `rename` | the new name | the old dim's labels, unchanged |
| `merge_dims` | the merged dim | tuples of the source labels, positions for an unlabelled source |
| `split_dims` | each source dim | recovered from the tuples, or none |
| `quantile` | `quantile` (sequence `q` only) | the values of `q` |
| `broadcast_like` | each dim taken from `other` | `other`'s labels |
| `dot` | each dim taken from an input after the first | that input's labels |

With `rewrap` only, `rename(cube, band="channel")` on a `DimArray` would drop the band labels, because `rewrap` looks them up under `channel`.

The fix is one optional adapter method, in a runtime-checkable `LabelledAdapter` protocol next to `NamedArrayAdapter` in `pipekit_ops.adapters`:

```python
# proposed (A22)
def relabel(self, x: Any, labels: Mapping[str, tuple[Any, ...] | None]) -> Any:
    """``x`` with the labels of the named dims set; None removes a dim's labels."""
```

Each of the six generic paths computes the output labels from `adapter.labels(...)` on its inputs, calls `rewrap`, then calls `relabel` with only the dims in the table above.
`DimArrayAdapter.relabel` rebuilds the `DimArray` with the merged label mapping, and checks each label tuple against its dim's size.
The xarray adapter's `relabel` is `assign_coords` for a tuple and `drop_vars` for None.
An adapter without `relabel` still runs the six functions when no output labels would be set; when some would be, the function raises `TypeError` naming the adapter and the dim, so labels are never lost silently.

As built in phase 1: `merge_dims` gives the merged dim no labels when none of its sources has any (positions alone carry nothing, and an adapter without `relabel` can then merge unlabelled dims), and `split_dims` returns a part unlabelled when its recovered labels are exactly its positions $0, \dots, n-1$, so both round trips are exact.
`relabel` is detected by name (`getattr(adapter, "relabel", None)`, as `_fast_path` does): pipekit-array never imports `LabelledAdapter` at runtime, so it still runs on a pipekit-ops without it.

### 3.1 Dims

| Function | Signature | Semantics | Adapter requirement | JAX | Status |
| --- | --- | --- | --- | --- | --- |
| `rename` | `rename(x, /, **names: str)` | New dim names; data, order and labels unchanged. | Generic: `rewrap` with new names, then `relabel` with the old dims' labels under the new names (section 3.0). A fast path is still needed on carriers with more metadata keyed by dim than labels (xarray's non-index coords and encodings, geoarray's grid). | jit, vmap safe; names static. | today |
| `expand_dims` | `expand_dims(x, dim: str \| Sequence[str], /, *, axis: int = 0)` | Insert length-1 dims at `axis`. | Generic: `xp.expand_dims`, `rewrap`. | jit, vmap safe; `dim`, `axis` static. | today |
| `squeeze` | `squeeze(x, dim: str \| Sequence[str], /)` | Drop named length-1 dims; raise if a size is not 1. `dim` is required. | Generic: `isel(x, dim=0)` after the size check. | jit, vmap safe. | today |
| `broadcast_like` | `broadcast_like(x, other, /, *, exclude: Sequence[str] = ())` | Add `other`'s missing dims to `x`, in front, in `other`'s order; shared dims must agree exactly. | Generic: `check_compatible` on the shared dims, `xp.expand_dims`, `xp.broadcast_to`, `rewrap`, then `relabel` with `other`'s labels on the added dims. | jit, vmap safe; `other`'s shape static. | today |
| `merge_dims` | `merge_dims(x, /, **dims: Sequence[str])` | `merge_dims(cube, pixel=("y", "x"))`: the listed dims become one, placed last, in C order. Labels become tuples. | Generic: `transpose`, `xp.reshape`, `rewrap`, then `relabel` with the tuple labels. | jit, vmap safe; names static. | today |
| `split_dims` | `split_dims(x, /, **dims: Mapping[str, int])` | `split_dims(p, pixel={"y": 256, "x": 256})`: one dim becomes several, in place; the sizes must multiply to its length. Tuple labels are split back. | Generic: `xp.reshape`, `rewrap`, then `relabel` with the labels split from the tuples. | jit, vmap safe; sizes static. | today |

### 3.2 Index-only transforms

Each keeps the dims it was given.
None reads a label; labels stay on their positions (xarray's default, `roll_coords=False`).

| Function | Signature | Semantics | Adapter requirement | JAX | Status |
| --- | --- | --- | --- | --- | --- |
| `pad` | `pad(x, /, *, mode: str = "constant", constant_values: float \| None = None, **pad_width: int \| tuple[int, int])` | `xp.pad` on named dims. Modes: `constant`, `edge`, `reflect`, `wrap`. A padded dim loses its labels. | Generic: `xp.pad`, `rewrap` (which drops labels whose size changed). | jit, vmap safe; widths, `mode` static. | today |
| `roll` | `roll(x, /, **shifts: int)` | Cyclic shift; labels stay. | Generic: `xp.roll`. | jit, vmap safe; a traced shift also works. | today |
| `shift` | `shift(x, /, *, fill_value: float \| None = None, **shifts: int)` | Positive shifts move data to higher positions; vacated entries are `fill_value`. | Generic: `isel` slice, `xp.full`, `xp.concat`, `rewrap`. | jit, vmap safe; shifts static. | today |
| `diff` | `diff(x, dim: str, /, n: int = 1, *, label: str = "upper")` | $n$-th forward difference; the result's labels are the minuend's (`"upper"`) or subtrahend's (`"lower"`). | Generic: two `isel` slices, `rewrap` on the labelled slice. | jit, vmap safe; `n` static. | today |
| `cumreduce` | `cumreduce(x, fn: Reduction, dim: str)` | A cumulative function along `dim`: `"cumulative_sum"`, `"cumulative_prod"`, `"nancumsum"`, or a callable taking `axis=` that keeps the length. | Generic: the namespace function, `rewrap`. The dask namespace maps `cumulative_sum` to `cumsum`, as `permute_dims` maps to `transpose` today. | jit, vmap safe. | today |
| `where` | `where(x, cond, /, other: float \| None = None)` | Keep `x` where `cond` is True, `other` elsewhere (default NaN). `cond` aligns by dim name, as `reduce(where=)` does. | Generic: the existing mask alignment (`_aligned_mask`), `xp.where`, `rewrap`. | jit, vmap safe. | today |

### 3.3 Windows

| Function | Signature | Semantics | Adapter requirement | JAX | Status |
| --- | --- | --- | --- | --- | --- |
| `window` | `window(x, /, *, window_dim: str = "window", stride: int = 1, **sizes: int)` | One named dim, windows of length $S$ with step $s$; the window becomes a new last dim. Full windows only. A window carries the label of its last entry. | Generic: $S$ strided slices of the data, `xp.stack` on the last axis, `rewrap` on the end-position slice. Fast path on xarray: `rolling(...).construct(...).isel(...)`. | jit, vmap safe; `sizes`, `stride` static. | today |
| `coarsen` | `coarsen(x, fn: Reduction, /, *, boundary: str = "exact", **windows: int)` | Non-overlapping blocks of $k$, reduced by `fn`: `split_dims` then `reduce`. `boundary="trim"` drops the tail. The coarsened dim loses its labels. | Generic: `isel` (trim), `split_dims`, `reduce`. | jit, vmap safe; `windows` static. | today |
| `map_over` | `map_over(x, fn, dim)` | `fn` per item along `dim`. | Generic; fast path optional. | Python loop unless an adapter vmaps. | today |

### 3.4 Statistics and algebra

| Function | Signature | Semantics | Adapter requirement | JAX | Status |
| --- | --- | --- | --- | --- | --- |
| `quantile` | `quantile(x, q: float \| Sequence[float], dim: str \| Sequence[str], /, *, method: str = "linear", skipna: bool = False)` | Quantiles over `dim`; a sequence `q` adds a leading dim `quantile` labelled with `q`, a float drops it. | Generic: `xp.quantile` or `xp.nanquantile`, `rewrap`, then `relabel` with `quantile=tuple(q)` for a sequence `q`. Dask needs `dim` in one chunk. | jit, vmap safe; `q`, `method` static. JAX supports `linear`, `lower`, `higher`, `midpoint`, `nearest`, `inverted_cdf`. | today |
| `dot` | `dot(*xs, dim: str \| Sequence[str])` | Sum of the product over `dim`; output dims in order of first appearance, `dim` removed. A dim shared by several inputs and not in `dim` is a batch dim: it appears once in the output, as in `xr.dot`. `dim` is required. | Generic: a named-index `xp.einsum` built from the dim names (below), `check_compatible` on the shared dims, `rewrap` on the first input, then `relabel` with the labels of dims taken from later inputs. | jit, vmap safe. | today |


`dot` cannot be built on `xp.tensordot`.
`tensordot` contracts pairs of axes and puts every other axis of each input side by side, so a dim on both inputs that is not contracted appears twice.
For inputs with dims `(time, band)` and `(time, band, component)`, contracted over `band`, `tensordot` returns `(time, time, component)`, not `(time, component)`; contracting `time` as well would remove the dim `dot` must keep.

The generic path writes one `einsum` subscript from the dim names instead.
Each distinct dim gets one letter; an input's subscript lists its dims in axis order; the output lists every dim that is not in `dim`, in order of first appearance.

```python
# Shapes: T = 12 scenes; B = 13 bands; C = 3 components

# proposed (A17): what dot builds for the two inputs above
# s_{tc} = ∑_b x_{tb} W_{tbc}
# letters: time → a, band → b, component → c
subscripts = "ab,abc->ac"
# (T, B), (T, B, C) → (T, C)
out_data = xp.einsum(subscripts, x_data, w_data)
```

`time` is a batch dim here: it is shared, kept, and not summed.
NumPy, JAX and dask all provide `einsum` on the namespaces `namespace_of` resolves, though it is not in the Array API standard.
For a namespace without `einsum`, the fallback is to `broadcast_like` every input to the union of their dims, multiply, and `xp.sum` over `dim`; this is exact but holds the full product in memory.

### 3.5 Operator twins

Every twin lives in [`dim_operators.py`](../../../src/pipekit_array/dim_operators.py), takes only names and numbers, round-trips through `dumps` / `loads`, and implements `compute_output_signature` from `Signature.drop_dims` and `Signature.replace_dims`.

| Function | Operator | Serialises | Output signature | Status |
| --- | --- | --- | --- | --- |
| `rename` | `Rename(**names)` | yes | names replaced | today |
| `expand_dims` | `ExpandDims(dim, axis=0)` | yes | size-1 dims added | today |
| `squeeze` | `Squeeze(dim)` | yes | dims dropped | today |
| `broadcast_like` | `BroadcastLike(exclude=())`, called `op(x, other)` | yes | other's dims added | today |
| `merge_dims` | `MergeDims(**dims)` | yes | product size | today |
| `split_dims` | `SplitDims(**dims)` | yes | sizes from config | today |
| `pad` | `Pad(mode, constant_values, **pad_width)` | yes | sizes grown | today |
| `roll` | `Roll(**shifts)` | yes | unchanged | today |
| `shift` | `Shift(fill_value, **shifts)` | yes | unchanged | today |
| `diff` | `Diff(dim, n=1, label="upper")` | yes | size minus $n$ | today |
| `cumreduce` | `CumReduce(fn: str, dim)` | yes (by name) | unchanged | today |
| `where` | `Where(other=None)`, called `op(x, cond)` | yes | unchanged | today |
| `window` | `Window(window_dim, stride, **sizes)` | yes | count and window size | today |
| `coarsen` | `Coarsen(fn: str, boundary, **windows)` | yes (by name) | size divided by $k$ | today |
| `quantile` | `Quantile(q, dim, method, skipna)` | yes | `quantile` dim added or not | today |
| `dot` | `Dot(dim)`, called `op(*xs)` | yes | contracted dims removed | today |
| `map_over` | `ApplyAlong(inner: Operator, dim)` | yes, if `inner` does | from `inner` | today |

`compute_output_signature` takes one input signature, so `BroadcastLike`'s takes the pair `(x's, other's)` and returns None for `x`'s alone; `MergeDims` and `SplitDims` store their merges and parts as ordered lists in config, since JSON objects sort their keys and the order sets the output dims.

`Diff` here clashes with the planned QC operator `Diff(reference, atol, rtol)` in [`qc.md`](qc.md).
Neither ships yet; decision A21 names the QC one `AssertClose`.

## 4. Worked sketch

The cube from section 2, with the proposed functions.
$v_{tij}$ is the cloud mask; $\mathrm{NDVI}_{tij} = (x_{t,B8,ij} - x_{t,B4,ij}) / (x_{t,B8,ij} + x_{t,B4,ij})$; $S = 3$ and $s = 1$ are the window size and stride; $k = 4$ is the coarsening factor; $W_{bc}$ is a $B \times C$ matrix of $C = 3$ components.
In the code, `valid` is $v$ as a `DimArray["time y x"]` and `weights` is $W$ as a `DimArray["band component"]`.

```python
# today (pipekit-array)
from pipekit_array import combine, reduce, sel

# proposed (this page)
from pipekit_array import broadcast_like, coarsen, cumreduce, diff, dot
from pipekit_array import merge_dims, quantile, split_dims, where, window

# Shapes: T = 12; B = 13; Y = X = 256; P = Y·X = 65,536 pixels; C = 3 components

# proposed (A16)
# x̂_{tbij} = x_{tbij} if v_{tij} else NaN
# (T, B, Y, X), (T, Y, X) → (T, B, Y, X)
clear: DimArray["time band y x"] = where(cube, valid)

# today (pipekit-array)
# m_{bij} = median over t of x̂_{tbij}, ignoring NaN
# (T, B, Y, X) → (B, Y, X)
median: DimArray["band y x"] = reduce(clear, "nanmedian", "time")

# proposed (A11)
# m̃_{tbij} = m_{bij}
# (B, Y, X) → (T, B, Y, X)
median_t: DimArray["time band y x"] = broadcast_like(median, clear)

# today (pipekit-array)
# a_{tbij} = x̂_{tbij} − m̃_{tbij}
# (T, B, Y, X) → (T, B, Y, X)
anomaly: DimArray["time band y x"] = combine(np.subtract, clear, median_t)

# proposed (A18)
# Q_{qbij} = quantile q of x_{·bij} over t, q ∈ {0.1, 0.9}
# (NaN propagates: skipna=False, the default)
# (T, B, Y, X) → (2, B, Y, X)
spread: DimArray["quantile band y x"] = quantile(cube, [0.1, 0.9], "time")

# today (pipekit-array)
# x_{t,B8,ij}
# (T, B, Y, X) → (T, Y, X)
nir: DimArray["time y x"] = sel(clear, band="B8")

# today (pipekit-array)
# x_{t,B4,ij}
# (T, B, Y, X) → (T, Y, X)
red: DimArray["time y x"] = sel(clear, band="B4")

# today (pipekit-array)
# NDVI_{tij} = (nir − red) / (nir + red)
# (T, Y, X), (T, Y, X) → (T, Y, X)
ndvi: DimArray["time y x"] = combine(lambda a, b: (a - b) / (a + b), nir, red)

# proposed (A12)
# Δ_{tij} = NDVI_{t+1,ij} − NDVI_{tij}, labelled t + 1
# (T, Y, X) → (T − 1, Y, X)
change: DimArray["time y x"] = diff(ndvi, "time")

# proposed (A12)
# c_{tij} = ∑_{τ ≤ t} Δ_{τij}
# (T − 1, Y, X) → (T − 1, Y, X)
drift: DimArray["time y x"] = cumreduce(change, "cumulative_sum", "time")

# proposed (A15)
# w_{n,ij,r} = NDVI_{n s + r, ij}, 0 ≤ r < S, n = ⌊(T − S)/s⌋ + 1 = 10
# (T, Y, X) → (10, Y, X, 3)
windows: DimArray["time y x window"] = window(ndvi, time=3, stride=1)

# today (pipekit-array)
# ŵ_{nij} = (1/S) ∑_r w_{n,ij,r}
# (10, Y, X, 3) → (10, Y, X)
smooth: DimArray["time y x"] = reduce(windows, "nanmean", "window")

# proposed (A15)
# z_{t,i′,j′} = (1/k²) ∑_{r,u < k} NDVI_{t, k i′ + r, k j′ + u}
# (T, Y, X) → (T, Y/4, X/4)
coarse: DimArray["time y x"] = coarsen(ndvi, "nanmean", y=4, x=4)

# proposed (A14)
# p = i·X + j
# (T, B, Y, X) → (T, B, P)
pixels: DimArray["time band pixel"] = merge_dims(clear, pixel=("y", "x"))

# proposed (A17)
# s_{tcp} = ∑_b W_{bc} x̂_{tbp}
# (T, B, P), (B, C) → (T, P, C)
scores: DimArray["time pixel component"] = dot(pixels, weights, dim="band")

# proposed (A14)
# (i, j) = (⌊p / X⌋, p mod X)
# (T, P, C) → (T, Y, X, C)
grid = {"y": 256, "x": 256}
score_map: DimArray["time y x component"] = split_dims(scores, pixel=grid)
```

`smooth` is xarray's `ndvi.rolling(time=3).mean()` without the two leading NaN scenes.
To match xarray's length, `pad(ndvi, time=(2, 0))` first.

## 5. Parity with xarray

Read from pydata/xarray at commit `5c54b9fd1db952142595c68af89eb6e8fca77c8e` (committed 2026-11-05), files `xarray/core/dataarray.py`, `xarray/core/common.py`, `xarray/core/variable.py`, `xarray/computation/rolling.py` and `xarray/computation/computation.py`; behaviour spot-checked on xarray 2026.4.0 from the workspace lock.

Status: **exists** (ships today), **proposed** (this page), **compose** (no new function; the composition is given), **out** (stays in xarray, with the reason).

| xarray | pipekit-array | Status | Notes and divergences |
| --- | --- | --- | --- |
| `isel(indexers, drop, missing_dims)` | `isel(x, **indexers)` | exists | No `missing_dims`; an unknown dim raises. |
| `sel(indexers, method, tolerance, drop)` | `sel(x, **labels)` | exists | Exact labels only; `method` and `tolerance` read label values. |
| `head`, `tail`, `thin` | `isel` with a slice | compose | `thin(y=2)` is `isel(x, y=slice(None, None, 2))`; retires `Subsample`. |
| `drop_isel` | `isel` with kept positions | compose | |
| `transpose(*dim, transpose_coords, missing_dims)` | `transpose(x, order)` | exists | No `...`; `order` names every dim. |
| `T` | `transpose` | exists | |
| `rename(new_name_or_name_dict, **names)` | `rename(x, **names)` | proposed | Dims only: no array name, no coordinate names. |
| `swap_dims(dims_dict)` | — | out | Swaps a dim for a coordinate's values. |
| `expand_dims(dim, axis, create_index_for_new_dim, **dim_kwargs)` | `expand_dims(x, dim, axis=0)` | proposed | Size 1 only; size $> 1$ is `broadcast_like` or `einx.repeat`. |
| `squeeze(dim, drop, axis)` | `squeeze(x, dim)` | proposed | `dim` required: squeezing every length-1 dim drops a batch of one. |
| `broadcast_like(other, exclude)`, `xr.broadcast` | `broadcast_like(x, other, exclude=())` | proposed | Shared dims must agree exactly; xarray outer-joins. Same placement: new dims first. |
| `stack(dim, create_index, index_cls, **dim_kwargs)` | `merge_dims(x, **dims)` | proposed | Same placement (last) and C order; tuple labels instead of a `MultiIndex`. pipekit's `stack` is `xr.concat` along a new dim. |
| `unstack(dim, fill_value, sparse)` | `split_dims(x, **dims)` | proposed | Sizes given, not read from an index; no fill (the product must be exact). |
| `coarsen(...).construct(...)` | `split_dims` | compose | `split_dims(x, time={"time": 4, "window": 3})`. |
| `pad(pad_width, mode, stat_length, constant_values, end_values, reflect_type)` | `pad(x, mode, constant_values, **pad_width)` | proposed | Four modes; labels dropped (xarray pads coordinates with NaN); integers need `constant_values`. |
| `roll(shifts, roll_coords=False)` | `roll(x, **shifts)` | proposed | Labels never roll (xarray's default). |
| `shift(shifts, fill_value)` | `shift(x, fill_value, **shifts)` | proposed | Integers need `fill_value`; xarray promotes them to float. |
| `diff(dim, n, label)` | `diff(x, dim, n, label)` | proposed | Same arguments and labels, except `label="lower"` with $n \ge 2$: xarray applies `label` to the first step only (its recursion drops it); `diff` keeps positions $0, \dots, N - n - 1$. |
| `cumsum`, `cumprod`, `cumulative(dim).sum()` | `cumreduce(x, fn, dim)` | proposed | No implicit `skipna`: xarray's `cumsum` skips NaN by default; ask for `"nancumsum"`. |
| `rolling(dim, min_periods, center).construct(window_dim, stride, fill_value)` | `window(x, window_dim, stride, **sizes)` | proposed | Full windows only, no NaN fill, no `center`. Equals `construct(...).isel(time=slice(S - 1, None, s))`. |
| `rolling(...).mean()` and the other reductions | `window` then `reduce` | compose | `min_periods` is `reduce(where=)` or a nan-reduction. |
| `rolling_exp` | — | out | A recurrence over time: `pipekit_cycle.Recurrence` or user code. |
| `coarsen(dim, boundary, side, coord_func)` | `coarsen(x, fn, boundary, **windows)` | proposed | `boundary` in `exact`, `trim`; no `pad` (`pad` first); no `coord_func` (reads coordinate values). |
| `quantile(q, dim, method, keep_attrs, skipna)` | `quantile(x, q, dim, method, skipna)` | proposed | Same leading `quantile` dim; `skipna=False` by default; methods limited to JAX's six. |
| `reduce(func, dim)`, `mean`, `max`, … | `reduce(x, fn, dim, where=)` | exists | |
| `argmax`, `argmin` | `argreduce(x, fn, dim)` | exists | |
| `idxmax`, `idxmin` | `argreduce`, then labels on the host | out | The result is label values as data. |
| `isel` with array indexers | `take_along(x, index, dim)` | exists | |
| `where(cond, other, drop)`, `xr.where` | `where(x, cond, other)` | proposed | Same polarity; no `drop` (a data-dependent size); integers need `other`, as `shift` needs `fill_value`. |
| `fillna`, `clip`, `isin`, `isnull`, `notnull`, `round`, `astype` | `combine(fn, x)` | compose | Element-wise; no dim. |
| `dropna(dim, how, thresh)` | `reduce(..., where=)` | out | Output size depends on the data. |
| `ffill`, `bfill` | `cumreduce` then `take_along` | compose | Last valid index by a running maximum, then a gather. A function if asked for twice. |
| `interpolate_na`, `differentiate`, `integrate`, `cumulative_integrate` | — | out | Read coordinate spacing. |
| `sortby`, `rank`, `argsort` | `cumreduce` with a callable | out for now | `sortby` reads values of another variable; `sort` along a dim fits the `cumreduce` generic path (a function of `axis=` that keeps the length). |
| `dot(other, dim)`, `xr.dot(*arrays, dim)` | `dot(*xs, dim)` | proposed | `dim` required; xarray contracts every shared dim when it is `None`. |
| `concat` | `concat`, `stack` | exists | pipekit's `stack` is `xr.concat(..., dim=new)`. |
| `apply_ufunc` | `combine`, `KernelOp` | exists | |
| `groupby`, `resample` | — | out | Select by label value. |
| `reindex`, `reindex_like`, `interp`, `interp_like` | — | out | Read label or coordinate values. |
| `polyfit`, `curvefit`, `weighted` | — | out | `polyfit` reads coordinates; `weighted` is `dot` or `reduce(where=)` for the simple cases. |
| `set_index`, `reset_index`, `drop_sel`, `drop_duplicates`, `query`, `combine_first` | — | out | Label values or alignment. |
| `map_blocks` | — | out | A dask graph helper; the xarray adapter's concern. |

## 6. Decisions

ADR IDs continue [decisions.md](../decisions.md); A11 to A22 are appended there.

**A11. Scope: dim-level and index-only.**
Recommended: add only functions that change dims or move data by position; leave element-wise maths to `combine`.
Alternative: mirror xarray's method list; rejected, because half of it reads coordinate values (section 5).

**A12. xarray's names and argument names, with stated divergences.**
Recommended: `rename`, `expand_dims`, `squeeze`, `broadcast_like`, `pad`, `roll`, `shift`, `diff`, `coarsen`, `quantile`, `where` and `dot` take xarray's names and argument names, so a user crossing with `to_xarray` meets one vocabulary.
Divergences are deliberate and listed in section 5: index-only labels, exact agreement instead of outer joins, no implicit `skipna`, dtype preserved (R4 in [architecture.md](../architecture.md)), static sizes.
Alternative: Array API names (`permute_dims`, `cumulative_sum`); rejected for the dim-level layer, kept for the `fn` strings.

**A13. No methods on `DimArray`.**
Recommended: free functions only, as the Array API does.
A method would exist only on `DimArray`, so code written with it would not run on an `xr.DataArray` or a geoarray carrier; a free function runs on every registered carrier.
Alternative: thin method sugar that calls the free functions; rejected, because it invites carrier-specific code and doubles the documented surface.

**A14. `merge_dims` / `split_dims`, not `stack` / `unstack`.**
Recommended: new names, because `stack` already means xarray's `concat` along a new dim ([`dims.py`](../../../src/pipekit_array/dims.py)).
Placement follows xarray's `stack`: other dims first, the merged dim last, C order.
Labels: the merged dim's label is the tuple of the source labels, with positions for an unlabelled source (xarray's `MultiIndex` does the same); `split_dims` recovers each source's labels from the tuples, so the round trip is exact.
`einx.rearrange("(y x)")` stays as is: it names the dim `y_x` and drops labels ([`einx.py`](../../../src/pipekit_array/einx.py)).
Alternative: `unstack` reading sizes from the tuple labels; rejected, because the sizes must be static under `jit`.

**A15. Windows: `window` keeps full windows; `coarsen` is `split_dims` plus `reduce`; `ApplyAlong` replaces `ApplyToBands`.**
Recommended: `window` returns $n = \lfloor (N - S)/s \rfloor + 1$ full windows, where $N$ is the dim's length, $S$ the window size and $s$ the stride, with no NaN fill, so float32 stays float32 and integers work.
The generic path stacks raw strided slices, not carriers: the slices carry different labels and `stack` rightly refuses them (checked).
It copies the data $S$ times; the xarray adapter's fast path uses `rolling(...).construct`, which is a view.
`coarsen` is a composition and keeps no labels on the coarsened dim; xarray's `coord_func` averages coordinate values, which this layer never reads.
`ApplyToBands(inner, axis)` in [operators.md](operators.md) is `MapOver` by semantics but not by serialisation: `MapOver` holds a callable and sets `forbid_in_yaml`, a class-level flag checked by `build_operator` ([`_base/operator.py`](../../../../pipekit-ops/src/pipekit_ops/_base/operator.py)).
So `ApplyAlong(inner: Operator, dim)` is a separate class that serialises when `inner` does.
Alternative: xarray's `rolling(...).construct` semantics (length $N$, NaN-padded); rejected, because `pad` then `window` gives it in one more line, and the reverse needs a slice.

**A16. `where`, not `fill_where`.**
Recommended: `where(x, cond, other)` with xarray's polarity: True keeps.
A `fill_where(x, cond, value)` reads the other way (True fills), which is the polarity bug ADR A10 names `valid` to avoid.
`cond` aligns by dim name with the rule `reduce(where=)` already uses, so a `time y x` cloud mask applies to a `time band y x` cube.
Alternative: `combine` with `xp.where`; it works only when `cond` has every dim of `x`.

**A17. `dot`, not `contract`.**
Recommended: xarray's name, with `dim` required.
xarray contracts every shared dim when `dim` is `None`; a dim shared by accident (`time` on both) would vanish silently.
A shared dim not listed in `dim` is a batch dim and appears once in the output, so the generic path is a named-index `einsum`, not `tensordot` (section 3.4).
`einx.einsum` stays for patterns `dot` cannot write, behind the `[einx]` extra; `dot` needs no extra.
Alternative: `contract(x, y, over=...)`; rejected, because it adds a second vocabulary for one operation.

**A18. `quantile` is a function, not a `reduce` argument.**
`reduce(x, functools.partial(np.quantile, q=0.9), "time")` works today for one $q$, but picks NumPy ahead of the data, cannot serialise, and fails for a list of $q$, whose new dim `rewrap` cannot name.
Recommended: `quantile(x, q, dim)` with xarray's leading `quantile` dim, labelled with $q$.
Alternative: a `**kwargs` pass-through on `reduce`; rejected, because the operator twin could not compute its output signature.

**A19. `BatchAlong` supersedes ADR A5.**
Recommended: the planned array-shaped `BatchedMap(inner, batch_size, axis)` becomes `BatchAlong(inner, size, dim, pad="none")`.
`pipekit_ops.BatchedMap` takes an iterable and calls `inner` on lists ([`parallel.py`](../../../../pipekit-ops/src/pipekit_ops/parallel.py)); the array one would split one carrier along a dim.
Two classes with one name and different inputs fail in confusing ways when both are imported, and the array one has not shipped, so the rename costs nothing now.
`pad` takes ADR A10's policy strings: `"last"` pads the short final chunk so `jit` compiles once.
Alternative: keep A5 and rely on the import path; rejected.

**A20. Out of scope: label-value operations.**
Recommended: `groupby`, `resample`, `swap_dims`, `reindex`, `interp`, `idxmax`, `polyfit`, `differentiate`, `interpolate_na`, `rolling_exp` stay in xarray (and coordax); a pipeline crosses with `DimArray.to_xarray` or an xarray carrier.
Alternative: label-value `sel` ranges; rejected, because it is the step that turns this layer into a second xarray.

**A21. The QC `Diff` becomes `AssertClose`.**
Recommended: `Diff` is the operator twin of `diff`, as `Reduce` is of `reduce`; the planned QC operator in [qc.md](qc.md) is renamed `AssertClose`, matching its `Assert*` siblings.
Alternative: name the twin `Difference`; rejected, because every other twin is the function's name in CamelCase.

**A22. One optional adapter method, `relabel`, for labels `rewrap` cannot carry.**
Recommended: an optional `relabel(x, labels)` in a runtime-checkable `LabelledAdapter` protocol (section 3.0); the seven required methods do not change.
`rename`, `merge_dims`, `split_dims`, `quantile`, `broadcast_like` and `dot` call it after `rewrap`; without it they raise `TypeError` when labels would be set, instead of dropping them.
`DimArrayAdapter` and the xarray adapter implement it; a third-party adapter that skips it keeps working for every other function.
Alternative: a `labels=` keyword on `rewrap`; rejected, because it changes a required method's signature and breaks every adapter written against the current protocol, while an optional method is additive.
Alternative: a required fast path for each of the six functions on every labelled carrier; rejected, because it repeats the label logic once per adapter, and the reference `DimArray` would need it too.

## 7. Testing gates

Fixtures: the test kit's layout ([`testing.py`](../../../src/pipekit_array/testing.py)), `time band y x` of shape (3, 3, 4, 5) with band labels, on NumPy, JAX and dask; and the running example at (12, 13, 256, 256), float32.

| Gate | Pass condition |
| --- | --- |
| G1 xarray parity | For every proposed function, `values_of(f(dimarray))` equals the xarray call on `dimarray.to_xarray()` (section 5), exactly for index-only functions and NaN in the same places (`assert_array_equal`), within `rtol=1e-12` for `quantile`, `coarsen`, `cumreduce` and `dot` on float64. Example: `shift(x, time=1)` equals `xr.DataArray.shift(time=1)`, NaN fills included. |
| G2 window parity | `window(x, time=3, stride=2)` equals `x.rolling(time=3).construct("window").isel(time=slice(2, None, 2))` exactly, labels included; on the running example it has $\lfloor (12 - 3)/2 \rfloor + 1 = 5$ windows. |
| G3 adapter agreement | `check_adapter` gains one case per function; the xarray adapter (NumPy and dask data) agrees with `DimArrayAdapter` exactly for index-only functions, within `rtol=1e-12` otherwise. |
| G4 jit | `jax.jit(f)` with the static arguments of section 3 gives results identical to the eager call (`assert_array_equal`; `rtol=1e-6` for float32 reductions), and the same dims and labels. |
| G5 vmap | `jax.vmap(f)` over a stacked batch of 4 cubes equals `stack([f(c) for c in cubes], "batch")` exactly. |
| G6 round trips | `split_dims(merge_dims(x, pixel=("y", "x")), pixel={"y": Y, "x": X})` equals `x` exactly, labels included; `rename` with inverted names is the identity; `squeeze(expand_dims(x, "member"), "member")` is the identity; `roll(roll(x, time=k), time=-k)` is the identity for $k \in \{1, 5, 12\}$. |
| G7 closed forms | `isel(shift(x, time=k), time=slice(k, None))` equals `isel(x, time=slice(None, -k))`; the last scene of `cumreduce(diff(x, "time"), "cumulative_sum", "time")` equals `x[T-1] - x[0]` within `rtol=1e-12`; on the running example, `coarsen(x, "mean", time=3)` equals `reduce(split_dims(x, time={"time": 4, "window": 3}), "mean", "window")` exactly. |
| G8 dtype | float32 in, float32 out for every function; `shift` and `pad` with `mode="constant"` on integer data and no fill value raise `TypeError`. |
| G9 operators | Every twin with only names and numbers round-trips through `dumps` / `loads` with a stable content hash; `compute_output_signature` equals `signature_of(op(x))` on the running example. |
| G10 backends | Each function runs on NumPy, JAX and dask; `cumreduce(x, "cumulative_sum", "time")` on dask equals NumPy exactly. |
| G11 labels | On `DimArray` and on the xarray adapter: `labels(rename(cube, band="channel"), "channel")` equals the 13 band names; `labels(merge_dims(cube, pb=("band", "y")), "pb")[0] == ("B1", 0)`; `labels(quantile(cube, [0.1, 0.9], "time"), "quantile") == (0.1, 0.9)`; `dot` keeps the `component` labels of its second input. An adapter without `relabel` raises `TypeError` from `rename` on a labelled dim, and returns the same data as `DimArrayAdapter` on an unlabelled one. |
| G12 batched dot | For `x` with dims `(time, band)` of shape (12, 13) and `w` with dims `(time, band, component)` of shape (12, 13, 3), `dot(x, w, dim="band")` has dims `(time, component)` and equals `xr.dot(x, w, dim="band")` and the loop $s_{tc} = \sum_b x_{tb} w_{tbc}$ within `rtol=1e-12` on float64; on NumPy, JAX and dask. |

## 8. Phased order

| Phase | Adds | Risk | Fallback |
| --- | --- | --- | --- |
| 1. Dims | `LabelledAdapter.relabel` on `DimArrayAdapter` and the xarray adapter (A22); `rename`, `expand_dims`, `squeeze`, `broadcast_like`, `merge_dims`, `split_dims` and twins; A19 and A21 renames in the catalogue | `rename` on xarray and geoarray must move more metadata keyed by dim name than labels (non-index coords, the grid), which `relabel` does not cover. | Make the `rename` fast path required for those adapters; `check_adapter` fails without it. |
| 2. Index-only transforms | `pad`, `roll`, `shift`, `diff`, `cumreduce`, `where` and twins | Backends name things differently: dask has `cumsum`, not `cumulative_sum`; `pad` is resolved per backend, not from the standard. | Extend `_DaskNamespace._NAMES`; restrict `pad` to the four modes all three backends share. |
| 3. Windows | `window`, `coarsen`, `ApplyAlong`, `BatchAlong` | `window` copies the data $S$ times in the generic path (a 12 × 13 × 256 × 256 float32 cube is 41 MB; $S = 3$ makes 123 MB). | The xarray fast path (a view); for large $S$, `coarsen` or `KernelOp` with a convolution kernel. |
| 4. Statistics and algebra | `quantile`, `dot` and twins | JAX supports six quantile methods; dask needs the quantile dim in one chunk. | Reject other methods with a `ValueError` naming the six; rechunk as `take_along` does today. |

Each phase is one PR with its tests, gates G1 to G12 for its functions, and the `api/README.md` table updated.
