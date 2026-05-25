---
status: draft
version: 0.1.0
---

# Vision

## 1. Motivation

Real scientific pipelines run on a mix of backends. The same NDVI
computation might run on numpy in a notebook, on JAX during model
training, on CuPy in a GPU-attached inference service, and on dask for
a cluster batch job. Writing four versions of the same operator —
one per backend — is the wrong path. Writing one version against a
specific backend (say numpy) and then `.cpu().numpy()`-converting
everywhere else is slower, error-prone, and breaks gradient flow.

The [Python Array API standard][array-api] is the modern, formal
answer: every conforming array implements `__array_namespace__()`,
which returns a module-like object with `mean`, `sum`, `reshape`,
`where`, … exposed as functions. Five major libraries — numpy 2.0+,
JAX, CuPy, PyTorch 2.0+, dask — all conform (fully or substantially).
`pipekit-array` is the package that **wraps the ~12 array-shaped
operators we keep rewriting** (`ApplyToBands`, `Subsample`, `Diff`,
`AssertValueRange`, `ModelOp`, …) against `array_namespace(x)` so the
operator picks its backend from the input.

[array-api]: https://data-apis.org/array-api/

What's *missing* from pipekit core is a place for these operators to
live. `pipekit.qc` deliberately stays carrier-agnostic — its
docstring is explicit that numeric checks (`AssertValueRange`,
`AssertNoNaN`) "live in sister libraries (Report 3,
`pipekit-array`)". `pipekit.observe.Histogram` ships a generic
controller with a user-supplied `to_array` adapter; `pipekit-array`
ships the carrier-aware version where `to_array` defaults to the
Array API's flatten path. The boundary is already drawn; this package
is the thing on the other side.

## 2. What pipekit-array is

A thin, dependency-light library: **pipekit-array depends on pipekit
and nothing else by default.** Backends are extras-gated
(`pipekit-array[numpy]`, `[jax]`, `[torch]`, `[cupy]`, `[dask]`).
Backend availability is determined at *call* time from the input:
`array_namespace(x)` raises `TypeError` if `x` doesn't implement
`__array_namespace__` (i.e. no Array-API-conforming backend is
installed for the input's type). Operators themselves construct
without needing any backend — that's the whole point of dispatching
on the input.

Three conceptual layers, all on top of `pipekit.Operator`:

- **Data-flow operators** — `ApplyToBands`, `StackAlong`,
  `ConcatenateAlong`, `Subsample`, `MeanScalar`, `BatchedMap`
  (array-shaped), `Histogram` (carrier-aware controller). The
  combinators and reducers that show up in every array pipeline.
- **Numeric QC** — `Diff`, `AssertValueRange`, `AssertNoNaN`,
  `AssertValidFraction`. The numeric-content checks `pipekit.qc`
  refused to write because they need to look inside the array.
- **Inference** — `ModelOp(model, method="__call__", batch_size=None)`.
  Framework-agnostic inference; the trained-model wrapper that the
  Lightning / Keras backends in `pipekit-train` will return from
  `loop.run()` (the Equinox backend already ships its own
  `EquinoxModelOp` / `pipekit_jax.JaxModelOp` — see ADR A8).

Every operator follows the same shape: `_apply(x, …)` calls
`array_namespace(x)` to pick the backend, calls the appropriate Array
API functions, returns an array of the same backend. No conversion,
no `.cpu()`, no `.numpy()`.

## 3. Design principles

These echo `pipekit-train`'s principles and extend them to the
array-carrier setting:

- **One operator, all backends.** The user installs *one* extra
  (`[numpy]`, `[jax]`, …), passes an array of that kind, and the
  operator works. Mixed-backend pipelines are supported but rare —
  most users pick one backend per pipeline.
- **Pure functions.** Every operator is side-effect-free; JAX
  immutability is the strict constraint that drives the design. No
  in-place mutation, ever.
- **Carrier-aware controllers over carrier-agnostic ones.** Where
  `pipekit.observe.Histogram` takes a `to_array` callable for
  flexibility, `pipekit-array.Histogram` defaults `to_array` to
  `xp.reshape(x, (-1,))` — the right thing for arrays. The carrier
  layer can make assumptions the core can't.
- **No backend-specific tricks.** No `jax.jit` integration, no
  `torch.compile` hooks, no CUDA-specific paths. Those belong in
  `pipekit-jax`, `pipekit-torch`, etc., or in the user's outer code.
- **Sister libraries re-export.** `geotoolz.qc.AssertValueRange`
  subclasses `pipekit_array.qc.AssertValueRange` and adds
  GeoTensor-aware defaults (e.g. reading `fill_value` as the implicit
  min/max). Same for `xr_toolz`. One implementation; three places
  it's used.

## 4. Where pipekit-array sits in the stack

```
┌────────────────────────────────────────────────────┐
│                Domain libraries                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │  geotoolz    │  │   xr_toolz   │  │  others  │  │
│  │ (GeoTensor)  │  │   (xarray)   │  │   ...    │  │
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  │
│         └────────┬────────┴───────────────┘        │
│                  ▼                                 │
│         ┌─────────────────┐                        │
│         │  pipekit-array  │  ← duck-typed arrays   │
│         │  (Array API)    │     (numpy/JAX/torch…) │
│         └────────┬────────┘                        │
│                  ▼                                 │
│         ┌─────────────────┐                        │
│         │     pipekit     │  ← carrier-agnostic    │
│         │     (core)      │     framework          │
│         └─────────────────┘                        │
└────────────────────────────────────────────────────┘
```

- **Below**: `pipekit` core. No internal deps; no third-party
  imports. pipekit-array adds no new framework primitives — its
  operators are vanilla `pipekit.Operator` subclasses.
- **Above**: domain libraries (`geotoolz`, `xr_toolz`, future
  `xr_toolz`-style remote-sensing or ocean libraries). They re-export
  pipekit-array operators with carrier-specific defaults; they don't
  reimplement them.
- **Sideways**: `pipekit-jax` (separate workspace package) wraps
  Equinox modules with weight-blob round-trip through
  `pipekit-experiment.ModelRegistry`. `pipekit-array.ModelOp` is for
  the *general* case (numpy classifiers, sklearn models, raw torch /
  Keras modules); `pipekit-jax.JaxModelOp` is for the Equinox-specific
  case where the weights need to round-trip through a `ModelRegistry`.
  See ADR A8 for the boundary.

## 5. The Array API bet — what works, what doesn't

The [Array API standard][array-api] is well-defined for the operations
pipekit-array needs (basic math, reductions, reshapes, indexing,
concatenation). It is **not** well-defined for:

- **FFTs, advanced linear algebra.** Not in the spec; needs a
  per-backend fallback via `array_namespace(x).__name__` lookup. None
  of the v0.1 operators need this, but it's the first place we'll
  feel the gap as we add operators.
- **NaN semantics.** numpy and JAX differ on edge cases (e.g.
  `nan_to_num` behaviour, `nanmean` on all-NaN slices). Operators
  that traffic in NaN (`AssertValidFraction`, future `nanmean`-style
  ops) need per-backend tests. See ADR A6.
- **dask laziness.** dask conforms to the Array API but its
  operations are lazy; the user must call `.compute()` at the end.
  v0.1 supports dask arrays but doesn't auto-compute. See ADR A7.
- **Mutability.** numpy and PyTorch tensors are mutable; JAX and
  CuPy are pragmatically immutable in user code. pipekit-array is
  strictly pure — no operator mutates its input. See ADR A2.

These constraints are *known*, *documented*, and *small enough* that
they don't motivate per-backend libraries. One library covers all
five backends with disciplined operator design.

## 6. Out of scope

These are explicitly considered and rejected for the v0.1 scope.

- **`jax.jit` / `torch.compile` traceability.** pipekit operators do
  dual-mode dispatch (`__call__` vs `_apply`); they don't trace
  cleanly. Live with it; if a user needs JAX-traceable pipelines they
  reach for `pipekit-jax` (a separate package).
- **A new framework primitive.** pipekit-array adds zero new base
  classes. Every operator is `class Foo(pipekit.Operator)` with an
  `_apply` method.
- **DataFrame / pandas operators.** Not array-shaped; doesn't fit
  the Array API. Defer indefinitely.
- **GPU memory management.** `[cupy]` and `[torch-gpu]` users own
  their device placement; pipekit-array doesn't move arrays between
  devices.
- **Backend conversion.** No `to_numpy()`, no auto-`device=`. If you
  want a numpy result, hand the operator a numpy input.
- **Dataset loaders, batching for training.** That's
  `pipekit-train`'s `TrainingDataset`. pipekit-array's `BatchedMap`
  is for splitting one large array along axis 0 for inference, *not*
  for serving training batches.

Boundary cases are catalogued in [boundaries.md](boundaries.md).
