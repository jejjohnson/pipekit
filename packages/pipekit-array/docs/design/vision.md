---
status: draft
version: 0.1.0
---

# Vision

## 1. Motivation

Real scientific pipelines run on a mix of backends.
The same NDVI computation might run on NumPy in a notebook, on JAX during model training, on CuPy in a GPU-attached inference service, and on dask for a cluster batch job.
Writing four versions of the same operator (one per backend) is the wrong path.
Writing one version against a specific backend (say NumPy) and then calling `.cpu().numpy()` everywhere else is slower, error-prone, and breaks gradient flow.

The [Python Array API standard][array-api] is the modern, formal answer.
Every conforming array implements `__array_namespace__()`, which returns a module-like object with `mean`, `sum`, `reshape`, `where`, … exposed as functions.
Five major libraries (NumPy 2.0+, JAX, CuPy, PyTorch 2.0+ and dask) conform, fully or substantially.
`pipekit-array` is the package that **wraps the ~12 array-shaped operators we keep rewriting** (`ApplyAlong`, `Subsample`, `AssertClose`, `AssertValueRange`, `ModelOp`, …) against `array_namespace(x)`, so the operator picks its backend from the input.

[array-api]: https://data-apis.org/array-api/

What is *missing* from pipekit-ops is a place for these operators to live.
`pipekit_ops.qc` deliberately stays carrier-agnostic; its docstring says that numeric checks (`AssertValueRange`, `AssertNoNaN`) "live in sister libraries (Report 3, `pipekit-array`)".
`pipekit_ops.observe.Histogram` ships a generic controller with a user-supplied `to_array` adapter, and `pipekit-array` would ship the carrier-aware version, where `to_array` defaults to the Array API's flatten path.
The boundary is already drawn, and this package is the thing on the other side.

**As built (0.1.1).** The data-flow operators (`ApplyAlong`, `StackAlong`, `ConcatenateAlong`, `MeanScalar`, `BatchAlong`) ship, and `Subsample` was retired in favour of `isel` with a stepped slice (ADR A15).
The QC family, the array `Histogram` and `ModelOp` are not built yet.
The package grew a second surface the original plan did not have: the named carrier `DimArray` and the dim-level functions of [api/dim_functions.md](api/dim_functions.md), which work by dim name on any registered carrier.

## 2. What pipekit-array is

A thin, dependency-light library: **pipekit-array depends on `pipekit-ops` and nothing else by default.**
Backends are extras (`pipekit-array[numpy]`, `[jax]`, `[torch]`, `[cupy]`, `[dask]`), and so are the integrations (`[xarray]`, `[sklearn]`, `[einx]`, `[compat]`).
Backend availability is determined at *call* time from the input: `array_namespace(x)` raises `TypeError` if `x` does not implement `__array_namespace__` (no Array-API-conforming backend is installed for the input's type).
Operators themselves construct without needing any backend, which is the whole point of dispatching on the input.

Three conceptual layers, all on top of `pipekit_ops.Operator`:

- **Data-flow operators:** `ApplyAlong`, `StackAlong`, `ConcatenateAlong`, `Subsample`, `MeanScalar`, `BatchAlong` (carrier-shaped) and `Histogram` (a carrier-aware controller).
  These are the combinators and reducers that show up in every array pipeline.
- **Numeric QC:** `AssertClose`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction`.
  These are the numeric-content checks `pipekit_ops.qc` refused to write, because they need to look inside the array.
- **Inference:** `ModelOp(model, method="__call__", batch_size=None)`, a framework-agnostic trained-model wrapper.
  It is what the planned Lightning and Keras backends in `pipekit-train` would return from `loop.run()`; the Equinox backend already ships its own `EquinoxModelOp`, and `pipekit_jax.JaxModelOp` round-trips weights (ADR A8).

Every operator follows the same shape: `_apply(x, …)` calls `array_namespace(x)` to pick the backend, calls the Array API functions, and returns an array of the same backend.
There is no conversion, no `.cpu()` and no `.numpy()`.

## 3. Design principles

These echo `pipekit-train`'s principles and extend them to the array-carrier setting:

- **One operator, all backends.** The user installs *one* extra (`[numpy]`, `[jax]`, …), passes an array of that kind, and the operator works.
  Mixed-backend pipelines are supported but rare; most users pick one backend per pipeline.
- **Pure functions.** Every operator is side-effect-free.
  JAX immutability is the strict constraint that drives the design: no in-place mutation, ever.
- **Carrier-aware controllers over carrier-agnostic ones.** Where `pipekit_ops.observe.Histogram` takes a `to_array` callable for flexibility, the planned `pipekit_array` `Histogram` defaults `to_array` to `xp.reshape(x, (-1,))`, the right thing for arrays.
  The carrier layer can make assumptions the core cannot.
- **No backend-specific tricks.** No `torch.compile` hooks and no CUDA-specific paths.
  Those belong in `pipekit-jax`, `pipekit-torch` and so on, or in the user's outer code.
  As built, JAX is the one backend the named layer is written to trace through: `DimArray` is a JAX pytree, and the dim-level functions pass `jax.jit` and `jax.vmap` (gates G4 and G5 of [api/dim_functions.md](api/dim_functions.md) §7).
- **Sister libraries re-export.** The plan is that `geoops.qc.AssertValueRange` subclasses `pipekit_array.qc.AssertValueRange` and adds GeoTensor-aware defaults (for example, reading `fill_value` as the implicit min or max).
  The same holds for `xr_toolz`: one implementation, used in three places.

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
│         │   pipekit-ops   │  ← carrier-agnostic    │
│         │     (core)      │     framework          │
│         └─────────────────┘                        │
└────────────────────────────────────────────────────┘
```

- **Below:** `pipekit-ops` core, with no internal deps and no third-party imports.
  pipekit-array adds no new framework primitives; its operators are plain `pipekit_ops.Operator` subclasses.
  The named-carrier registry (`register_adapter`, `NamedArrayAdapter`, `LabelledAdapter`) lives in core too, and pipekit-array registers `DimArrayAdapter` and the lazy xarray adapter with it.
- **Above:** domain libraries (`geotoolz`, `xr_toolz`, future `xr_toolz`-style remote-sensing or ocean libraries).
  They re-export pipekit-array operators with carrier-specific defaults; they do not reimplement them.
- **Sideways:** `pipekit-jax` (a separate workspace package) wraps Equinox modules with a weight-blob round trip through `pipekit-experiment.ModelRegistry`.
  The planned `pipekit_array` `ModelOp` is for the *general* case (NumPy classifiers, sklearn models, raw torch or Keras modules), and `pipekit_jax.JaxModelOp` is for the Equinox case, where the weights must round-trip through a `ModelRegistry` (ADR A8).

## 5. The Array API bet: what works, what doesn't

The [Array API standard][array-api] is well defined for the operations pipekit-array needs (basic maths, reductions, reshapes, indexing, concatenation).
It is **not** well defined for:

- **FFTs and advanced linear algebra.** These are extensions, inconsistently implemented, and need a per-backend fallback via `array_namespace(x).__name__`.
  None of the v0.1 operators needs them, but it is the first place we will feel the gap as we add operators.
- **`einsum`.** It is not in the standard either; `dot` uses it where the namespace has it (NumPy, JAX, dask) and falls back to broadcast, multiply and sum otherwise ([api/dim_functions.md](api/dim_functions.md) §3.4).
- **NaN semantics.** NumPy and JAX differ on edge cases (`nan_to_num` behaviour, `nanmean` on all-NaN slices).
  Operators that traffic in NaN (`AssertValidFraction`, `nanmean`-style reductions) need per-backend tests (ADR A6).
- **dask laziness.** dask conforms to most of the Array API, but its operations are lazy and the user must call `.compute()` at the end.
  v0.1 supports dask arrays but does not auto-compute (ADR A7).
- **Mutability.** NumPy and PyTorch tensors are mutable; JAX and CuPy are pragmatically immutable in user code.
  pipekit-array is strictly pure, and no operator mutates its input (ADR A2).

These constraints are *known*, *documented* and *small enough* that they do not motivate per-backend libraries.
One library covers all five backends with disciplined operator design.

## 6. Out of scope

These were explicitly considered and rejected for the v0.1 scope.

- **`jax.jit` / `torch.compile` traceability of operators.** pipekit operators do dual-mode dispatch (`__call__` vs `_apply`), and they do not trace cleanly.
  We live with it; a user who needs JAX-traceable pipelines reaches for `pipekit-jax` (a separate package).
  The dim-level *functions* are the exception: they are plain functions on a pytree and trace under `jax.jit` and `jax.vmap` (section 3).
- **A new framework primitive.** pipekit-array adds zero new base classes.
  Every operator is `class Foo(pipekit_ops.Operator)` with an `_apply` method.
- **DataFrame / pandas operators.** They are not array-shaped and do not fit the Array API, so they are deferred indefinitely.
- **GPU memory management.** CuPy and GPU PyTorch users own their device placement; pipekit-array does not move arrays between devices.
- **Backend conversion.** No `to_numpy()` and no automatic `device=`.
  If you want a NumPy result, hand the operator a NumPy input.
- **Dataset loaders and batching for training.** That is `pipekit-train`'s `TrainingDataset`.
  pipekit-array's `BatchAlong` splits one large carrier along a named dim for inference, *not* for serving training batches.

Boundary cases are catalogued in [boundaries.md](boundaries.md).
