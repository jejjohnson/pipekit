---
status: draft
version: 0.1.0
---

# Boundaries

## 1. Open questions & future work

### Q1. FFT and advanced linear algebra

The Array API standard's `linalg` and `fft` extensions are specified but inconsistently implemented across backends.
None of the v0.1 operators needs them, so they are not in scope for v0.1.
When a user asks for `Spectrogram`, `WaveletDecompose` or `MatchedFilter`-style operators (the natural pull), the right answer is probably a `pipekit-array[fft]` sub-extra that pins [`array-api-compat[fft]`][api-compat-fft] and adds the extension-using operators in their own module (`pipekit_array.spectral`).
This is logged as v0.2 work and does not block anything.

[api-compat-fft]: https://data-apis.org/array-api-compat/

### Q2. `pipekit_array.ModelOp` and torchscript / Keras SavedModel

`ModelOp` is not built yet.
The v0.1 design holds the model object directly and calls it, without serialising the model.
The natural next step ("serialise the model alongside the operator") splits along backend:

- **PyTorch:** `torch.jit.script` / `torch.save(state_dict)`;
- **Keras:** `model.save("path.keras")`;
- **sklearn:** `joblib.dump`.

A v0.2 `ModelOp.serialise(format=...)` would orchestrate these.
v0.1 expects the user to pickle the whole operator (which works for sklearn and NumPy models) or to supply the serialised model out of band.

### Q3. Sharding for `BatchAlong`

`pipekit_array.BatchAlong` (shipped in #206) is a single-device sequential loop: split a named dim, apply, concatenate.
The natural v0.2 extension is `BatchAlong(..., devices=jax.devices())`, using `jax.experimental.shard_map` for JAX inputs and `torch.distributed` for torch.
It is single-backend (no Array API path), so it needs per-backend dispatch, which likely belongs in a `pipekit-jax` / `pipekit-torch` follow-up and not in pipekit-array core (ADR A4).

### Q4. Streaming `Histogram` for online statistics

The v0.1 `Histogram` controller (not built yet) holds all captured arrays in memory and computes the histogram at `report()` time.
For very large arrays, a streaming implementation (a `hist` field updated on every call via `xp.histogram`) is the v0.2 polish.
The API does not change; only the storage shape does.

### Q5. Auto-detect `compute()` boundary for dask

Per ADR A7, dask laziness is the user's responsibility.
If user demand grows, a `pipekit_array.dask.AutoCompute()` operator (a no-op for non-dask backends, a `.compute()` for dask) is the safe escape hatch.
It is opt-in, so it does not break the laziness contract for users who want it.

### Q6. Mixed-backend pipelines

ADR A1 and the `array_namespace(*xs)` discipline (architecture R2) make mixed-backend pipelines fail loudly at the first multi-input operator.
The alternative, converting automatically at boundaries, was rejected.
If a real use case demands it (rare, but for example "compute on JAX, log to NumPy"), we would add a `pipekit_array.convert.To(backend)` operator: single-purpose, explicit and opt-in.
This is logged as a v0.2 issue if anyone asks for it.

**As built.** The named layer follows the same rule: multi-input dim functions (`dot`, `combine`, `concat`, …) raise `BackendError` when their inputs sit on more than one backend (`require_one_backend` in `dims.py`).

### Q7. `Signature` integration depth

The v0.1 plan: operators that reshape (`ApplyAlong`, `Subsample`, `BatchAlong`) override `compute_output_signature`, and operators that pass through inherit.
The deeper integration (Array API operators taking part in `Sequential.summary`'s shape inference and catching shape mismatches at compose time) is a v0.2 polish, tracked by a single follow-up issue once v0.1 lands.

**As built.** `ApplyAlong`, `BatchAlong`, `MeanScalar` and every dim-function twin implement `compute_output_signature`, and gate G9 checks it against `signature_of(op(x))` on the running example of [api/dim_functions.md](api/dim_functions.md).

## 2. Non-goals

These were explicitly considered and rejected for the v0.1 scope.

- **A new framework primitive.** pipekit-array adds zero base classes; every operator subclasses `pipekit_ops.Operator`.
  The library is a bag of operators.
- **Backend conversion.** No `to_numpy()`, no `device=` argument, no automatic `x.cpu().numpy()`.
  Conversion lives in user code or in a future opt-in operator (Q6).
- **JAX traceability of operators.** pipekit operators dispatch in two modes (`__call__` vs `_apply`) and are not `jax.jit`-compatible.
  We live with it; `pipekit-jax` is the answer for differentiable or traceable pipelines.
  The dim-level functions trace under `jit` and `vmap` because they are plain functions on a pytree (gates G4 and G5).
- **`torch.compile` / `tf.function` hooks.** The same reasoning as JAX: framework-specific tracing belongs in per-backend sister packages.
- **DataFrame / pandas / polars operators.** They are not array-shaped and not in the Array API, so they are deferred indefinitely.
- **Distributed scheduling.** dask graphs work because the user scheduled them; pipekit-array does not run a distributed scheduler (Q3).
- **GPU memory management.** CuPy and GPU PyTorch users own their device placement; pipekit-array does not call `.to(device)` for them.
- **A NaN-tolerant arithmetic suite.** `nanmean`, `nansum` and the like are *partially* in the Array API, but the edge cases (all-NaN slices) differ between backends.
  A user who needs them writes a per-backend operator in user code, or waits for a future `pipekit_array.numeric_nan` module; it is out of v0.1 scope.
  As built, `reduce` resolves any namespace function by name, so `reduce(x, "nanmean", "time")` works wherever the namespace has `nanmean`.

## 3. v0.1 Definition of Done

The v0.1 implementation PRs close on the list below; ticks mark what pipekit-array 0.1.1 has.

- [ ] All 12 operators implemented and tested per the `api/` files:
  - [x] `ApplyAlong` (in `dim_operators.py`, not `combinators.py`), `StackAlong`, `ConcatenateAlong` (in `combinators.py`);
  - [x] `Subsample`: retired, it is `isel` with a stepped slice (A15);
  - [ ] `observe.py`: `Histogram` (controller);
  - [x] `MeanScalar`, in `reductions.py` (the module was `reduce.py` before 0.0.3);
  - [x] `BatchAlong`, in `batching.py` (not `parallel.py`);
  - [ ] `qc.py`: `AssertClose`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction`;
  - [ ] `inference.py`: `ModelOp`.
- [x] `_namespace.array_namespace()` shim implemented and tested.
- [ ] Per-backend CI fan-out for `numpy`, `jax`, `torch` via the conftest `_make_array(backend, ...)` fixture.
  The fixture exists, but the workspace dev group installs neither `torch` nor `array-api-compat`, so the torch cases skip.
- [ ] `cupy` and `dask` exercised by opt-in `make test-cupy` / `make test-dask` targets.
  Neither target exists; dask is in the dev group instead, so the dim-function and xarray tests run on dask in the main suite, and cupy is not exercised.
- [x] All YAML-eligible operators round-trip (`dumps(op)`, then `loads`), with a content hash stable across runs (`test_gate_a1_round_trip_array.py`).
  `ModelOp` and the per-tap operators returned by `Histogram.at(key)` would carry `forbid_in_yaml = True` and are excluded from this check (`architecture.md` §5).
- [x] Workspace pre-commit gates pass with `[numpy]` installed (the default CI shape).
- [x] API docs published at `docs/api/pipekit-array.md` next to the existing package docs.
- [ ] One quickstart notebook under `docs/notebooks/` (`pipekit_array_quickstart.ipynb`) showing the dispatch pattern on NumPy, JAX and torch inputs through the same operator.

## 4. v0.1 implementation phasing

Suggested chunking for the implementation PRs:

1. **Phase A: Foundations.** `_namespace.py`, the conftest fixture, `MeanScalar`, `StackAlong`, `ConcatenateAlong`.
   About 150 LOC; it sets the per-backend test pattern.
   One PR. **Done.**
2. **Phase B: Catalogue body.** `Subsample`, `ApplyAlong`, `BatchAlong`, `Histogram`.
   About 200 LOC, one PR.
   **Partly done:** `ApplyAlong` and `BatchAlong` shipped with phase 3 of the dim functions (#206), and `Subsample` was retired; `Histogram` is not built.
3. **Phase C: QC suite.** `AssertClose`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction`.
   About 100 LOC, one PR (it delegates the check logic to shared helpers).
   **Not started.**
4. **Phase D: `ModelOp`, docs, notebook and DoD close.** About 150 LOC plus docs, one PR, which closes v0.1.
   **Not started.**

The total estimate is 4 PRs, about 600 LOC of source, about 400 LOC of tests, and docs.
The numbers track Report 3's "12 operators, ~400 LOC" estimate plus the test fan-out across backends.

The dim-level functions were not in this phasing; they shipped in their own four phases (#204 to #207), described in [api/dim_functions.md](api/dim_functions.md) §8.

## 5. Versioning targets

- **v0.1:** all 12 operators, NumPy / JAX / torch CI fan-out, basic signature inference for reshaping operators (this design).
- **v0.2:** sharding for `BatchAlong` (Q3), likely in a per-backend sister package; spectral extension operators (Q1); streaming `Histogram` (Q4); deeper signature integration (Q7).
- **v0.3:** whatever the user feedback drives; likely candidates are `ModelOp.serialise()` (Q2) and `AutoCompute` for dask (Q5).

The cadence matches the rest of the pipekit workspace: roughly one minor bump per quarter, gated on the v0.1 scope landing first.
