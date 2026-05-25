---
status: draft
version: 0.1.0
---

# Boundaries

## 1. Open questions & future work

### Q1. FFT and advanced linear algebra

The Array API standard's `linalg` / `fft` extensions are spec'd but
inconsistently implemented across backends. None of the v0.1
operators need them, so they're not in scope for v0.1. When a user
asks for `Spectrogram` / `WaveletDecompose` / `MatchedFilter`-style
operators (the natural pull), the right answer is probably a
`pipekit-array[fft]` sub-extra that pins
[`array-api-compat[fft]`][api-compat-fft] and adds the
extension-using operators in their own module
(`pipekit_array.spectral`). Logged as v0.2 work, not blocking.

[api-compat-fft]: https://data-apis.org/array-api-compat/

### Q2. `pipekit_array.ModelOp` and torchscript / Keras SavedModel

`ModelOp` v0.1 holds the model object directly and calls it. It
doesn't serialise the model. The natural next step — "serialise the
model alongside the operator" — splits along backend:

- **PyTorch:** `torch.jit.script` / `torch.save(state_dict)`
- **Keras:** `model.save("path.keras")`
- **sklearn:** `joblib.dump`

A v0.2 `ModelOp.serialise(format=...)` would orchestrate these; v0.1
expects the user to pickle the whole operator (works for
sklearn / numpy models) or supply the serialised model out-of-band.

### Q3. Sharding for `BatchedMap`

`pipekit_array.parallel.BatchedMap` v0.1 is a single-device sequential
loop: split axis 0, apply, concatenate. The natural v0.2 extension
is `BatchedMap(..., devices=jax.devices())` that uses
`jax.experimental.shard_map` for JAX inputs and `torch.distributed`
for torch. Single-backend (no Array API path), so per-backend dispatch
is needed — likely in a `pipekit-jax` / `pipekit-torch` followup,
not in pipekit-array core. See ADR A4.

### Q4. Streaming `Histogram` for online statistics

The v0.1 `Histogram` controller holds all captured arrays in memory
and computes histogram at `report()` time. For very large arrays a
streaming implementation (`hist` field updated on every call via
`xp.histogram`) is the v0.2 polish. The API doesn't change; just the
storage shape.

### Q5. Auto-detect `compute()` boundary for dask

Per ADR A7, dask laziness is the user's responsibility. If user
demand grows, a `pipekit_array.dask.AutoCompute()` operator (a
no-op for non-dask backends, a `.compute()` for dask) is the safe
escape hatch — opt-in, doesn't break the laziness contract for users
who want it.

### Q6. Mixed-backend pipelines

ADR A1 + the `array_namespace(*xs)` discipline (architecture R2)
makes mixed-backend pipelines fail loudly at the first multi-input
operator. The alternative — auto-convert at boundaries — was rejected.
If a real use case demands it (rare, but e.g. "compute on JAX, log
to numpy") we'd add a `pipekit_array.convert.To(backend)` operator —
single-purpose, explicit, opt-in. Logged as a v0.2 issue if asked for.

### Q7. `Signature` integration depth

The v0.1 plan: operators that reshape (`ApplyToBands`, `Subsample`,
`BatchedMap`) override `compute_output_signature`; operators that
pass through inherit. The deeper integration — Array API operators
participating in `Sequential.summary`'s shape inference graph and
catching shape mismatches at compose time — is a v0.2 polish.
Tracked via a single follow-up issue once v0.1 lands.

## 2. Non-goals

These were explicitly considered and rejected for the v0.1 scope.

- **A new framework primitive.** pipekit-array adds zero base
  classes; every operator subclasses `pipekit.Operator`. The library
  is a bag of operators.
- **Backend conversion.** No `to_numpy()`, no `device=` argument, no
  auto-`x.cpu().numpy()`. Conversion lives in user code or in a
  future opt-in operator (Q6).
- **JAX traceability.** pipekit operators dual-mode dispatch on
  `__call__` vs `_apply` and aren't `jax.jit`-compatible. Live with
  it; `pipekit-jax` is the answer for differentiable / traceable
  pipelines.
- **`torch.compile` / `tf.function` hooks.** Same reasoning as JAX —
  framework-specific tracing belongs in per-backend sister packages.
- **DataFrame / pandas / polars operators.** Not array-shaped; not
  in the Array API. Deferred indefinitely.
- **Distributed scheduling.** dask graphs work because the user
  scheduled them; pipekit-array doesn't run a distributed scheduler.
  See Q3.
- **GPU memory management.** `[cupy]` and `[torch-gpu]` users own
  their device placement; pipekit-array doesn't `.to(device)` for
  them.
- **An `nan`-tolerant arithmetic suite.** `nanmean`, `nansum`, etc.
  are *partially* in the Array API but the edge cases (all-NaN
  slices) differ between backends. If a user needs them, they get a
  per-backend operator in user code or a future
  `pipekit_array.numeric_nan` module — out of v0.1 scope.

## 3. v0.1 Definition of Done

The v0.1 implementation PR(s) close on:

- [ ] All 12 operators implemented and tested per the api/ files:
  - [ ] `combinators.py`: `ApplyToBands`, `StackAlong`, `ConcatenateAlong`
  - [ ] `geom.py`: `Subsample`
  - [ ] `observe.py`: `Histogram` (controller)
  - [ ] `reduce.py`: `MeanScalar`
  - [ ] `parallel.py`: `BatchedMap`
  - [ ] `qc.py`: `Diff`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction`
  - [ ] `inference.py`: `ModelOp`
- [ ] `_namespace.array_namespace()` shim implemented and tested.
- [ ] Per-backend CI fan-out for `numpy`, `jax`, `torch` (via the
  conftest `_make_array(backend, ...)` fixture).
- [ ] `cupy` and `dask` exercised by opt-in `make test-cupy` /
  `make test-dask` targets — not in the core CI matrix.
- [ ] All YAML-eligible operators round-trip (`dumps(op) → loads`);
  content hash stable across runs. `ModelOp` and the per-tap
  operators returned by `Histogram.at(key)` are flagged with
  `forbid_in_yaml = True` and are excluded from this check — see
  `architecture.md` §5 for the rationale.
- [ ] Workspace pre-commit gates pass with `[numpy]` installed
  (the default-CI shape).
- [ ] API docs published at `docs/api/pipekit-array.md` next to the
  existing package docs.
- [ ] One quickstart notebook under `docs/notebooks/`
  (`pipekit_array_quickstart.ipynb`) showing the dispatch pattern
  on numpy, JAX, and torch inputs through the same operator.

## 4. v0.1 implementation phasing

Suggested chunking for the implementation PR(s):

1. **Phase A — Foundations.** `_namespace.py` + conftest fixture +
   `MeanScalar` + `StackAlong` + `ConcatenateAlong`. ~150 LOC, sets
   the per-backend test pattern. One PR.
2. **Phase B — Catalog body.** `Subsample`, `ApplyToBands`,
   `BatchedMap`, `Histogram`. ~200 LOC. One PR.
3. **Phase C — QC suite.** `Diff`, `AssertValueRange`, `AssertNoNaN`,
   `AssertValidFraction`. ~100 LOC. One PR (delegates check logic to
   shared helpers).
4. **Phase D — `ModelOp` + docs + notebook + DoD close.** ~150 LOC +
   docs. One PR; closes v0.1.

Total estimate: 4 PRs, ~600 LOC of source + ~400 LOC of tests + docs.
The numbers track Report 3's "12 operators, ~400 LOC" estimate plus
the test surface fan-out across backends.

## 5. Versioning targets

- **v0.1** — All 12 operators, numpy / JAX / torch CI fan-out, basic
  signature inference for reshaping operators (this design).
- **v0.2** — Sharding for `BatchedMap` (Q3) — likely lands in a
  per-backend sister package; spectral extension operators (Q1);
  streaming `Histogram` (Q4); deeper signature integration (Q7).
- **v0.3** — Whatever the user feedback drives. Likely candidates:
  `ModelOp.serialise()` (Q2), `AutoCompute` for dask (Q5).

The cadence matches the rest of the pipekit workspace: roughly one
minor bump per quarter, gated on the v0.1 scope landing first.
