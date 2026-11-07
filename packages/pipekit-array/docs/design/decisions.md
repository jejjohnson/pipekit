---
status: draft
version: 0.1.0
---

# Decisions (ADRs)

Format: each decision states the choice, the alternatives considered,
and the reason. Reviewers should treat these as the load-bearing
commitments of the design; everything else flows from them.

## A1. Dispatch via `array_namespace(x)`, not a backend registry

**Decision:** Operators call `array_namespace(x)` inside `_apply` and
use the returned namespace exclusively. No backend registry, no
`Backend` enum, no `backend=` constructor argument.

**Alternatives considered:**

- **Backend registry** — `Operator(..., backend="jax")` selects the
  implementation. Rejected: duplicates information the input array
  already carries, adds a configuration knob the user has to keep in
  sync.
- **Per-backend operator subclasses** — `NumpyMeanScalar`,
  `JaxMeanScalar`. Rejected: explodes the surface area, defeats the
  whole point of the Array API.

**Rationale:** `array_namespace(x)` is the standard's blessed
dispatch path. The input picks the backend; the operator is
backend-agnostic. This is the single most important decision; it
defines what pipekit-array *is*.

## A2. Strict purity — no in-place mutation

**Decision:** Every operator is side-effect-free. No
`x[i] = value`, no `x += other`, no `xp.copyto(dst, src)`. Operators
that need to "modify" return a fresh array via
`xp.where` / `xp.concatenate` / `xp.stack`.

**Alternatives considered:**

- **Per-backend mutation** — mutate when the backend supports it
  (numpy, torch), copy otherwise (JAX). Rejected: makes operator
  semantics backend-dependent, breaks R1 of the shape rules, defeats
  test reproducibility.
- **Optional in-place flag** — `Op(..., copy=False)`. Rejected:
  silently breaks under JAX, lures users into backend-coupling.

**Rationale:** JAX immutability is the strict constraint; designing
to it gets the others for free. The performance gap is
backend-internal — JAX's XLA compiler eliminates intermediate copies;
numpy users pay one extra allocation per operator, which is
negligible at the operator granularity pipekit-array works at.

## A3. Carrier-aware controllers ship in pipekit-array, not pipekit

**Decision:** `pipekit-array.observe.Histogram` ships its own
controller with `to_array=lambda x: xp.reshape(x, (-1,))` as the
default. `pipekit_ops.observe.Histogram` keeps its `to_array: Callable`
constructor argument for the generic case.

**Alternatives considered:**

- **Add Array API logic to `pipekit_ops.observe.Histogram`** — rejected:
  pipekit core has no third-party deps, by rule. The Array API import
  would couple core to a sister library's domain.
- **Make `pipekit-array.Histogram` a subclass.** Rejected: the two
  Histograms share the controller pattern but the array version has
  array-specific stat options (per-axis histograms, dtype-aware bin
  edges) that don't generalise to "arbitrary carrier flattened to a
  sequence of numbers".

**Rationale:** Same logic, different defaults. The carrier-aware
version makes the right choice for arrays without forcing the user
to spell it out; the core version stays generic for non-array
carriers (custom dataclasses, GeoTensor metadata stats, etc.).

## A4. No per-backend conditionals inside operators

**Decision:** If an operator can't be written against
`array_namespace(x)` alone, it doesn't live in pipekit-array. It
either:

- Lives in a per-backend sister package (`pipekit-jax`, future
  `pipekit-torch`).
- Lives in user code.
- Doesn't exist (drop the operator from the catalogue).

**Alternatives considered:**

- **Internal `if xp.__name__ == "jax.numpy"` branches.** Rejected:
  the backend coupling defeats the dispatch story; tests must run
  per-branch; bugs become per-backend bugs.

**Rationale:** Backend-specific tricks are the wrong layer's
problem. pipekit-array's value is the *generic* operator; if the
operator can't be generic, it doesn't belong here.

## A5. `pipekit-array.BatchedMap` is array-shaped, not a shadow

> Superseded by A19: the array-shaped operator is `BatchAlong`.

**Decision:** `pipekit-array.parallel.BatchedMap(op, batch_size=8,
axis=0)` splits the input array along `axis`, applies `op` to each
chunk, and concatenates the results back along `axis`. It is
**distinct from** `pipekit_ops.parallel.BatchedMap`, which takes an
iterable and applies `op` to each item.

The two coexist; users pick by carrier shape:

| `op` input type      | Use                              |
| -------------------- | -------------------------------- |
| iterable of carriers | `pipekit_ops.parallel.BatchedMap`    |
| single large array   | `pipekit_array.parallel.BatchedMap` |

**Rationale:** Same name, different problem. The iterable version is
for streams; the array version is for splitting one big tensor for
inference. Renaming either is gratuitous churn — the import path
disambiguates.

## A6. NaN semantics are documented, not hidden

**Decision:** Operators that touch NaN
(`AssertNoNaN`, `AssertValidFraction`) document the per-backend
edge cases in their docstrings. We do *not* try to unify them.

**Alternatives considered:**

- **Per-operator NaN normalisation** — call `xp.nan_to_num` before
  the check. Rejected: lossy, surprises users who passed NaN
  deliberately.
- **Skip NaN-touching operators in v0.1.** Rejected: these are the
  most-requested QC operators; the workaround is dropped users.

**Rationale:** The documented edge cases are small (numpy and JAX
agree on `xp.isnan` and `xp.sum(isnan_mask)`, which is all the v0.1
operators need). Other gaps are caught by the per-backend test matrix.

## A7. Dask laziness is the user's responsibility

**Decision:** pipekit-array supports `dask.array` inputs but does
not call `.compute()` for the user. A dask-backed pipeline returns a
dask array; the user calls `.compute()` (or `.persist()`, etc.) at
the consumption boundary.

**Alternatives considered:**

- **Auto-compute at operator boundaries** — kills the laziness
  benefit; turns dask into "slow numpy".
- **Auto-compute at pipeline boundary** — pipekit operators don't
  know they're at a "pipeline boundary"; the operator API is the
  same whether you compose ten of them or call one in a notebook.

**Rationale:** dask's value is the lazy graph; we don't break it.
Document the pattern; trust the user.

## A8. ModelOp split: pipekit-array vs pipekit-jax

**Decision:** The trained-model wrapper splits along
weight-round-trippability:

- **`pipekit_array.inference.ModelOp(model, method, batch_size)`** —
  the *generic* wrapper. Works for any callable model (sklearn
  `predict`, raw Keras `__call__`, plain PyTorch `nn.Module.forward`,
  numpy function). No registry round-trip — the model object is the
  artifact.
- **`pipekit_jax.JaxModelOp(module, ...)`** — the *weight-blob*
  variant for Equinox `eqx.Module`s. Implements
  `serialize_weights() → bytes` and `with_weights(blob) →
  JaxModelOp`, so `pipekit-experiment.ModelRegistry` can store and
  reload weights byte-identically (the v0.2 reproducibility upgrade
  for pipekit-train).

**Alternatives considered:**

- **One `ModelOp` for everything** — would force pipekit-array to
  carry an Orbax dependency for the JAX path. Rejected:
  pipekit-array has zero hard backend deps.
- **`ModelOp` always serializes via `pickle`.** Rejected: pickle
  isn't byte-stable across Python versions; defeats reproducibility.

**Rationale:** Two cases, two operators. `ModelOp` handles 95% of
users (PyTorch / Keras / sklearn) without touching JAX. `JaxModelOp`
handles the Equinox reproducibility case without pulling Orbax into
pipekit-array's dep cone.

`pipekit-train`'s Lightning / Keras adapters return
`pipekit-array.ModelOp` from `loop.run()`; the Equinox adapter
returns `EquinoxModelOp` (in-package, no round-trip) or
`pipekit-jax.JaxModelOp` (full round-trip) per user choice. The
contract is "the returned op is a `pipekit_ops.Operator`"; the specific
class differs by backend.

## A9. Named carriers: `DimArray`, through pipekit's adapter registry

**Decision (PK-P7, 23 Oct 2026):** pipekit-array's plain named carrier is
`DimArray(data, dims, labels)`, a JAX pytree whose only leaf is the data.
It is what a geo carrier becomes when a kernel consumes one of its
spatial dims (per-column statistics consume `row` and keep `col`), and
the carrier for tests and for libraries with none of their own. Its
adapter, `DimArrayAdapter`, is registered with `pipekit_ops.register_adapter`
on import and is the reference adapter every other one is tested
against.

`DimArray` holds no georeference, no coordinates beyond static labels and
no attrs, and does no alignment, arithmetic or selection by value; it
crosses to xarray with `to_xarray()` when an analysis needs those.

The design pages' separate two-field `Named` record (`data`, `dims`) is
folded into `DimArray`: a `DimArray` with no labels is that record.

**Rationale:** one plain carrier, not two, and one registry (pipekit
core's, PK-P1) through which the dim-level functions and `KernelOp` reach
it, `xr.DataArray` and geoarray's carriers alike. See geostack's
`docs/design/pipekit_kernels.md`.

## A10. Padded batches: `batch`, `Padded`, `stack_items`, `reduce(where=)`

**Decision (#119):** a fixed-size batch is `Padded(value, valid)`, a
`NamedTuple` of the items stacked along a new leading dim and a `(n,)`
bool mask of the real ones, made by `batch(items, *, size, dim,
pad="last" | "none" | "drop")`.

- **Padding copies the last real item.** Values stay finite and in
  distribution, so a masked loss, its gradient and a normalisation over
  the batch behave (the rule geoarray's `pad_points` and geopatcher's
  `PatchBatch` already follow).
- **Policy strings**, not `pad_last` / `drop_last` flags: one argument,
  three outcomes, no invalid combination.
- **`stack_items`, not `stack`.** `stack` joins carriers on one shared
  grid (geoarray's never aligns two). A batch's items may each carry
  their own geometry (patch transforms), the inverse of the split
  `map_over` makes, so it is a separate adapter verb. Without a fast path
  it is `stack`.
- **Masks are `reduce(..., where=)`**, not a `masked_mean`: every
  reduction, with the kept-count denominator, passed through to the
  namespace's `where=`. A named mask aligns by dim name; a plain array
  must match the carrier's leading dims (a `(sample,)` mask on
  `(sample, y, x)`), never NumPy's right-broadcast; a named mask's
  lengths must match the data's (or be 1). Only NumPy and JAX reductions
  take `where=`; dask, CuPy and PyTorch carriers raise `TypeError`.
- **The name is `Padded`, with `valid`.** "Masked" invites polarity bugs:
  geotoolz masks and `np.ma` masks are True = drop, `valid` is
  True = keep. `Batch` is taken by pipekit-train's `(x, y)` tuples.

**Rationale:** three copies of "group, pad, mask" across the stack
(geoarray `pad_points`, geopatcher `PatchBatch`, pipekit-train's eval
batches with no mask), one home for the rule.

## A11–A22. The missing dim-level functions

**Decision (6 Nov 2026):** sixteen dim-level functions join `dims`, each with an operator twin in `dim_operators`.
The full catalogue, the xarray parity table and the testing gates are in [api/dim_functions.md](api/dim_functions.md); the decisions are summarised here.

- **A11. Scope.** Only functions that change dims or move data by position; element-wise maths stays `combine`.
- **A12. xarray's names and argument names** (`rename`, `expand_dims`, `squeeze`, `broadcast_like`, `pad`, `roll`, `shift`, `diff`, `coarsen`, `quantile`, `where`, `dot`), with stated divergences: labels follow positions, shared dims agree exactly instead of outer-joining, no implicit `skipna`, dtype preserved, sizes static.
- **A13. No methods on `DimArray`.** A method would exist only on `DimArray`; a free function runs on every registered carrier.
- **A14. `merge_dims` / `split_dims`**, not `stack` / `unstack`, because `stack` already means a new dim from several carriers; merged labels are tuples, so the round trip is exact.
- **A15. Windows.** `window` returns full windows only; `coarsen` is `split_dims` then `reduce`; `ApplyAlong(inner, dim)` replaces the planned `ApplyToBands` and the planned `Subsample` is `isel` with a stepped slice.
- **A16. `where`, not `fill_where`**: True keeps, as in xarray and as `Padded.valid` does (A10).
- **A17. `dot`, not `contract`**, with `dim` required so a dim shared by accident is never summed; shared dims not in `dim` are batch dims, so the generic path is a named-index `einsum`, not `tensordot`.
- **A18. `quantile` is a function**, because a list of $q$ adds a dim that `reduce` cannot name.
- **A19. `BatchAlong` supersedes A5.** The planned array-shaped `BatchedMap` is renamed `BatchAlong(inner, size, dim, pad="none")`, so it no longer shares a name with `pipekit_ops.BatchedMap`.
- **A20. Label-value operations stay in xarray**: `groupby`, `resample`, `swap_dims`, `reindex`, `interp`, `idxmax`, `polyfit`, `differentiate`, `interpolate_na`, `rolling_exp`.
- **A21. The planned QC `Diff` becomes `AssertClose`**, so `Diff` is the twin of `diff`.
- **A22. An optional `relabel` adapter method** sets the labels `rewrap` cannot carry (renamed, merged, split, quantile, broadcast and contracted-in dims); without it those functions raise instead of dropping labels. The seven required methods do not change.

## Optional extras matrix

For reference, the extras shape that drops out of these decisions:

| Extra        | Pins                    | Why                                                   |
| ------------ | ----------------------- | ----------------------------------------------------- |
| `[numpy]`    | `numpy>=2.0`            | Array API conformance landed in numpy 2.0             |
| `[jax]`      | `jax>=0.4.20`           | Array API conformance landed in jax 0.4.20            |
| `[torch]`    | `torch>=2.0`            | Array API namespace in torch 2.0+                     |
| `[cupy]`     | `cupy>=13`              | Array API conformance from cupy 13                    |
| `[dask]`     | `dask[array]>=2024`     | Partial conformance; lazy                             |
| `[xarray]`   | `xarray>=2024.6`        | `DimArray.to_xarray`; the `DataArray` adapter (A9)    |
| `[compat]` *(proposed, v0.2)* | `array-api-compat>=1.4` | Optional shim for older / partial backends — not yet wired in `pyproject.toml`; lands with the v0.1 impl PRs |

The first five extras already exist in `packages/pipekit-array/pyproject.toml`.
The `[compat]` extra is a v0.2-proposed addition that the v0.1
implementation PRs will introduce alongside the `_namespace.py`
shim's `array_api_compat` fallback (see `architecture.md` §4).

The default install (`pip install pipekit-array`) brings nothing.
Backend availability is determined at call time via
`array_namespace(x)`: if the input doesn't implement the Array API,
`TypeError` is raised at call time, *not* at construction. The
recommended user install is `pipekit-array[numpy]`.
