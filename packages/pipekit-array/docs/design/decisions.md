---
status: draft
version: 0.1.0
---

# Decisions (ADRs)

Format: each decision states the choice, the alternatives considered and the reason.
Reviewers should treat these as the load-bearing commitments of the design; everything else follows from them.
An **As built** paragraph records where the shipped code (pipekit-array 0.1.1) departs from a decision or adds to it.

## A1. Dispatch via `array_namespace(x)`, not a backend registry

**Decision:** Operators call `array_namespace(x)` inside `_apply` and use the returned namespace exclusively.
There is no backend registry, no `Backend` enum and no `backend=` constructor argument.

**Alternatives considered:**

- **Backend registry:** `Operator(..., backend="jax")` selects the implementation.
  Rejected: it duplicates information the input array already carries, and adds a configuration knob the user has to keep in sync.
- **Per-backend operator subclasses:** `NumpyMeanScalar`, `JaxMeanScalar`.
  Rejected: it explodes the surface area and defeats the whole point of the Array API.

**Rationale:** `array_namespace(x)` is the standard's blessed dispatch path.
The input picks the backend and the operator stays backend-agnostic.
This is the single most important decision; it defines what pipekit-array *is*.

## A2. Strict purity: no in-place mutation

**Decision:** Every operator is side-effect-free.
No `x[i] = value`, no `x += other`, no `xp.copyto(dst, src)`.
Operators that need to "modify" return a fresh array via `xp.where`, `xp.concat` or `xp.stack`.

**Alternatives considered:**

- **Per-backend mutation:** mutate when the backend supports it (NumPy, torch), copy otherwise (JAX).
  Rejected: it makes operator semantics backend-dependent, breaks R1 of the shape rules, and defeats test reproducibility.
- **Optional in-place flag:** `Op(..., copy=False)`.
  Rejected: it silently breaks under JAX and lures users into backend coupling.

**Rationale:** JAX immutability is the strict constraint, and designing to it gets the others for free.
The performance gap is backend-internal: JAX's XLA compiler removes intermediate copies, and NumPy users pay one extra allocation per operator, which is negligible at the operator granularity pipekit-array works at.

## A3. Carrier-aware controllers ship in pipekit-array, not pipekit

**Decision:** `pipekit-array.observe.Histogram` ships its own controller, with `to_array=lambda x: xp.reshape(x, (-1,))` as the default.
`pipekit_ops.observe.Histogram` keeps its `to_array: Callable` constructor argument for the generic case.

**Alternatives considered:**

- **Add Array API logic to `pipekit_ops.observe.Histogram`.**
  Rejected: pipekit core has no third-party deps, by rule, and the Array API import would couple core to a sister library's domain.
- **Make `pipekit-array.Histogram` a subclass.**
  Rejected: the two Histograms share the controller pattern, but the array version has array-specific stat options (per-axis histograms, dtype-aware bin edges) that don't generalise to "an arbitrary carrier flattened to a sequence of numbers".

**Rationale:** the same logic with different defaults.
The carrier-aware version makes the right choice for arrays without forcing the user to spell it out, and the core version stays generic for non-array carriers (custom dataclasses, GeoTensor metadata stats, etc.).

**As built:** the array `Histogram` is not built yet (phase B).

## A4. No per-backend conditionals inside operators

**Decision:** If an operator can't be written against `array_namespace(x)` alone, it doesn't live in pipekit-array.
It either:

- lives in a per-backend sister package (`pipekit-jax`, a future `pipekit-torch`);
- lives in user code;
- doesn't exist (the operator is dropped from the catalogue).

**Alternatives considered:**

- **Internal `if xp.__name__ == "jax.numpy"` branches.**
  Rejected: the backend coupling defeats the dispatch story, tests must run per branch, and bugs become per-backend bugs.

**Rationale:** backend-specific tricks are the wrong layer's problem.
pipekit-array's value is the *generic* operator; if the operator can't be generic, it doesn't belong here.

**As built:** the dim-level functions keep a few backend checks, each in a shared helper and none in an operator's `_apply`.
`namespace_of` gives dask data a `_DaskNamespace` that maps the Array API names dask lacks (`permute_dims`, `concat`, `cumulative_sum`, `cumulative_prod`); `reduce(..., where=)` raises `TypeError` for backends other than NumPy and JAX; `quantile` rechunks dask's reduced axes into one chunk; and `dot` falls back to broadcast, multiply and sum on a namespace without `einsum` (architecture R3).

## A5. `pipekit-array.BatchedMap` is array-shaped, not a shadow

> Superseded by A19: the array-shaped operator is `BatchAlong`, and it ships.

**Decision:** `pipekit-array.parallel.BatchedMap(op, batch_size=8, axis=0)` splits the input array along `axis`, applies `op` to each chunk, and concatenates the results back along `axis`.
It is **distinct from** `pipekit_ops.parallel.BatchedMap`, which takes an iterable and applies `op` to each item.

The two coexist, and users pick by carrier shape:

| `op` input type | Use |
| --- | --- |
| iterable of carriers | `pipekit_ops.parallel.BatchedMap` |
| single large array | `pipekit_array.parallel.BatchedMap` |

**Rationale:** the same name for a different problem.
The iterable version is for streams; the array version is for splitting one big tensor for inference.
Renaming either is gratuitous churn, because the import path disambiguates.

## A6. NaN semantics are documented, not hidden

**Decision:** Operators that touch NaN (`AssertNoNaN`, `AssertValidFraction`) document the per-backend edge cases in their docstrings.
We do *not* try to unify them.

**Alternatives considered:**

- **Per-operator NaN normalisation:** call `xp.nan_to_num` before the check.
  Rejected: it is lossy, and it surprises users who passed NaN deliberately.
- **Skip NaN-touching operators in v0.1.**
  Rejected: these are the most-requested QC operators, and the workaround is losing users.

**Rationale:** the documented edge cases are small (NumPy and JAX agree on `xp.isnan` and `xp.sum(isnan_mask)`, which is all the v0.1 operators need).
Other gaps are caught by the per-backend test matrix.

**As built:** the dim-level functions take NaN explicitly instead of implicitly (A12): `quantile` has `skipna=False` by default, `cumreduce` asks for `"nancumsum"` by name, and `reduce` resolves `"nanmean"`, `"nanmedian"` and the like from the namespace.

## A7. Dask laziness is the user's responsibility

**Decision:** pipekit-array supports `dask.array` inputs but does not call `.compute()` for the user.
A dask-backed pipeline returns a dask array, and the user calls `.compute()` (or `.persist()`, etc.) at the consumption boundary.

**Alternatives considered:**

- **Auto-compute at operator boundaries:** this kills the laziness benefit and turns dask into "slow NumPy".
- **Auto-compute at the pipeline boundary:** pipekit operators don't know they're at a "pipeline boundary"; the operator API is the same whether you compose ten of them or call one in a notebook.

**Rationale:** dask's value is the lazy graph, and we don't break it.
We document the pattern and trust the user.

**As built:** the dim-level functions keep dask lazy (gate G10; `quantile`'s dask output stays lazy).
Two shipped pieces cannot: `Estimator` computes a dask input on the host, because scikit-learn computes on NumPy, and the einx bridge raises `BackendError` naming `.compute()`, because einx has no dask backend.

## A8. ModelOp split: pipekit-array vs pipekit-jax

**Decision:** The trained-model wrapper splits along weight round-trippability:

- **`pipekit_array.inference.ModelOp(model, method, batch_size)`** is the *generic* wrapper.
  It works for any callable model (sklearn `predict`, raw Keras `__call__`, plain PyTorch `nn.Module.forward`, a NumPy function).
  There is no registry round trip; the model object is the artifact.
- **`pipekit_jax.JaxModelOp(module, ...)`** is the *weight-blob* variant for Equinox `eqx.Module`s.
  It implements `serialize_weights() → bytes` and `with_weights(blob) → JaxModelOp`, so `pipekit-experiment.ModelRegistry` can store and reload weights byte-identically (the v0.2 reproducibility upgrade for pipekit-train).

**Alternatives considered:**

- **One `ModelOp` for everything:** this would force pipekit-array to carry an Orbax dependency for the JAX path.
  Rejected: pipekit-array has zero hard backend deps.
- **`ModelOp` always serialises via `pickle`.**
  Rejected: pickle isn't byte-stable across Python versions, which defeats reproducibility.

**Rationale:** two cases, two operators.
`ModelOp` handles 95% of users (PyTorch, Keras, sklearn) without touching JAX, and `JaxModelOp` handles the Equinox reproducibility case without pulling Orbax into pipekit-array's dependency cone.

`pipekit-train`'s Lightning and Keras adapters would return `pipekit-array.ModelOp` from `loop.run()`.
The Equinox adapter returns `EquinoxModelOp` (in-package, no round trip) or `pipekit-jax.JaxModelOp` (full round trip), per user choice.
The contract is "the returned op is a `pipekit_ops.Operator`"; the specific class differs by backend.

**As built:** `JaxModelOp` ships in pipekit-jax; `ModelOp` is not built yet (phase D).
For sklearn-style estimators on named carriers, `Estimator` ships instead (OP-O3): it is fitted and applied by dim name, and sets `forbid_in_yaml`.

## A9. Named carriers: `DimArray`, through pipekit's adapter registry

**Decision (PK-P7, 23 Oct 2026):** pipekit-array's plain named carrier is `DimArray(data, dims, labels)`, a JAX pytree whose only leaf is the data.
It is what a geo carrier becomes when a kernel consumes one of its spatial dims (per-column statistics consume `row` and keep `col`), and the carrier for tests and for libraries with none of their own.
Its adapter, `DimArrayAdapter`, is registered with `pipekit_ops.register_adapter` on import, and it is the reference adapter every other one is tested against.

`DimArray` holds no georeference, no coordinates beyond static labels and no attrs, and does no alignment, arithmetic or selection by value.
It crosses to xarray with `to_xarray()` when an analysis needs those.

The design pages' separate two-field `Named` record (`data`, `dims`) is folded into `DimArray`: a `DimArray` with no labels is that record.

**Rationale:** one plain carrier, not two, and one registry (pipekit core's, PK-P1) through which the dim-level functions and `KernelOp` reach it, `xr.DataArray` and geoarray's carriers alike.
See geostack's `docs/design/pipekit_kernels.md`.

## A10. Padded batches: `batch`, `Padded`, `stack_items`, `reduce(where=)`

**Decision (#119):** a fixed-size batch is `Padded(value, valid)`, a `NamedTuple` of the items stacked along a new leading dim and a `(n,)` bool mask of the real ones, made by `batch(items, *, size, dim, pad="last" | "none" | "drop")`.

- **Padding copies the last real item.**
  Values stay finite and in distribution, so a masked loss, its gradient and a normalisation over the batch behave (the rule geoarray's `pad_points` and geopatcher's `PatchBatch` already follow).
- **Policy strings**, not `pad_last` / `drop_last` flags: one argument, three outcomes, no invalid combination.
- **`stack_items`, not `stack`.**
  `stack` joins carriers on one shared grid (geoarray's never aligns two).
  A batch's items may each carry their own geometry (patch transforms), the inverse of the split `map_over` makes, so it is a separate adapter verb.
  Without a fast path it is `stack`.
- **Masks are `reduce(..., where=)`**, not a `masked_mean`: every reduction, with the kept-count denominator, passed through to the namespace's `where=`.
  A named mask aligns by dim name; a plain array must match the carrier's leading dims (a `(sample,)` mask on `(sample, y, x)`), never NumPy's right-broadcast; a named mask's lengths must match the data's (or be 1).
  Only NumPy and JAX reductions take `where=`; dask, CuPy and PyTorch carriers raise `TypeError`.
- **The name is `Padded`, with `valid`.**
  "Masked" invites polarity bugs: geotoolz masks and `np.ma` masks are True = drop, while `valid` is True = keep.
  `Batch` is taken by pipekit-train's `(x, y)` tuples.

**Rationale:** three copies of "group, pad, mask" across the stack (geoarray `pad_points`, geopatcher `PatchBatch`, pipekit-train's eval batches with no mask), and one home for the rule.

## A11 to A22. The missing dim-level functions

**Decision (6 Nov 2026):** sixteen dim-level functions join the named layer, each with an operator twin in `dim_operators`.
The full catalogue, the xarray parity table and the testing gates are in [api/dim_functions.md](api/dim_functions.md); the decisions are summarised here.
All sixteen shipped in #204 to #207, together with `ApplyAlong`, `BatchAlong` and `relabel` on both shipped adapters.

- **A11. Scope.** Only functions that change dims or move data by position; element-wise maths stays `combine`.
- **A12. xarray's names and argument names** (`rename`, `expand_dims`, `squeeze`, `broadcast_like`, `pad`, `roll`, `shift`, `diff`, `coarsen`, `quantile`, `where`, `dot`), with stated divergences: labels follow positions, shared dims agree exactly instead of outer-joining, no implicit `skipna`, dtype preserved, sizes static.
- **A13. No methods on `DimArray`.** A method would exist only on `DimArray`; a free function runs on every registered carrier.
- **A14. `merge_dims` / `split_dims`**, not `stack` / `unstack`, because `stack` already means a new dim from several carriers; merged labels are tuples, so the round trip is exact.
- **A15. Windows.** `window` returns full windows only; `coarsen` is `split_dims` then `reduce`; `ApplyAlong(inner, dim)` replaces the planned `ApplyToBands`, and the planned `Subsample` is `isel` with a stepped slice.
- **A16. `where`, not `fill_where`**: True keeps, as in xarray and as `Padded.valid` does (A10).
- **A17. `dot`, not `contract`**, with `dim` required so a dim shared by accident is never summed; shared dims not in `dim` are batch dims, so the generic path is a named-index `einsum`, not `tensordot`.
- **A18. `quantile` is a function**, because a list of $q$ adds a dim that `reduce` cannot name.
- **A19. `BatchAlong` supersedes A5.** The planned array-shaped `BatchedMap` is renamed `BatchAlong(inner, size, dim, pad="none")`, so it no longer shares a name with `pipekit_ops.BatchedMap`.
- **A20. Label-value operations stay in xarray**: `groupby`, `resample`, `swap_dims`, `reindex`, `interp`, `idxmax`, `polyfit`, `differentiate`, `interpolate_na`, `rolling_exp`.
- **A21. The planned QC `Diff` becomes `AssertClose`**, so `Diff` is the twin of `diff`.
- **A22. An optional `relabel` adapter method** (in the `LabelledAdapter` protocol of `pipekit_ops`) sets the labels `rewrap` cannot carry (renamed, merged, split, quantile, broadcast and contracted-in dims); without it those functions raise instead of dropping labels. The seven required methods do not change.

**As built (#204 to #207).** The PRs departed from these decisions in the following ways; [api/dim_functions.md](api/dim_functions.md) §6 has the detail under each ID.

- **Placement.** The functions live in four new modules, `dim_shapes`, `dim_shifts`, `dim_windows` and `dim_algebra`, not in `dims`, and are re-exported from `pipekit_array`.
- **A14.** `merge_dims` gives the merged dim no labels when none of its sources has any, and `split_dims` returns a part unlabelled when its recovered labels are exactly its positions.
- **A15.** The xarray `window` fast path fills with zero, not NaN, before `isel` drops the partial windows, so integer and bool input keep their dtype.
  A dim cannot be named like a keyword-only argument (`window_dim` or `stride` for `window`, `boundary` for `coarsen`), because Python binds the keyword first.
- **A16.** `where` on integer or bool data without `other` raises `TypeError`, the same rule as `shift` and `pad`.
- **A17, A18.** There are no xarray fast paths for `dot` or `quantile`: `xr.dot` silently inner-joins mismatched coordinates, and `DataArray.quantile` adds a scalar `quantile` coordinate for a float `q`.
  `quantile` casts `q` to the input's float dtype, and raises `ValueError` for a `q` outside [0, 1] or a list `q` that meets a kept dim named `quantile`.
- **A19.** `BatchAlong` restores `dim`'s labels after `concat` (which drops them) when the output length matches the input, and `pad="drop"` raises `ValueError` when every entry would be dropped.
- **A22.** pipekit-array finds `relabel` by name (`getattr`), so it still imports on a pipekit-ops without `LabelledAdapter`.

## Optional extras matrix

For reference, the extras that follow from these decisions, as declared in `packages/pipekit-array/pyproject.toml`:

| Extra | Pins | Why |
| --- | --- | --- |
| `[numpy]` | `numpy>=2.0` | Array API conformance landed in NumPy 2.0 |
| `[jax]` | `jax>=0.4.20` | Array API conformance landed in jax 0.4.20 |
| `[torch]` | `torch>=2.0` | Array API namespace in torch 2.0+ (dispatch needs `[compat]`) |
| `[cupy]` | `cupy>=13` | Array API conformance from CuPy 13 |
| `[dask]` | `dask[array]>=2024` | Partial conformance; lazy |
| `[xarray]` | `xarray>=2024.6` | `DimArray.to_xarray`; the `DataArray` adapter (A9) |
| `[sklearn]` | `scikit-learn>=1.4` | `Estimator` (geostack OP-O3) |
| `[einx]` | `einx>=0.4` | `pipekit_array.einx` (geostack OP-O2) |
| `[compat]` | `array-api-compat>=1.4` | The `_namespace.py` shim's first path; required for PyTorch |

The first design marked `[compat]` as a proposed v0.2 addition; it ships, together with the shim's `array_api_compat` path (`architecture.md` §4).

The default install (`pip install pipekit-array`) brings only `pipekit-ops`.
Backend availability is determined at call time via `array_namespace(x)`: if the input doesn't implement the Array API, `TypeError` is raised at call time, *not* at construction.
The recommended user install is `pipekit-array[numpy]`.
