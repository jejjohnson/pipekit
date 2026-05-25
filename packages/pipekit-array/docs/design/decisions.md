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
default. `pipekit.observe.Histogram` keeps its `to_array: Callable`
constructor argument for the generic case.

**Alternatives considered:**

- **Add Array API logic to `pipekit.observe.Histogram`** — rejected:
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

**Decision:** `pipekit-array.parallel.BatchedMap(op, batch_size=8,
axis=0)` splits the input array along `axis`, applies `op` to each
chunk, and concatenates the results back along `axis`. It is
**distinct from** `pipekit.parallel.BatchedMap`, which takes an
iterable and applies `op` to each item.

The two coexist; users pick by carrier shape:

| `op` input type      | Use                              |
| -------------------- | -------------------------------- |
| iterable of carriers | `pipekit.parallel.BatchedMap`    |
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
contract is "the returned op is a `pipekit.Operator`"; the specific
class differs by backend.

## Optional extras matrix

For reference, the extras shape that drops out of these decisions:

| Extra        | Pins                    | Why                                                   |
| ------------ | ----------------------- | ----------------------------------------------------- |
| `[numpy]`    | `numpy>=2.0`            | Array API conformance landed in numpy 2.0             |
| `[jax]`      | `jax>=0.4.20`           | Array API conformance landed in jax 0.4.20            |
| `[torch]`    | `torch>=2.0`            | Array API namespace in torch 2.0+                     |
| `[cupy]`     | `cupy>=13`              | Array API conformance from cupy 13                    |
| `[dask]`     | `dask[array]>=2024`     | Partial conformance; lazy                             |
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
