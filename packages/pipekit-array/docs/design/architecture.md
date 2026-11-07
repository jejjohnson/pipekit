---
status: draft
version: 0.1.0
---

# Architecture

## 1. The dispatch pattern

Every operator follows the same three-line core, as `MeanScalar` does in [`reductions.py`](../../src/pipekit_array/reductions.py):

```python
from typing import Any

from pipekit_ops import Operator

from pipekit_array._namespace import array_namespace


class MeanScalar(Operator):
    """Reduce an array to a scalar (or per-axis means) via xp.mean."""

    def __init__(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def _apply(self, x: Any) -> Any:
        xp = array_namespace(x)
        return xp.mean(x, axis=self.axis, keepdims=self.keepdims)

    def get_config(self) -> dict[str, Any]:
        return {"axis": self.axis, "keepdims": self.keepdims}
```

The `array_namespace(x)` call is the entire backend-dispatch story.
It returns the module-like object that exposes the Array API surface: `numpy`, `jax.numpy`, `cupy`, `dask.array`, or `array_api_compat.torch` for PyTorch (which needs the `[compat]` extra, because torch tensors do not implement `__array_namespace__` themselves).
The operator never imports a specific backend; the input array picks.

The shipped `MeanScalar` also turns a list `axis` (as it comes back from JSON) into a tuple, and implements `compute_output_signature`; the sketch leaves both out.

## 2. The four shape rules

Every pipekit-array operator obeys these four rules.
Reviewers should treat any violation as a blocking comment.

### R1. Pure functions, no in-place mutation

```python
# Don't:
x[0] = 0
return x

# Do:
xp = array_namespace(x)
zero = xp.zeros_like(x[:1])
return xp.concat([zero, x[1:]], axis=0)
```

JAX immutability is the strict constraint that drives this.
NumPy and PyTorch users won't notice, CuPy follows NumPy's semantics, and dask is immutable by design.

### R2. Single namespace per `_apply` call

```python
# Don't:
xp1 = array_namespace(x)
xp2 = array_namespace(y)  # a different namespace?
return xp1.add(x, y)  # crashes if y is JAX and x is NumPy

# Do:
xp = array_namespace(x, y)  # raises TypeError on two backends
return xp.add(x, y)
```

`array_namespace(x, y)` is the standard's way to assert "both inputs are on the same backend" and pick the namespace.
Operators with several array inputs always use the multi-argument form.

The named layer has its own form of this rule: `require_one_backend` in [`dims.py`](../../src/pipekit_array/dims.py) raises `BackendError` when the inputs of `dot`, `combine`, `concat`, `stack` or `take_along` sit on more than one backend.

### R3. No backend-name conditionals

```python
# Don't:
xp = array_namespace(x)
if xp.__name__ == "jax.numpy":
    return jax_specific_path(x)
else:
    return generic_path(x)

# Do:
xp = array_namespace(x)
return generic_path(x, xp)  # via xp dispatch only
```

If an operator genuinely *needs* a per-backend path (for example an FFT that is not in the Array API standard), it does not belong in pipekit-array.
It belongs in a per-backend sister package (`pipekit-jax`, a future `pipekit-torch`) or in user code (ADR A4).

**As built.** The dim-level functions bend this rule in a few places, each confined to a helper in `dims.py` or `dim_algebra.py` rather than written inside an operator:

- `namespace_of` returns a `_DaskNamespace` for dask data, which maps the Array API names dask lacks (`permute_dims`, `concat`, `cumulative_sum`, `cumulative_prod`) to `transpose`, `concatenate`, `cumsum` and `cumprod`, and supplies a blockwise `take_along_axis`;
- `reduce(..., where=)` checks the backend and raises `TypeError` for anything but NumPy and JAX, whose reductions are the only ones that take `where=`;
- `quantile` rechunks the reduced axes of dask data into one chunk;
- `dot` falls back to broadcast, multiply and sum on a namespace without `einsum`.

### R4. Output dtype matches input dtype (default)

```python
# Don't:
xp = array_namespace(x)
return xp.astype(xp.mean(x), xp.float64)  # silently upcast

# Do:
xp = array_namespace(x)
return xp.mean(x)  # the Array API mean keeps a floating input's dtype
```

The principle of least surprise: a float32 input gives a float32 output.
Operators that *must* change dtype take an explicit `dtype` argument.

**As built.** The dim-level functions keep this rule (gate G8 of [api/dim_functions.md](api/dim_functions.md) §7):

- `quantile` casts `q` to the input's float dtype, because NumPy would otherwise promote float32 data to float64 for a list of Python floats;
- `shift`, `pad(mode="constant")` and `where` on integer or bool data raise `TypeError` when no fill value is given, instead of promoting to float for a NaN fill;
- the xarray fast path of `window` fills with zero, not NaN, before it drops the partial windows, so integer and bool input keep their dtype.

## 3. The three-layer surface

The planned layout of the v0.1 catalogue, and what ships in 0.1.1:

```
src/pipekit_array/
├── __init__.py          # Re-exports the public surface; registers the adapters.
├── _namespace.py        # array_namespace(): array-api-compat if installed,
│                        # x.__array_namespace__() otherwise.
├── reductions.py        # MeanScalar                              (today)
├── combinators.py       # StackAlong, ConcatenateAlong            (today)
├── dim_operators.py     # ApplyAlong and the dim-function twins   (today)
├── batching.py          # batch, Padded, BatchAlong               (today)
├── geom.py              # Subsample                    (retired: isel, A15)
├── observe.py           # Histogram (carrier-aware controller)  (not built)
├── qc.py                # AssertClose, AssertValueRange,         (not built)
│                        # AssertNoNaN, AssertValidFraction
└── inference.py         # ModelOp                               (not built)
```

The named-carrier layer that grew alongside it (all today):

```
src/pipekit_array/
├── dimarray.py          # DimArray, DimArrayAdapter (the reference adapter)
├── dims.py              # isel, sel, transpose, reduce, argreduce, take_along,
│                        # map_over, stack, stack_items, concat, combine
├── dim_shapes.py        # rename, expand_dims, squeeze, broadcast_like,
│                        # merge_dims, split_dims                  (#204)
├── dim_shifts.py        # pad, roll, shift, diff, cumreduce, where (#205)
├── dim_windows.py       # window, coarsen                          (#206)
├── dim_algebra.py       # quantile, dot                            (#207)
├── rows.py              # to_rows, from_rows, RowLayout
├── estimator.py         # Estimator                         ([sklearn])
├── einx.py              # rearrange, reduce, einsum, repeat, Einx  ([einx])
├── kernel.py            # KernelSpec, KernelOp
├── xarray.py            # the xr.DataArray adapter, loaded lazily ([xarray])
└── testing.py           # check_adapter, check_kernel_op
```

The module boundaries mirror `pipekit-ops` core's grouping (`combinators` with `pipekit_ops.combine`, `qc` with `pipekit_ops.qc`, and so on), so discovery is consistent.
The planned re-exports through `__init__.py`, marked by status:

```python
# pipekit_array/__init__.py
# today (pipekit-array)
from pipekit_array.batching import BatchAlong
from pipekit_array.combinators import ConcatenateAlong, StackAlong
from pipekit_array.dim_operators import ApplyAlong
from pipekit_array.reductions import MeanScalar

# proposed (phase B; retired by A15)
from pipekit_array.geom import Subsample

# proposed (phase D)
from pipekit_array.inference import ModelOp

# proposed (phase B)
from pipekit_array.observe import Histogram

# proposed (phase C)
from pipekit_array.qc import (
    AssertClose,
    AssertNoNaN,
    AssertValidFraction,
    AssertValueRange,
)
```

The shipped `__all__` lists the `today` names above plus the named layer: `DimArray`, every dim function and twin, `batch`, `Padded`, `to_rows`, `from_rows`, `RowLayout`, `Estimator`, `KernelSpec`, `KernelOp` and `BackendError`.
`pipekit_array.einx` and `pipekit_array.testing` are submodules, imported by name.

## 4. The `_namespace.py` shim

The standard needs a small wrapper because Array API namespace discovery has two paths:

1. **Native conformance** (NumPy 2.0+, JAX, CuPy): the array implements `__array_namespace__()` directly.
2. **Wrapped conformance** (PyTorch, older NumPy, dask edge cases): the `array-api-compat` library provides the namespace.

The shipped shim in [`_namespace.py`](../../src/pipekit_array/_namespace.py), simplified:

```python
# pipekit_array/_namespace.py
def array_namespace(*xs: Any) -> Any:
    """Return the Array API namespace shared by all inputs."""
    compat_ns = _compat_namespace()  # looked up once, then cached
    if compat_ns is not None:
        try:
            return compat_ns(*xs)
        except TypeError as exc:
            # re-raised with the same wording as the path below
            raise TypeError(_NO_BACKEND_MSG) from exc

    namespaces = set()
    for x in xs:
        ns = getattr(x, "__array_namespace__", None)
        if ns is None:
            raise TypeError(f"Input of type {type(x).__name__!r} ...")
        namespaces.add(ns())
    if len(namespaces) == 0:
        raise TypeError(_NO_BACKEND_MSG)
    if len(namespaces) > 1:
        names = sorted({ns.__name__ for ns in namespaces})
        raise TypeError(_multi_backend_msg(names))
    return next(iter(namespaces))
```

Three details differ from the first design:

- the `array_api_compat` import is tried once and cached (`_compat_namespace`), because a failed import is not cached by Python and cost about 0.2 ms per call;
- compat's `TypeError`s are re-raised with pipekit-array's own wording, so the message is the same with and without the shim, and a multi-backend error names the backends;
- an input without `__array_namespace__` raises at once, naming its type, instead of being skipped.

The compat shim is a soft dependency: pipekit-array works without it, and installing it widens backend coverage.
It ships as the `[compat]` extra (`array-api-compat>=1.4`), which PyTorch inputs need.

## 5. Integration with pipekit core

pipekit-array operators are *just* `pipekit_ops.Operator` subclasses.
They inherit:

- **Composition:** `|` chains them into a `Sequential`, `Graph` nodes accept them, and `Fanout` runs them side by side.
- **YAML round trip:** `dumps(op)` / `loads(yaml)` serialise through `get_config()`, with the same content hashing as the core.
  Most v0.1 operators round-trip cleanly because they carry no user state.
  The two planned exceptions set `forbid_in_yaml: ClassVar[bool] = True`:
  - `ModelOp` wraps an opaque trained-model object that is not YAML-serialisable.
    The model is the artifact; the config captures only the call shape (see `api/inference.md`).
  - The per-tap operators returned by `Histogram.at(key)` close over the controller instance, which is interactive state.
    The captures dict is the artifact, not the config.

  All other v0.1 operators (`ApplyAlong`, `StackAlong`, `ConcatenateAlong`, `Subsample`, `MeanScalar`, `BatchAlong` and the QC family) round-trip cleanly.
  `ApplyAlong` and `BatchAlong` take an `inner: Operator` that is itself round-trippable, stored as a nested `{"module", "class", "config"}` record, so the recursion bottoms out cleanly unless `inner` is a flagged operator.

  **As built**, the shipped operators that set `forbid_in_yaml` are `MapOver` and `Combine` (they hold a callable), `Estimator` (it holds a fitted estimator) and `KernelOp`.
  Gate A1 (`test_gate_a1_round_trip_array.py`) round-trips every other operator in `pipekit_array.__all__`.
- **Signatures:** operators may override `compute_output_signature(input_sig: Signature) -> Signature | None` so that `Sequential.summary` can print shape information.
  Operators that reshape (`ApplyAlong`, `Subsample`, `BatchAlong`) override it, and operators that pass through (the `Assert*` family) inherit the default.
  Operators that reduce to a scalar (`MeanScalar` with `axis=None`) return a `Signature` with empty `dims`.

The signature integration is the only piece pipekit-array adds beyond "a bag of operators": it makes the new operators show up usefully in `Sequential.summary` output without special-casing.

## 6. Integration with sister libraries

The planned pattern from Report 3 §1.5:

```python
# geotoolz/qc.py
# proposed (phase C): pipekit_array.qc does not exist yet
from pipekit_array.qc import AssertValueRange as _AssertValueRange


class AssertValueRange(_AssertValueRange):
    """GeoTensor-aware value-range assertion.

    Same numeric logic, but reads GeoTensor's `fill_value` as the
    implicit min/max bounds if the user doesn't specify them.
    """

    def __init__(self, min_val=None, max_val=None, on_fail="raise") -> None:
        super().__init__(min_val=min_val, max_val=max_val, on_fail=on_fail)
```

geotoolz and xr_toolz add carrier-specific defaults on top of the generic logic.
The numeric work (`xp.any(x < min_val)`) lives once, in pipekit-array.

`xr_toolz` operators that wrap an `xr.DataArray` would typically:

1. extract `.data` (the underlying array) and `.dims`;
2. apply the pipekit-array operator to `.data`;
3. rebuild the `xr.DataArray` with the same coords, dims and attrs.

The extraction and reconstruction are xarray-specific; the array maths is pipekit-array's.
It is the same delegation pattern as geotoolz, on a different carrier.

**As built.** For the dim-level functions the delegation already lives in pipekit-array: the `[xarray]` adapter (`pipekit_array.xarray`) is registered lazily with `pipekit_ops`, so `rename`, `window`, `dot` and the rest run on an `xr.DataArray` directly and keep its coordinates.
geoarray carriers reach the same functions through their own registered adapter (ADR A9).

## 7. Integration with pipekit-train

`pipekit-train` ships `EquinoxModelOp` in-package as its trained-model wrapper, with `pipekit_jax.JaxModelOp` as the weight-blob round-trippable upgrade.
**For the Lightning and Keras adapters planned for pipekit-train v0.2 and v0.3**, `pipekit_array.ModelOp` (not built yet) is the natural trained-model wrapper to return.
The boundary:

| Backend | Trained-model op | Lives in |
| --- | --- | --- |
| Equinox | `EquinoxModelOp` (light) / `JaxModelOp` (round trip) | `pipekit-train` / `pipekit-jax` |
| Lightning | `ModelOp` | `pipekit-array` (not built) |
| Keras 3 | `ModelOp` | `pipekit-array` (not built) |
| sklearn | `ModelOp` | `pipekit-array` (not built) |

This implements pipekit-train's open promise in its `docs/design/api/adapters.md`: "wrap the final module's `nn.Module` in `pipekit-array.ModelOp` and return" (ADR A8).
For sklearn on named carriers, `Estimator` already ships (`[sklearn]`): it fits and applies an estimator by dim name through `to_rows` and `from_rows`.

## 8. Test matrix

The CI surface fans out across backends.
The pattern, as in [`tests/test_reduce.py`](../../tests/test_reduce.py):

```python
# tests/test_reduce.py
from pipekit_array import MeanScalar


def test_mean_scalar_full_reduction(backend, make_array):
    x = make_array(backend, shape=(4, 8), seed=0)
    out = MeanScalar()(x)
    assert out.ndim == 0
```

`backend` is a conftest fixture parametrised over `("numpy", "jax", "torch")`, and `make_array` returns the `_make_array(backend, shape, seed)` helper, which calls `pytest.importorskip` and returns a small random float32 array.
Tests that need only one backend (`numpy` is the cheapest) leave the `backend` fixture out.

The first design kept `cupy` and `dask` out of the core CI matrix, because CuPy needs a GPU and dask shifts the timing a lot, and planned opt-in `make test-cupy` / `make test-dask` targets.
**As built**, neither target exists:

- `dask` (with `xarray`, `scikit-learn` and `einx`) is in the workspace dev group, so the dim-function, xarray and batching tests run on dask in the main suite;
- `torch` and `array-api-compat` are not in the dev group, so the torch cases of the `backend` fixture skip;
- CuPy is not exercised.

Per-backend extras would be validated separately by their own CI shape gate, mirroring the `uv sync --extra equinox` discipline pipekit-train uses (one `make` target per extra).
