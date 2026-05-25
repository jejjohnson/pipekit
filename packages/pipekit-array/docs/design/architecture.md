---
status: draft
version: 0.1.0
---

# Architecture

## 1. The dispatch pattern

Every operator follows the same three-line core:

```python
from pipekit import Operator
from pipekit_array._namespace import array_namespace


class MeanScalar(Operator):
    """Reduce an array to a scalar via xp.mean."""

    def __init__(self, axis: int | tuple[int, ...] | None = None) -> None:
        self.axis = axis

    def _apply(self, x):
        xp = array_namespace(x)
        return xp.mean(x, axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        return {"axis": self.axis}
```

The `array_namespace(x)` call is the entire backend-dispatch story.
It returns the module-like object — `numpy`, `jax.numpy`, `cupy`,
`torch._C._functorch.array_namespace`, `dask.array` — exposing the
Array API surface. The operator never imports a specific backend; the
input array picks.

## 2. The four shape rules

Every pipekit-array operator obeys these four rules. Reviewers should
treat any violation as a blocking comment:

### R1. Pure functions, no in-place mutation

```python
# ❌ Don't:
x[0] = 0
return x

# ✅ Do:
xp = array_namespace(x)
zero = xp.zeros_like(x[:1])
return xp.concatenate([zero, x[1:]], axis=0)
```

JAX immutability is the strict constraint that drives this. Numpy and
PyTorch users won't notice. CuPy follows numpy's semantics. Dask is
inherently immutable.

### R2. Single namespace per `_apply` call

```python
# ❌ Don't:
xp1 = array_namespace(x)
xp2 = array_namespace(y)  # different namespace?
return xp1.add(x, y)      # crashes if y is JAX and x is numpy

# ✅ Do:
xp = array_namespace(x, y)  # asserts both are same backend
return xp.add(x, y)
```

`array_namespace(x, y)` is the standard's way to assert "both inputs
are the same backend" and pick the namespace. Operators with multiple
array inputs always use the multi-arg form.

### R3. No backend-name conditionals

```python
# ❌ Don't:
xp = array_namespace(x)
if xp.__name__ == "jax.numpy":
    return jax_specific_path(x)
else:
    return generic_path(x)

# ✅ Do:
xp = array_namespace(x)
return generic_path(x, xp)  # via xp dispatch only
```

If an operator genuinely *needs* a per-backend path (e.g. an FFT
that's not in the Array API spec), it doesn't belong in
pipekit-array — it belongs in a per-backend sister package
(`pipekit-jax`, future `pipekit-torch`) or in user code. See ADR A4.

### R4. Output dtype matches input dtype (default)

```python
# ❌ Don't:
xp = array_namespace(x)
return xp.mean(x).astype(xp.float64)  # silently upcast

# ✅ Do:
xp = array_namespace(x)
return xp.mean(x, dtype=x.dtype)  # preserve
```

The principle of least surprise: a float32 input gives a float32
output. Operators that *must* change dtype (rare — `MeanScalar` with
`dtype=` is the obvious example) take an explicit `dtype` argument.

## 3. The three-layer surface

```
src/pipekit_array/
├── __init__.py          # Re-exports the public surface.
├── _namespace.py        # array_namespace() shim — wraps
│                        # array-api-compat if installed, falls back
│                        # to x.__array_namespace__() otherwise.
├── combinators.py       # ApplyToBands, StackAlong, ConcatenateAlong
├── geom.py              # Subsample
├── observe.py           # Histogram (carrier-aware controller)
├── reduce.py            # MeanScalar
├── parallel.py          # BatchedMap (array-shaped)
├── qc.py                # Diff, AssertValueRange, AssertNoNaN,
│                        # AssertValidFraction
└── inference.py         # ModelOp
```

The module boundaries mirror `pipekit` core's grouping
(`combinators` ↔ `pipekit.combine`, `qc` ↔ `pipekit.qc`, etc.) so
discovery is consistent. Re-exports through `__init__.py`:

```python
# pipekit_array/__init__.py
from pipekit_array.combinators import ApplyToBands, ConcatenateAlong, StackAlong
from pipekit_array.geom import Subsample
from pipekit_array.inference import ModelOp
from pipekit_array.observe import Histogram
from pipekit_array.parallel import BatchedMap
from pipekit_array.qc import (
    AssertNoNaN,
    AssertValidFraction,
    AssertValueRange,
    Diff,
)
from pipekit_array.reduce import MeanScalar

__all__ = [
    "ApplyToBands",
    "AssertNoNaN",
    "AssertValidFraction",
    "AssertValueRange",
    "BatchedMap",
    "ConcatenateAlong",
    "Diff",
    "Histogram",
    "MeanScalar",
    "ModelOp",
    "StackAlong",
    "Subsample",
]
```

## 4. The `_namespace.py` shim

Standard requires a tiny wrapper because the Array API namespace
discovery has two paths:

1. **Native conformance** (numpy 2.0+, JAX, CuPy, PyTorch 2.0+) —
   the array implements `__array_namespace__()` directly.
2. **Wrapped conformance** (older numpy, edge-case dask) — the
   `array-api-compat` library provides shims.

```python
# pipekit_array/_namespace.py
from __future__ import annotations
from typing import Any


def array_namespace(*xs: Any) -> Any:
    """Return the Array API namespace shared by all inputs.

    Tries `array_api_compat.array_namespace` first (handles older
    numpy and dask edge cases); falls back to the standard
    `x.__array_namespace__()` if compat shim is not installed.

    Raises:
        TypeError: if inputs span more than one backend, or if no
            input implements the Array API.
    """
    try:
        from array_api_compat import array_namespace as _compat_ns
        return _compat_ns(*xs)
    except ImportError:
        pass

    namespaces = {x.__array_namespace__() for x in xs if hasattr(x, "__array_namespace__")}
    if len(namespaces) == 0:
        raise TypeError(
            "No input implements the Array API. Install one of: "
            "pipekit-array[numpy], pipekit-array[jax], "
            "pipekit-array[torch], pipekit-array[cupy], "
            "pipekit-array[dask]."
        )
    if len(namespaces) > 1:
        names = sorted({ns.__name__ for ns in namespaces})
        raise TypeError(
            f"Inputs span multiple Array API backends: {names}. "
            "pipekit-array operators require a single backend per call."
        )
    return next(iter(namespaces))
```

The compat shim is a soft dependency — pipekit-array works without
it, but installing it (`pip install array-api-compat`) widens backend
coverage. Tracked as a v0.2 polish.

## 5. Integration with pipekit core

pipekit-array operators are *just* `pipekit.Operator` subclasses.
They inherit:

- **Composition** — `|` operator chains them into `Sequential`;
  `Graph` nodes accept them; `Fanout` parallel-maps them.
- **YAML round-trip** — `dumps(op)` / `loads(yaml)` serialises via
  `get_config()`. Same content-hashing as the core. Most v0.1
  operators round-trip cleanly because they don't carry user state;
  the two exceptions are flagged with
  `forbid_in_yaml: ClassVar[bool] = True`:
  - `ModelOp` — wraps an opaque trained-model object that isn't
    YAML-serialisable. The model is the artifact; the config
    captures only the call shape (see `api/inference.md`).
  - The per-tap operators returned by `Histogram.at(key)` — they
    close over the controller instance, which is interactive state.
    The captures dict is the artifact, not the config.

  All other v0.1 operators (`ApplyToBands`, `StackAlong`,
  `ConcatenateAlong`, `Subsample`, `MeanScalar`, `BatchedMap`, and
  the QC family) round-trip cleanly. `ApplyToBands` and `BatchedMap`
  take an `inner: Operator` which is itself round-trippable, so the
  recursion bottoms out cleanly unless `inner` is a flagged
  operator.
- **Signatures** — operators may override
  `compute_output_signature(input_sig: Signature) -> Signature | None`
  for `Sequential.summary` to print shape info. Operators that
  reshape (`ApplyToBands`, `Subsample`, `BatchedMap`) override;
  operators that pass through (the `Assert*` family) inherit the
  default passthrough. Operators that reduce to a scalar (`MeanScalar`
  with `axis=None`) return a `Signature` with empty `dims`.

The signature integration is the only piece pipekit-array adds beyond
"a bag of operators": it makes the new operators show up usefully in
`Sequential.summary` output without special-casing.

## 6. Integration with sister libraries

The pattern from Report 3 §1.5:

```python
# geotoolz/qc.py
from pipekit_array.qc import AssertValueRange as _AssertValueRange


class AssertValueRange(_AssertValueRange):
    """GeoTensor-aware value-range assertion.

    Same numeric logic, but reads GeoTensor's `fill_value` as the
    implicit min/max bounds if the user doesn't specify them.
    """

    def __init__(self, min_val=None, max_val=None, on_fail="raise") -> None:
        super().__init__(min_val=min_val, max_val=max_val, on_fail=on_fail)
```

geotoolz / xr_toolz add carrier-specific defaults on top of the
generic logic. The numeric work — `xp.any(x < min_val)` — lives once
in pipekit-array.

`xr_toolz` operators that wrap an `xr.DataArray` will typically:

1. Extract `.data` (the underlying array) and `.dims`.
2. Apply the pipekit-array operator to `.data`.
3. Reconstruct the `xr.DataArray` with the same coords / dims / attrs.

The extraction / reconstruction is xarray-specific; the array math is
pipekit-array. Same delegation pattern as geotoolz, different carrier.

## 7. Integration with pipekit-train

`pipekit-train` has historically shipped `EquinoxModelOp` in-package
as the v0.1 trained-model wrapper, with `pipekit-jax.JaxModelOp` as
the v0.2 weight-blob-round-trippable upgrade. **For the Lightning and
Keras adapters that land in pipekit-train v0.2 / v0.3,**
`pipekit-array.ModelOp` is the natural trained-model wrapper return
type. The boundary:

| Backend  | Trained-model op                       | Lives in                  |
| -------- | -------------------------------------- | ------------------------- |
| Equinox  | `EquinoxModelOp` (light) / `JaxModelOp` (round-trip) | `pipekit-train` / `pipekit-jax` |
| Lightning| `ModelOp`                              | `pipekit-array`           |
| Keras 3  | `ModelOp`                              | `pipekit-array`           |
| sklearn  | `ModelOp`                              | `pipekit-array`           |

This is the implementation of pipekit-train's open promise in
`docs/design/api/adapters.md`: "wrap the final module's `nn.Module`
in `pipekit-array.ModelOp` and return". See ADR A8.

## 8. Test matrix

The CI surface fans out across backends. Pattern from pipekit-train:

```python
# tests/test_mean_scalar.py
import pytest


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_mean_scalar_axis_none(backend):
    x = _make_array(backend, shape=(4, 8), seed=0)
    out = MeanScalar()(x)
    # Backend-specific assertion: ndim 0, dtype same as x
    assert out.ndim == 0
    assert out.dtype == x.dtype
```

The `_make_array(backend, ...)` helper is a single conftest fixture
that calls `pytest.importorskip(backend)` and returns a small random
array. Tests that need only one backend (`numpy` is the cheapest)
omit the parametrize.

`cupy` and `dask` are *not* in the core CI matrix because cupy needs
a GPU and dask shifts the timing characteristics significantly.
Both are exercised via opt-in `make test-cupy` / `make test-dask`
targets that run locally — same pattern as `pipekit-train`'s
`[lightning]` extra is gated.

Per-backend extras are validated separately by their own CI shape
gate, mirroring the `uv sync --extra equinox` discipline that
pipekit-train uses today (one `make` target per extra).
