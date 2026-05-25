---
status: draft
version: 0.1.0
---

# QC — numeric assertions

Four pass-through assertions that look inside the array. They sit
above `pipekit.qc` (which stays carrier-agnostic — see its module
docstring's explicit deferral to pipekit-array) and below
`geotoolz.qc` / `xr_toolz.qa` (which subclass these with
carrier-specific defaults).

All four follow the same pass-through shape:

- On success: return the carrier unchanged.
- On failure: raise `QCError` (re-exported from `pipekit.qc`) or, for
  ops that take `on_fail="warn"`, emit a `UserWarning` and return
  unchanged.

The `on_fail` argument is consistent across the family:

| Value         | Behaviour                                                |
| ------------- | -------------------------------------------------------- |
| `"raise"`     | Raise `QCError` with a message describing the violation. |
| `"warn"`      | `warnings.warn(...)`, return the carrier unchanged.       |

Per ADR A6, NaN edge cases are documented per-operator; we don't
unify them.

---

## `Diff(reference: Array, atol: float = 1e-6, rtol: float = 0.0, on_fail: str = "raise")`

Compare the input to a stored reference array; raise (or warn) if
the per-element absolute difference exceeds `atol + rtol * |reference|`.

```python
gold = np.load("reference_output.npy")
checked = Diff(gold, atol=1e-5)
out = pipe | checked  # raises if pipe(x) drifts from gold
```

**Contract:**

- `_apply(x)`:
  1. `xp = array_namespace(x, self.reference)` (asserts same backend
     and converts mismatches to errors).
  2. Compute `delta = xp.abs(x - self.reference)`.
  3. Compute `tol = self.atol + self.rtol * xp.abs(self.reference)`.
  4. If `xp.any(delta > tol)`: dispatch per `on_fail`.
  5. Return `x` unchanged.

**`get_config()`:**

```python
{
    # The reference array is a numpy-shaped artifact. We serialise
    # it as base64 of the array's bytes, plus shape/dtype, so YAML
    # round-trips. Same pattern as TrainingDataset's content-hash
    # serialisation; spelled out in implementation phase C.
    "reference": {"data": b64, "shape": list(ref.shape), "dtype": str(ref.dtype)},
    "atol": self.atol,
    "rtol": self.rtol,
    "on_fail": self.on_fail,
}
```

**Use case:** regression tests for pipelines. The reference is
captured once (during the trusted run); subsequent runs diff against
it. The `atol` / `rtol` combination matches `xp.allclose`'s contract.

**Per-backend note:** All five backends support `xp.abs` and
`xp.any` per the spec.

---

## `AssertValueRange(min_val: float | None = None, max_val: float | None = None, on_fail: str = "raise")`

Pass-through; raise/warn if the input contains values outside
`[min_val, max_val]`. Either bound may be `None` for one-sided
checks.

```python
checked = AssertValueRange(0.0, 1.0)  # reflectance bounds
out = checked(arr)  # raises if any value < 0 or > 1
```

**Contract:**

- `_apply(x)`:
  1. `xp = array_namespace(x)`.
  2. `oob = xp.zeros_like(x, dtype=bool)`.
  3. If `min_val is not None`: `oob = oob | (x < self.min_val)`.
  4. If `max_val is not None`: `oob = oob | (x > self.max_val)`.
  5. If `xp.any(oob)`: dispatch per `on_fail` with a message
     including the offending value count and (for the raise path)
     a few example out-of-range values.
  6. Return `x` unchanged.

**`get_config()`:**

```python
{
    "min_val": self.min_val,  # float | None
    "max_val": self.max_val,
    "on_fail": self.on_fail,
}
```

**NaN handling:** NaN values are silently excluded from the
comparison (numpy's `nan < n == False`; same for JAX). To assert
no-NaN, compose with `AssertNoNaN()`.

**Sister-library override pattern** (Report 3 §1.5):

```python
# geotoolz/qc.py
from pipekit_array.qc import AssertValueRange as _AssertValueRange


class AssertValueRange(_AssertValueRange):
    """GeoTensor-aware: reads .fill_value as default bounds."""

    def __init__(self, min_val=None, max_val=None, on_fail="raise") -> None:
        # If GeoTensor with fill_value, default bounds from it.
        super().__init__(min_val=min_val, max_val=max_val, on_fail=on_fail)
```

---

## `AssertNoNaN(on_fail: str = "raise")`

Pass-through; raise/warn if the input contains any NaN values.

```python
checked = AssertNoNaN()
out = checked(arr)  # raises if any NaN
```

**Contract:**

- `_apply(x)`:
  1. `xp = array_namespace(x)`.
  2. If `xp.any(xp.isnan(x))`: dispatch per `on_fail` with a message
     including the NaN count and (for non-trivially-small arrays) the
     fraction of NaN entries.
  3. Return `x` unchanged.

**`get_config()`:** `{"on_fail": self.on_fail}`.

**Per-backend note:** `xp.isnan` is in the Array API spec; numpy,
JAX, CuPy, PyTorch all agree on the result. Integer dtypes never
contain NaN; the op is a no-op (fast path: skip if `not xp.isdtype(x.dtype, "real floating")`).

---

## `AssertValidFraction(min_valid: float = 0.5, on_fail: str = "raise")`

Pass-through; raise/warn if the *fraction* of non-NaN entries is
below `min_valid`. The "is this data usable" gate.

```python
checked = AssertValidFraction(min_valid=0.5)  # need ≥ 50% valid
out = checked(arr)  # raises if too many NaNs
```

**Contract:**

- `_apply(x)`:
  1. `xp = array_namespace(x)`.
  2. If `not xp.isdtype(x.dtype, "real floating")`: short-circuit
     (no NaN possible; valid fraction is 1.0).
  3. `valid_count = xp.sum(~xp.isnan(x))`.
  4. `total = x.size`.
  5. `valid_fraction = float(valid_count) / total`.
  6. If `valid_fraction < self.min_valid`: dispatch per `on_fail`
     with a message including the actual valid fraction and the
     threshold.
  7. Return `x` unchanged.

**`get_config()`:** `{"min_valid": self.min_valid, "on_fail":
self.on_fail}`.

**Distinction from `AssertNoNaN`:** `AssertNoNaN` requires *no* NaN
(strict); `AssertValidFraction` allows up to `1 - min_valid` NaN
(tolerant of partial cloud cover, sensor gaps, etc.). The two
compose: use `AssertValidFraction(0.5)` upstream of
`AssertNoNaN()` to first guard quality then guard zero-tolerance
downstream operations.

**Per-backend note:** `xp.sum` over a boolean array is in the spec;
all five backends agree on the cast-to-int-then-sum semantics.

---

## Failure messages

All four operators raise `QCError` (subclass of `AssertionError`,
re-exported from `pipekit.qc`) on failure. The message format is
operator-specific but follows a common shape:

```
QCError: AssertValueRange: 13 / 1024 values out of [0.0, 1.0].
  Examples: [1.04, -0.02, 1.71, ...]

QCError: AssertNoNaN: 47 NaN values found (4.6% of 1024).

QCError: AssertValidFraction: only 0.32 of values are valid;
  required >= 0.50.

QCError: Diff: 3 elements exceed atol=1e-06.
  Max abs diff: 2.3e-04. Reference: refs/output_v3.npy
```

The implementation phase pulls these into `pipekit_array.qc._messages`
to keep the per-operator code lean.
