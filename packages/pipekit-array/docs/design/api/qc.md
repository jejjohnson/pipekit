---
status: draft
version: 0.1.0
---

# QC: numeric assertions

> **Status: not built** (phase C of `boundaries.md` §4).
> `pipekit_array.qc` does not exist in pipekit-array 0.1.1, so every operator on this page is `proposed`.
> Until it lands, `pipekit_ops.qc.AssertThat` (an operator as the predicate) and `AssertCallable` (a function) cover one-off numeric checks.

Four pass-through assertions that look inside the array.
They sit above `pipekit_ops.qc`, which stays carrier-agnostic (its module docstring defers numeric checks to pipekit-array), and below `geoops.qc` / `xr_toolz.qa`, which would subclass these with carrier-specific defaults.

All four follow the same pass-through shape:

- on success, return the carrier unchanged;
- on failure, raise `QCError` (defined in `pipekit_ops.qc`) or, for ops that take `on_fail="warn"`, emit a `UserWarning` and return the carrier unchanged.

The `on_fail` argument is the same across the family:

| Value | Behaviour |
| --- | --- |
| `"raise"` | Raise `QCError` with a message describing the violation. |
| `"warn"` | `warnings.warn(...)`, then return the carrier unchanged. |

Per ADR A6, NaN edge cases are documented per operator; we don't unify them.

---

## `AssertClose(reference: Array, atol: float = 1e-6, rtol: float = 0.0, on_fail: str = "raise")`

Renamed from `Diff` by ADR A21: `Diff` is the operator twin of the dim-level `diff` ([dim_functions.md](dim_functions.md)), and `AssertClose` matches its `Assert*` siblings.
`Diff` ships today (#205); `AssertClose` is not built yet.

Compare the input to a stored reference array, and raise (or warn) if the per-element absolute difference exceeds `atol + rtol * |reference|`.

```python
gold = np.load("reference_output.npy")
# proposed (phase C)
checked = AssertClose(gold, atol=1e-5)
guarded = pipe | checked  # raises if pipe(x) drifts from gold
```

**Contract:**

- `_apply(x)`:
  1. `xp = array_namespace(x)`, to dispatch on the *input's* backend.
     The stored `self.reference` is always rehydrated as a NumPy array (see `get_config()` below), and it is adapted to the input's backend on every call via `ref = xp.asarray(self.reference)`.
     This handles the asymmetry where the operator is YAML-loaded with a NumPy reference but applied to JAX, torch or CuPy inputs.
  2. `delta = xp.abs(x - ref)`.
  3. `tol = self.atol + self.rtol * xp.abs(ref)`.
  4. If `xp.any(delta > tol)`, dispatch per `on_fail`.
  5. Return `x` unchanged.

**`get_config()`:**

```python
{
    # The reference is always a NumPy-shaped artifact in the
    # serialised form, regardless of what backend produced it at
    # capture time. We base64-encode the array's bytes plus shape /
    # dtype so YAML round-trips. At load time the reference is
    # reconstructed as a NumPy array; `_apply` then converts it to
    # the input's backend via `xp.asarray` on every call.
    "reference": {"data": b64, "shape": list(ref.shape), "dtype": str(ref.dtype)},
    "atol": self.atol,
    "rtol": self.rtol,
    "on_fail": self.on_fail,
}
```

The cross-backend conversion via `xp.asarray(numpy_ref)` is expected to be cheap: JAX, torch and CuPy all accept NumPy in `asarray`, and avoid a copy when the dtypes match and the data is host-resident.

**Use case:** regression tests for pipelines.
The reference is captured once (during the trusted run), and later runs compare against it.
The `atol` / `rtol` combination matches the contract of `xp.allclose`.

**Per-backend note:** all five backends support `xp.abs` and `xp.any` per the standard.

---

## `AssertValueRange(min_val: float | None = None, max_val: float | None = None, on_fail: str = "raise")`

Pass-through; raise or warn if the input has values outside `[min_val, max_val]`.
Either bound may be `None` for a one-sided check.

```python
# proposed (phase C)
checked = AssertValueRange(0.0, 1.0)  # reflectance bounds
out = checked(arr)  # raises if any value < 0 or > 1
```

**Contract:**

- `_apply(x)`:
  1. `xp = array_namespace(x)`.
  2. `oob = xp.zeros_like(x, dtype=bool)`.
  3. If `min_val is not None`: `oob = oob | (x < self.min_val)`.
  4. If `max_val is not None`: `oob = oob | (x > self.max_val)`.
  5. If `xp.any(oob)`, dispatch per `on_fail`, with a message that gives the count of offending values and (on the raise path) a few examples.
  6. Return `x` unchanged.

**`get_config()`:**

```python
{
    "min_val": self.min_val,  # float | None
    "max_val": self.max_val,
    "on_fail": self.on_fail,
}
```

**NaN handling:** NaN values are silently excluded from the comparison (in NumPy, `nan < n` is False, and the same holds in JAX).
To assert there is no NaN, compose with `AssertNoNaN()`.

**Sister-library override pattern** (Report 3 §1.5):

```python
# geotoolz/qc.py
# proposed (phase C)
from pipekit_array.qc import AssertValueRange as _AssertValueRange


class AssertValueRange(_AssertValueRange):
    """GeoTensor-aware: reads .fill_value as default bounds."""

    def __init__(self, min_val=None, max_val=None, on_fail="raise") -> None:
        # If GeoTensor with fill_value, default bounds from it.
        super().__init__(min_val=min_val, max_val=max_val, on_fail=on_fail)
```

---

## `AssertNoNaN(on_fail: str = "raise")`

Pass-through; raise or warn if the input contains any NaN.

```python
# proposed (phase C)
checked = AssertNoNaN()
out = checked(arr)  # raises if any NaN
```

**Contract:**

- `_apply(x)`:
  1. `xp = array_namespace(x)`.
  2. If `xp.any(xp.isnan(x))`, dispatch per `on_fail`, with a message that gives the NaN count and (for arrays that are not tiny) the fraction of NaN entries.
  3. Return `x` unchanged.

**`get_config()`:** `{"on_fail": self.on_fail}`.

**Per-backend note:** `xp.isnan` is in the Array API standard, and NumPy, JAX, CuPy and PyTorch agree on the result.
Integer dtypes never contain NaN, so the op is a no-op there (fast path: skip if `not xp.isdtype(x.dtype, "real floating")`).

---

## `AssertValidFraction(min_valid: float = 0.5, on_fail: str = "raise")`

Pass-through; raise or warn if the *fraction* of non-NaN entries is below `min_valid`.
It is the "is this data usable" gate.

```python
# proposed (phase C)
checked = AssertValidFraction(min_valid=0.5)  # need at least 50% valid
out = checked(arr)  # raises if too many NaNs
```

With $N$ the number of entries of the input and $x_n \in \mathbb{R}$ its $n$-th entry, the check is

```math
f_\text{valid} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{1}[x_n \text{ is not NaN}] \ge f_\text{min}
```

where $f_\text{min} \in [0, 1]$ is `min_valid` and $f_\text{valid} \in [0, 1]$ is the valid fraction.

**Contract:**

- `_apply(x)`:
  1. `xp = array_namespace(x)`.
  2. If `not xp.isdtype(x.dtype, "real floating")`, short-circuit (no NaN is possible, so the valid fraction is 1.0).
  3. `valid_count = xp.sum(~xp.isnan(x))`.
  4. `total = x.size`.
  5. `valid_fraction = float(valid_count) / total`.
  6. If `valid_fraction < self.min_valid`, dispatch per `on_fail`, with a message that gives the actual valid fraction and the threshold.
  7. Return `x` unchanged.

**`get_config()`:** `{"min_valid": self.min_valid, "on_fail": self.on_fail}`.

**Distinction from `AssertNoNaN`:** `AssertNoNaN` requires *no* NaN (strict), while `AssertValidFraction` allows up to `1 - min_valid` NaN (tolerant of partial cloud cover, sensor gaps and the like).
The two compose: use `AssertValidFraction(0.5)` upstream of `AssertNoNaN()`, to guard quality first and then guard the downstream operations that tolerate no NaN.

**Per-backend note:** `xp.sum` over a boolean array is in the standard, and all five backends agree on the cast-to-int-then-sum semantics.

---

## Failure messages

All four operators raise `QCError` (a subclass of `AssertionError`, defined in `pipekit_ops.qc`) on failure.
The message format is operator-specific but follows a common shape:

```
QCError: AssertValueRange: 13 / 1024 values out of [0.0, 1.0].
  Examples: [1.04, -0.02, 1.71, ...]

QCError: AssertNoNaN: 47 NaN values found (4.6% of 1024).

QCError: AssertValidFraction: only 0.32 of values are valid;
  required >= 0.50.

QCError: AssertClose: 3 elements exceed atol=1e-06.
  Max abs diff: 2.3e-04. Reference: refs/output_v3.npy
```

The implementation phase pulls these into `pipekit_array.qc._messages`, to keep the per-operator code lean.
