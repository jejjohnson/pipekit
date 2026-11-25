"""Checks for the test suites of the libraries built on pipekit-array.

They ship in the wheel, so that a library tests itself with one call instead of
copying the check. Each raises `AssertionError`, saying where and by how much.

- **Contracts** of a carrier library (gates PK-K1 and PK-K4): `check_adapter`
  and `check_kernel_op`.
- **Array libraries**: `assert_backends_agree` runs a function on each kind of
  array that is installed (`array_kinds`), and `assert_refuses_dataarrays` checks
  that labels are never lost silently.
- **Values**: `assert_matches_oracle`, and `assert_adjoint`, the dot-product test.
- **JAX**: `assert_traces`, `assert_grad_matches_fd` and `assert_finite_gradient`.
  They skip without JAX, and need its 64-bit mode.
- **Samples**: `assert_mean`, `assert_variance` and `assert_uniform`, with
  thresholds from the sampling distribution.

Importing this module needs NumPy, and neither pytest nor another array library.
"""

from __future__ import annotations

from pipekit_array.testing._arrays import (
    ArrayKind,
    array_kinds,
    assert_backends_agree,
    assert_refuses_dataarrays,
)
from pipekit_array.testing._carriers import (
    DIMS,
    LABELS,
    check_adapter,
    check_kernel_op,
    values_of,
)
from pipekit_array.testing._oracles import assert_adjoint, assert_matches_oracle
from pipekit_array.testing._samples import (
    assert_mean,
    assert_uniform,
    assert_variance,
    ks_critical_value,
    ks_statistic,
)
from pipekit_array.testing._transforms import (
    assert_finite_gradient,
    assert_grad_matches_fd,
    assert_traces,
)


__all__ = [
    "DIMS",
    "LABELS",
    "ArrayKind",
    "array_kinds",
    "assert_adjoint",
    "assert_backends_agree",
    "assert_finite_gradient",
    "assert_grad_matches_fd",
    "assert_matches_oracle",
    "assert_mean",
    "assert_refuses_dataarrays",
    "assert_traces",
    "assert_uniform",
    "assert_variance",
    "check_adapter",
    "check_kernel_op",
    "ks_critical_value",
    "ks_statistic",
    "values_of",
]
