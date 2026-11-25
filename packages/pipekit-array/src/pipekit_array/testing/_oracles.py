"""Comparisons with a reference: an oracle, and the adjoint of a linear map.

From functax (``tests/_harness.py``), by way of radiomax and diagnostix.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


__all__ = ["assert_adjoint", "assert_matches_oracle"]


def assert_matches_oracle(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float = 0.0,
    where: Any = None,
) -> None:
    """Assert ``|actual - expected| ≤ atol + rtol·|expected|`` wherever ``where`` holds.

    NaNs match only NaNs at the same index. Boolean and integer arrays must
    match exactly, whatever the tolerances.

    Args:
        actual: The values under test.
        expected: The reference values, the same shape.
        rtol: The relative tolerance.
        atol: The absolute tolerance, for references that cross zero.
        where: A boolean mask; only ``True`` entries are compared. ``None``
            compares every entry.

    Raises:
        AssertionError: If the shapes differ or any compared entry is out of
            tolerance. The message gives the number of failures, the worst
            index, both values there, the error and the tolerance.

    Example:
        >>> import numpy as np
        >>> assert_matches_oracle(
        ...     np.array([1.0, np.nan]), np.array([1.0, np.nan]), rtol=1e-12
        ... )
    """
    actual_values = np.asarray(actual)
    expected_values = np.asarray(expected)
    if actual_values.shape != expected_values.shape:
        raise AssertionError(
            f"shape mismatch: actual {actual_values.shape}, "
            f"expected {expected_values.shape}"
        )
    if where is None:
        mask = np.ones(actual_values.shape, dtype=bool)
    else:
        mask = np.broadcast_to(np.asarray(where, dtype=bool), actual_values.shape)
    floating = np.issubdtype(actual_values.dtype, np.inexact) or np.issubdtype(
        expected_values.dtype, np.inexact
    )
    if not floating:
        # Booleans and integers (masks, indices) must match exactly.
        actual_values = actual_values.astype(np.float64)
        expected_values = expected_values.astype(np.float64)
        rtol, atol = 0.0, 0.0

    # e = |a - b|; a NaN on one side only counts as an infinite error
    with np.errstate(invalid="ignore"):
        error = np.abs(actual_values - expected_values)
    both_nan = np.isnan(actual_values) & np.isnan(expected_values)
    # equal values (infinities included) have no error, and no inf - inf
    error = np.where(both_nan | (actual_values == expected_values), 0.0, error)
    error = np.where(np.isnan(error), np.inf, error)
    # τ = atol + rtol·|b|; a NaN or infinite reference contributes no tolerance
    reference_scale = np.where(np.isfinite(expected_values), expected_values, 0.0)
    tolerance = atol + rtol * np.abs(reference_scale)
    failing = mask & (error > tolerance)
    if not failing.any():
        return

    # The worst entry is the one furthest past its own tolerance.
    excess = np.where(failing, error - tolerance, -np.inf)
    worst = np.unravel_index(np.argmax(excess), excess.shape)
    raise AssertionError(
        f"{int(failing.sum())} of {int(mask.sum())} compared entries out of "
        f"tolerance; worst at index {tuple(int(i) for i in worst)}: "
        f"actual {actual_values[worst]!r}, expected {expected_values[worst]!r}, "
        f"error {error[worst]:.3e} > tolerance {tolerance[worst]:.3e} "
        f"(rtol={rtol}, atol={atol})"
    )


def assert_adjoint(
    apply: Callable[[Any], Any],
    apply_adjoint: Callable[[Any], Any],
    n_source: int,
    n_target: int,
    *,
    seed: int = 0,
    rtol: float = 1e-12,
) -> None:
    """Assert the dot-product test ⟨Hx, r⟩ = ⟨x, Gr⟩ on random NumPy x and r.

    Args:
        apply: H, a linear map from ℝᴺ to ℝᴹ.
        apply_adjoint: G, the claimed adjoint Hᵀ, from ℝᴹ to ℝᴺ.
        n_source: N, the length of x.
        n_target: M, the length of Hx.
        seed: The seed of x and r.
        rtol: The tolerance, relative to max(‖Hx‖ ‖r‖, ‖x‖ ‖Gr‖).

    Raises:
        AssertionError: If an output has the wrong shape, or the relative error
            exceeds ``rtol``.

    Example:
        >>> import numpy as np
        >>> H = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])
        >>> assert_adjoint(lambda x: H @ x, lambda r: H.T @ r, 3, 2)
    """
    rng = np.random.default_rng(seed)
    source = rng.standard_normal(n_source)
    residual = rng.standard_normal(n_target)
    forward = np.asarray(apply(source))
    backward = np.asarray(apply_adjoint(residual))
    if forward.shape != (n_target,):
        raise AssertionError(f"H x has shape {forward.shape}, expected ({n_target},)")
    if backward.shape != (n_source,):
        raise AssertionError(f"Hᵀ r has shape {backward.shape}, expected ({n_source},)")
    # ⟨Hx, r⟩ and ⟨x, Gr⟩
    lhs = float(forward @ residual)
    rhs = float(source @ backward)
    # the larger product, so a G that is too large also fails
    scale = max(
        float(np.linalg.norm(forward) * np.linalg.norm(residual)),
        float(np.linalg.norm(source) * np.linalg.norm(backward)),
    )
    if abs(lhs - rhs) > rtol * scale:
        raise AssertionError(
            f"dot-product test failed: <Hx, r> = {lhs!r}, <x, H^T r> = {rhs!r}, "
            f"relative error {abs(lhs - rhs) / scale:.3e} > rtol {rtol}"
        )
