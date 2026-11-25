"""Checks on random samples, with thresholds from the sampling distribution.

A result that is a sample is compared through one of these, with a constant key,
so that a test is deterministic and no tolerance is tuned to a seed.

From tracerax (``tests/_sampling.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np


__all__ = [
    "assert_mean",
    "assert_uniform",
    "assert_variance",
    "ks_critical_value",
    "ks_statistic",
]


def _draws(sample: Any, *, at_least: int = 30) -> np.ndarray:
    """The sample as a flat float64 array; every entry is finite."""
    draws = np.asarray(sample, dtype=np.float64).ravel()
    if draws.size < at_least:
        raise AssertionError(f"{draws.size} draws are too few, need {at_least}")
    if not np.isfinite(draws).all():
        raise AssertionError("the sample holds a draw that is not finite")
    return draws


def ks_statistic(sample: Any, low: float, high: float) -> float:
    """The Kolmogorov-Smirnov distance of a sample from the uniform law.

    D = sup_x |F_N(x) - (x - low) / (high - low)|.

    Args:
        sample: Independent draws, of any shape.
        low: The lower end of the interval.
        high: Its upper end.

    Returns:
        D.

    Raises:
        AssertionError: If there are fewer than 30 draws, or one is not finite.

    Example:
        >>> import numpy as np
        >>> round(ks_statistic((np.arange(50) + 0.5) / 50, 0.0, 1.0), 12)
        0.01
    """
    draws = np.sort(_draws(sample))
    size = draws.size
    # F(x) = (x - low) / (high - low)
    cdf = (draws - low) / (high - low)
    ranks = np.arange(1, size + 1)
    # D⁺ = max(i/N - F), D⁻ = max(F - (i - 1)/N)
    above = np.max(ranks / size - cdf)
    below = np.max(cdf - (ranks - 1) / size)
    return float(max(above, below))


def ks_critical_value(size: int, level: float) -> float:
    """The large-sample critical value of the Kolmogorov-Smirnov statistic.

    c = √(-½ ln(level / 2)) / √N, which is 1.95 / √N at a level of 10⁻³.

    Args:
        size: N, how many draws.
        level: The probability that a sample of the law exceeds it.

    Returns:
        c.

    Example:
        >>> round(ks_critical_value(100, 1e-3), 3)
        0.195
    """
    return float(np.sqrt(-0.5 * np.log(level / 2.0)) / np.sqrt(size))


def assert_uniform(
    sample: Any, low: float, high: float, *, level: float = 1e-3
) -> None:
    """Assert that independent draws are uniform on ``[low, high]``.

    Args:
        sample: Independent draws, of any shape.
        low: The lower end of the interval.
        high: Its upper end.
        level: The probability that a uniform sample fails.

    Raises:
        AssertionError: If a draw is outside the interval or not finite, if
            there are fewer than 30, or if the Kolmogorov-Smirnov statistic
            exceeds its critical value at N.

    Example:
        >>> import numpy as np
        >>> assert_uniform(
        ...     np.random.default_rng(0).uniform(2.0, 5.0, 1000), 2.0, 5.0
        ... )
    """
    draws = _draws(sample)
    outside = int(np.sum((draws < low) | (draws > high)))
    if outside:
        raise AssertionError(f"{outside} of {draws.size} draws outside [{low}, {high}]")
    statistic = ks_statistic(draws, low, high)
    critical = ks_critical_value(draws.size, level)
    if statistic > critical:
        raise AssertionError(
            f"not uniform on [{low}, {high}]: KS statistic {statistic:.4g} exceeds "
            f"{critical:.4g}, its critical value at N = {draws.size}, level {level}"
        )


def assert_mean(sample: Any, expected: float, *, n_sigma: float = 4.0) -> None:
    """Assert that a sample's mean is the expected one, within its sampling error.

    The bound is ``n_sigma`` standard errors s/√N of the sample itself: at 4 a
    right mean fails about once in 16,000 draws. The entries must be
    independent, so average over anything that shares a draw first.

    Args:
        sample: Independent draws, of any shape.
        expected: The mean they should have.
        n_sigma: The bound, in standard errors.

    Raises:
        AssertionError: If the mean is further away, if a draw is not finite, or
            if the sample is too small or has no spread to bound it by.

    Example:
        >>> import numpy as np
        >>> assert_mean(np.random.default_rng(0).normal(3.0, 1.0, 1000), 3.0)
    """
    draws = _draws(sample)
    # s/√N
    standard_error = float(np.std(draws, ddof=1)) / np.sqrt(draws.size)
    if standard_error == 0.0:
        raise AssertionError("the sample has no spread, so compare its values exactly")
    distance = abs(float(np.mean(draws)) - expected) / standard_error
    if distance > n_sigma:
        raise AssertionError(
            f"mean {np.mean(draws):.6g} is {distance:.1f} standard errors "
            f"({standard_error:.3g} each) from {expected:.6g}, more than {n_sigma}"
        )


def assert_variance(sample: Any, expected: float, *, n_sigma: float = 4.0) -> None:
    """Assert that Gaussian draws have the expected variance.

    The variance of N Gaussian draws has a relative standard error (2/N)^½, so
    the bound on s²/σ² - 1 is ``n_sigma`` of it. A bias of the scheme counts as
    an error, so the step must be short enough for it to be smaller.

    Args:
        sample: Independent Gaussian draws, of any shape.
        expected: The variance they should have.
        n_sigma: The bound, in standard errors.

    Raises:
        AssertionError: If the variance is further away, or if the sample is too
            small or holds a draw that is not finite.

    Example:
        >>> import numpy as np
        >>> assert_variance(np.random.default_rng(0).normal(0.0, 2.0, 1000), 4.0)
    """
    draws = _draws(sample)
    ratio = float(np.var(draws, ddof=1)) / expected
    # (2/N)^½
    standard_error = np.sqrt(2.0 / draws.size)
    distance = abs(ratio - 1.0) / standard_error
    if distance > n_sigma:
        raise AssertionError(
            f"variance is {ratio:.4f} of {expected:.6g}, {distance:.1f} standard "
            f"errors ({standard_error:.3g} each) from 1, more than {n_sigma}"
        )
