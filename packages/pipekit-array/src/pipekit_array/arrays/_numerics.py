"""Quotients, roots, angles and powers whose gradient is finite where it matters.

A missing value stays missing: NaN in gives NaN out (ADR A26).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import array_api_compat

from pipekit_array.arrays._host import constants_like, like


__all__ = [
    "divide_or_nan",
    "safe_atan2",
    "safe_power",
    "safe_sqrt",
    "squared_magnitude",
    "stand_in",
]


def divide_or_nan(numerator: Any, denominator: Any, xp: ModuleType) -> Any:
    """The quotient of two arrays, NaN where the denominator is 0.

    It goes through a double ``where``, so a gradient is finite in the other
    elements: the division that is not used divides by 1.

    Args:
        numerator: An array of the namespace ``xp``.
        denominator: An array that broadcasts against it, of its dtype.
        xp: The namespace of the call.

    Returns:
        The quotient.

    Example:
        >>> import numpy as np
        >>> divide_or_nan(np.asarray([1.0, 1.0]), np.asarray([2.0, 0.0]), np)
        array([0.5, nan])
    """
    zero = denominator == 0
    safe = xp.where(zero, xp.ones_like(denominator), denominator)
    quotient = numerator / safe
    return xp.where(zero, xp.full_like(quotient, xp.nan), quotient)


def safe_sqrt(x: Any, xp: ModuleType) -> Any:
    """√x for x > 0, 0 with a zero gradient for x ≤ 0, and NaN for NaN.

    A difference of squares can round below 0, and its root is then 0; a missing
    value stays missing. It goes through a double ``where``, so the gradient is
    finite at 0.

    Args:
        x: A floating array of the namespace ``xp``.
        xp: The namespace of the call.

    Returns:
        The roots.

    Example:
        >>> import numpy as np
        >>> safe_sqrt(np.asarray([4.0, 0.0, -1.0, np.nan]), np)
        array([ 2.,  0.,  0., nan])
    """
    zero = x <= 0  # False for NaN
    # the root of 1 where the result is 0, so that the unused branch is finite
    root = xp.sqrt(xp.where(zero, xp.ones_like(x), x))
    return xp.where(zero, xp.zeros_like(x), root)


def safe_power(x: Any, p: Any, xp: ModuleType) -> Any:
    """xᵖ for x > 0, 0 with a zero gradient for x ≤ 0, and NaN for NaN.

    The rule at x ≤ 0 is that of `safe_sqrt`, and does not depend on p: it is 0
    for a negative p, too. The gradient by p is finite.

    Args:
        x: A floating array of the namespace ``xp``.
        p: The exponent, a number or an array that broadcasts against ``x``.
        xp: The namespace of the call.

    Returns:
        The powers.

    Example:
        >>> import numpy as np
        >>> safe_power(np.asarray([4.0, 0.0, -1.0, np.nan]), 1.5, np)
        array([ 8.,  0.,  0., nan])
    """
    zero = x <= 0
    power = xp.pow(xp.where(zero, xp.ones_like(x), x), constants_like(p, x, xp))
    return xp.where(zero, xp.zeros_like(power), power)


def safe_atan2(y: Any, x: Any, xp: ModuleType) -> Any:
    """atan2(y, x), and 0 with a zero gradient at the origin.

    Args:
        y: A floating array of the namespace ``xp``.
        x: An array that broadcasts against it, of its dtype.
        xp: The namespace of the call.

    Returns:
        The angles [rad], in (-π, π].

    Example:
        >>> import numpy as np
        >>> safe_atan2(np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0]), np).round(6)
        array([0.      , 1.570796])
    """
    origin = (x == 0) & (y == 0)
    # atan2(0, 1) = 0 at the origin, so that the unused branch is finite
    safe_x = xp.where(origin, xp.ones_like(x), x)
    safe_y = xp.where(origin, xp.zeros_like(y), y)
    return xp.atan2(safe_y, safe_x)


def squared_magnitude(z: Any, xp: ModuleType) -> Any:
    """|z|² = Re(z)² + Im(z)², with a gradient at 0, which that of |z| has not.

    Args:
        z: A real or complex array of the namespace ``xp``.
        xp: The namespace of the call.

    Returns:
        The squares, real.

    Example:
        >>> import numpy as np
        >>> squared_magnitude(np.asarray([3.0 + 4.0j, 0.0]), np)
        array([25.,  0.])
    """
    if not xp.isdtype(z.dtype, "complex floating"):
        return z * z
    return xp.real(z) ** 2 + xp.imag(z) ** 2


def stand_in(x: Any, mask: Any, xp: ModuleType, *, reference: Any = None) -> Any:
    """Rows of an array, with a valid one in place of each row to leave out.

    A function that is undefined at a row gives NaN there, and masking its result
    alone leaves 0 times NaN in the reverse pass, which spoils the gradient of every
    row. So it is evaluated somewhere valid instead, and its result masked.

    Args:
        x: The rows, (n, *rest), an array of the namespace ``xp``.
        mask: Which rows to use (True = use), boolean (n,), of the same library.
        xp: The namespace of the call.
        reference: The row, (*rest), to put in place of the others. If not given,
            the first one in use, or the first one of all if none is.

    Returns:
        The rows, in ``x``'s shape and dtype. No gradient reaches a row that is
        not in use, or the reference.

    Example:
        >>> import numpy as np
        >>> x = np.asarray([[np.nan, 0.0], [1.0, 2.0]])
        >>> stand_in(x, np.asarray([False, True]), np)
        array([[1., 2.],
               [1., 2.]])
    """
    if reference is None:
        first = xp.argmax(xp.astype(mask, xp.int32), keepdims=True)
        reference = xp.take(x, first, axis=0)[0, ...]
    reference = constant(like(reference, x, xp), xp)
    return xp.where(xp.reshape(mask, (-1, *[1] * (x.ndim - 1))), x, reference)


def constant(x: Any, xp: ModuleType) -> Any:
    """An array that no gradient reaches."""
    if array_api_compat.is_jax_namespace(xp):
        import jax

        return jax.lax.stop_gradient(x)
    return x.detach() if array_api_compat.is_torch_namespace(xp) else x
