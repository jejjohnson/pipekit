"""JAX's transforms and gradients of a function.

From functax (``tests/_harness.py``) and diagnostix.
"""

from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import Any

import numpy as np

from pipekit_array.testing._oracles import assert_matches_oracle


__all__ = ["assert_finite_gradient", "assert_grad_matches_fd", "assert_traces"]


def require_jax() -> Any:
    """JAX in its 64-bit mode, which every tolerance assumes.

    Raises:
        unittest.SkipTest: Without JAX; pytest skips the test.
        AssertionError: If the 64-bit mode is off.
    """
    try:
        import jax
        import jax.flatten_util
    except ImportError as err:
        raise unittest.SkipTest("JAX is not installed") from err
    if not jax.config.read("jax_enable_x64"):
        raise AssertionError(
            "this check needs float64: turn jax_enable_x64 on, in conftest.py for "
            "a whole suite"
        )
    return jax


def assert_traces(fn: Callable[..., Any], *args: Any, rtol: float = 1e-12) -> None:
    """Assert ``jit(fn)``, ``vmap(fn)`` and an eager loop agree over a batch.

    Every array of ``args`` has a leading batch axis of the same length B. The
    reference is the eager loop over it, stacked; ``jit(fn)`` on each slice,
    ``vmap(fn)`` and ``jit(vmap(fn))`` on the batch must match it. Skips without
    JAX.

    Args:
        fn: The function under test; it may take and return pytrees.
        *args: The batched arguments; a NumPy array goes to JAX.
        rtol: The relative tolerance.

    Raises:
        AssertionError: If JAX's 64-bit mode is off, the batch axes differ, or a
            variant disagrees; the message names the variant.

    Example:
        >>> assert_traces(lambda x: 2.0 * x, np.arange(4.0))  # doctest: +SKIP
    """
    jax = require_jax()
    args = _on_jax(jax, args)
    shapes = [np.shape(leaf) for leaf in jax.tree.leaves(args)]
    if len({shape[:1] for shape in shapes}) != 1 or () in shapes:
        raise AssertionError(f"every array needs the same leading batch axis: {shapes}")

    def stack(outputs: list[Any]) -> Any:
        return jax.tree.map(lambda *leaves: jax.numpy.stack(leaves), *outputs)

    slices = [
        jax.tree.map(lambda leaf, index=index: leaf[index], args)
        for index in range(shapes[0][0])
    ]
    reference = stack([fn(*one) for one in slices])
    jitted = jax.jit(fn)
    variants = {
        "jit(fn)": lambda: stack([jitted(*one) for one in slices]),
        "vmap(fn)": lambda: jax.vmap(fn)(*args),
        "jit(vmap(fn))": lambda: jax.jit(jax.vmap(fn))(*args),
    }
    for name, variant in variants.items():
        got = variant()
        structures = jax.tree.structure(got), jax.tree.structure(reference)
        if structures[0] != structures[1]:
            raise AssertionError(f"{name}: tree {structures[0]} != {structures[1]}")
        pairs = zip(jax.tree.leaves(got), jax.tree.leaves(reference), strict=True)
        for index, (actual, expected) in enumerate(pairs):
            try:
                assert_matches_oracle(actual, expected, rtol=rtol)
            except AssertionError as err:
                raise AssertionError(f"{name}, output {index}: {err}") from err


def assert_grad_matches_fd(
    fn: Callable[[Any], Any], x: Any, *, eps: float = 1e-6, rtol: float = 1e-6
) -> None:
    """Assert ``jax.jacrev(fn)`` agrees with central finite differences at x.

    For each entry k of the input, (f(x + ε eₖ) - f(x - ε eₖ)) / 2ε approximates
    column k of the Jacobian. The tolerance is relative to its largest entry,
    since an entry near zero carries the O(ε²) truncation error at full size, and
    never below the rounding of f divided by ε. Skips without JAX.

    Args:
        fn: The function under test, which JAX can trace.
        x: The point: a float64 array, or a pytree of them.
        eps: The step ε.
        rtol: The relative tolerance.

    Raises:
        AssertionError: If JAX's 64-bit mode is off, if the Jacobians differ, or
            if an entry of one is not finite; the message gives the (output,
            input) index.

    Example:
        >>> assert_grad_matches_fd(lambda x: x**2, np.ones(2))  # doctest: +SKIP
    """
    jax = require_jax()
    flat_fn, point = _flat(jax, fn, x)
    jacobian = np.asarray(jax.jit(jax.jacrev(flat_fn))(point))

    def beside(step: Any) -> Any:
        return flat_fn(point + step), flat_fn(point - step)

    steps = eps * jax.numpy.eye(point.shape[0], dtype=point.dtype)
    # compiled once, with one column per input entry
    above, below = jax.jit(jax.vmap(beside, out_axes=1))(steps)
    above, below = np.asarray(above), np.asarray(below)
    expected = (above - below) / (2.0 * eps)
    _require_finite("derivatives", jacobian)
    # a NaN scale would let every entry pass
    _require_finite("finite differences", expected)
    scale = float(np.max(np.abs(expected), initial=0.0))
    # the rounding of f, divided by the step: it is what is left at a minimum
    largest = float(np.max(np.abs([above, below]), initial=0.0))
    noise = float(np.finfo(expected.dtype).eps) * largest / eps
    assert_matches_oracle(jacobian, expected, rtol=0.0, atol=rtol * scale + noise)


def assert_finite_gradient(
    fn: Callable[[Any], Any], x: Any, *, where: Any = None
) -> None:
    """Assert the derivative of every output by each chosen input is finite.

    It checks that an undefined or invalid cell does not spoil the others: a
    single ``where`` passes a test of the values and fails this one, because 0
    times NaN is NaN in the backward pass. Skips without JAX.

    Args:
        fn: The function under test, which JAX can trace.
        x: The point, a float64 array; its invalid cells may be NaN.
        where: A boolean mask of the inputs to check, ``None`` for all.

    Raises:
        AssertionError: If JAX's 64-bit mode is off, or a derivative is not
            finite; the message gives the first (output, input) index.

    Example:
        >>> assert_finite_gradient(lambda x: x**2, np.zeros(2))  # doctest: +SKIP
    """
    jax = require_jax()
    flat_fn, point = _flat(jax, fn, x)
    jacobian = np.asarray(jax.jit(jax.jacrev(flat_fn))(point))
    chosen = True if where is None else np.ravel(np.broadcast_to(where, np.shape(x)))
    _require_finite("derivatives", jacobian, where=chosen)


def _on_jax(jax: Any, tree: Any) -> Any:
    """A pytree with its NumPy leaves on JAX."""
    return jax.tree.map(
        lambda leaf: (
            jax.numpy.asarray(leaf)
            if isinstance(leaf, (np.ndarray, np.generic))
            else leaf
        ),
        tree,
    )


def _flat(jax: Any, fn: Callable[[Any], Any], x: Any) -> tuple[Callable, Any]:
    """A function of one flat array, and the point as one."""
    ravel = jax.flatten_util.ravel_pytree
    point, unravel = ravel(_on_jax(jax, x))
    return lambda flat: ravel(fn(unravel(flat)))[0], point


def _require_finite(what: str, jacobian: np.ndarray, *, where: Any = True) -> None:
    """Raise if an entry of a Jacobian, (output, input), is not finite."""
    bad = ~np.isfinite(jacobian) & where
    if bad.any():
        output, entry = (int(index) for index in np.argwhere(bad)[0])
        raise AssertionError(
            f"{int(bad.sum())} {what} are not finite; the first is of output "
            f"{output} by input {entry}: {jacobian[output, entry]!r}"
        )
