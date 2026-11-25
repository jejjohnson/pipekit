"""`pipekit_array.arrays`: quotients, roots, angles, powers, and rows to leave out."""

from __future__ import annotations

import pytest


# the module under test needs both, and so do its tests
pytest.importorskip("array_api_compat")
strict = pytest.importorskip("array_api_strict")

import numpy as np
from _array_toys import SECOND, needs_jax
from pipekit_array.arrays import (
    divide_or_nan,
    safe_atan2,
    safe_power,
    safe_sqrt,
    squared_magnitude,
    stand_in,
)


# --- quotients ---


def test_divide_or_nan() -> None:
    num = np.asarray([1.0, 0.0, 2.0, np.nan], dtype=np.float32)
    den = np.asarray([2.0, 0.0, -4.0, 1.0], dtype=np.float32)
    with np.errstate(all="raise"):
        out = divide_or_nan(num, den, np)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, [0.5, np.nan, -0.5, np.nan])


@needs_jax
def test_divide_or_nan_has_a_finite_gradient() -> None:
    import jax
    import jax.numpy as jnp

    den = jnp.asarray([2.0, 0.0])

    def loss(num, den):
        out = divide_or_nan(num, den, jnp)
        return jnp.where(den == 0, 0.0, out).sum()

    for gradient in jax.grad(loss, (0, 1))(jnp.ones(2), den):
        assert np.isfinite(gradient).all()


# --- roots, angles and powers ---

# 4 is ordinary, 0 singular, -1 a rounded difference of squares, NaN missing
POINTS = np.asarray([4.0, 0.0, -1.0, np.nan])
ROOTS = [
    (safe_sqrt, [2.0, 0.0, 0.0, np.nan], [0.25, 0.0, 0.0, np.nan]),
    (
        lambda x, xp: safe_power(x, 1.5, xp),
        [8.0, 0.0, 0.0, np.nan],
        [3.0, 0.0, 0.0, np.nan],
    ),
    (
        lambda x, xp: safe_power(x, -0.5, xp),
        [0.5, 0.0, 0.0, np.nan],
        [-0.0625, 0.0, 0.0, np.nan],
    ),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("fn", "values", "gradient"), ROOTS)
def test_a_safe_root_at_each_kind_of_point(fn, values, gradient, dtype) -> None:
    with np.errstate(all="raise"):
        out = fn(POINTS.astype(dtype), np)
    assert out.dtype == dtype
    np.testing.assert_allclose(out, values, rtol=1e-6)
    out = fn(strict.asarray(POINTS.astype(dtype), device=SECOND), strict)
    assert out.device == SECOND
    out = np.asarray(out.to_device(strict.Device("CPU_DEVICE")))
    assert out.dtype == dtype
    np.testing.assert_allclose(out, values, rtol=1e-6)


@needs_jax
@pytest.mark.parametrize(("fn", "values", "gradient"), ROOTS)
def test_the_gradient_of_a_safe_root(fn, values, gradient) -> None:
    import jax
    import jax.numpy as jnp

    derivative = jax.grad(lambda x: fn(x, jnp))
    out = jax.jit(jax.vmap(derivative))(jnp.asarray(POINTS))
    np.testing.assert_allclose(out, gradient, rtol=1e-6)


@needs_jax
def test_the_gradient_of_a_safe_power_by_its_exponent() -> None:
    import jax
    import jax.numpy as jnp

    out = jax.jacfwd(lambda p: safe_power(jnp.asarray(POINTS[:3]), p, jnp))(2.0)
    np.testing.assert_allclose(out, [16.0 * np.log(4.0), 0.0, 0.0], rtol=1e-6)


def test_safe_atan2() -> None:
    y, x = [0.0, 1.0, 0.0, -1.0, np.nan], [0.0, 0.0, -1.0, -1.0, 1.0]
    want = np.arctan2(y, x)
    np.testing.assert_array_equal(safe_atan2(np.asarray(y), np.asarray(x), np), want)
    out = safe_atan2(strict.asarray(y), strict.asarray(x), strict)
    np.testing.assert_array_equal(np.asarray(out), want)


@needs_jax
def test_the_gradient_of_safe_atan2() -> None:
    import jax
    import jax.numpy as jnp

    derivative = jax.grad(lambda y, x: safe_atan2(y, x, jnp), (0, 1))
    y, x = jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 2.0])
    by_y, by_x = jax.jit(jax.vmap(derivative))(y, x)
    np.testing.assert_allclose(by_y, [0.0, 0.4], rtol=1e-6)
    np.testing.assert_allclose(by_x, [0.0, -0.2], rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_squared_magnitude_is_real(dtype) -> None:
    for xp in (np, strict):
        z = xp.asarray([3.0, 0.0], dtype=getattr(xp, dtype))
        out = np.asarray(squared_magnitude(z, xp))
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, [9.0, 0.0])
    assert squared_magnitude(np.asarray(3.0 + 4.0j), np) == 25.0


@needs_jax
def test_the_gradient_of_squared_magnitude_at_zero() -> None:
    import jax
    import jax.numpy as jnp

    def power(x):
        return squared_magnitude(x * (1.0 + 1.0j), jnp)

    np.testing.assert_allclose(jax.grad(power)(0.0), 0.0)
    np.testing.assert_allclose(jax.grad(power)(3.0), 12.0, rtol=1e-6)


# --- rows to leave out ---


@pytest.mark.parametrize("xp", [np, strict])
@pytest.mark.parametrize(
    ("mask", "reference", "want"),
    [
        ([False, True, True], None, [[3, 4], [3, 4], [5, 6]]),
        ([False, True, True], [0.0, 9.0], [[0, 9], [3, 4], [5, 6]]),
        ([False, False, False], None, [[1, 2], [1, 2], [1, 2]]),
    ],
)
def test_stand_in(xp, mask, reference, want) -> None:
    x = xp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=xp.float32)
    out = np.asarray(stand_in(x, xp.asarray(mask), xp, reference=reference))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, want)


@needs_jax
def test_stand_in_keeps_the_gradient_of_the_other_rows_finite() -> None:
    import jax
    import jax.numpy as jnp

    mask = jnp.asarray([True, False])

    def loss(x, replace):
        rows = stand_in(x, mask, jnp) if replace else x
        return jnp.where(mask, jnp.log(rows).sum(axis=1), 0.0).sum()

    x = jnp.asarray([[1.0, 2.0], [-1.0, 2.0]])
    gradient = jax.jit(jax.grad(loss), static_argnums=1)
    np.testing.assert_allclose(gradient(x, True), [[1.0, 0.5], [0.0, 0.0]])
    spoilt = jnp.where(mask[:, None], x, jnp.nan)
    assert np.isnan(jax.grad(loss)(spoilt, False)).any()
    np.testing.assert_allclose(gradient(spoilt, True)[0], [1.0, 0.5])
