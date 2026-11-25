"""`pipekit_array.arrays`: values of the host next to an array."""

from __future__ import annotations

import pytest


# the module under test needs both, and so do its tests
pytest.importorskip("array_api_compat")
strict = pytest.importorskip("array_api_strict")

import numpy as np
from _array_toys import SECOND, needs_jax
from pipekit_array.arrays import (
    boolean_like,
    constants_like,
    table_like,
)


# --- tables and masks ---


@pytest.mark.parametrize("device", [None, SECOND])
def test_table_like_keeps_its_dtype_on_the_device(device) -> None:
    ref = strict.zeros(2, dtype=strict.float32, device=device)
    out = table_like(np.arange(3), ref, strict)
    assert out.dtype == strict.int64
    assert out.device == ref.device


def test_boolean_like_takes_the_host_and_its_own_library() -> None:
    ref = strict.zeros(2, device=SECOND)
    for value in (True, [True, False], np.asarray([True, False]), np.True_):
        out = boolean_like(value, ref, strict)
        assert out.dtype == strict.bool
        assert out.device == SECOND
    own = strict.asarray([True, False], device=SECOND)
    assert boolean_like(own, ref, strict) is own


@pytest.mark.parametrize("value", [1, [1.0], np.ones(2), strict.ones(2)])
def test_boolean_like_refuses_what_is_not_boolean(value) -> None:
    with pytest.raises(ValueError, match="'wet' must be boolean"):
        boolean_like(value, strict.zeros(2), strict, name="wet")


# --- tables of numbers ---


@pytest.mark.parametrize("table", [[0.5, 1.5], (0.5, 1.5), np.asarray([0.5, 1.5]), 0.5])
@pytest.mark.parametrize("device", [None, SECOND])
def test_constants_like_takes_the_dtype_and_the_device(table, device) -> None:
    ref = strict.zeros((4, 2), dtype=strict.float32, device=device)
    out = constants_like(table, ref, strict)
    assert (out.dtype, out.device) == (strict.float32, ref.device)
    assert out.shape == np.shape(table)


@pytest.mark.parametrize("table", [[0.5], strict.asarray([0.5])])
def test_constants_like_leaves_a_table_next_to_integers_alone(table) -> None:
    out = constants_like(table, strict.zeros(2, dtype=strict.int64), strict)
    assert out.dtype == strict.float64


@needs_jax
def test_constants_like_keeps_the_gradient_of_a_traced_table() -> None:
    import jax
    import jax.numpy as jnp

    def loss(spacing, field):
        return (field / constants_like(spacing, field, jnp)).sum()

    spacing = jnp.asarray([2.0, 4.0])
    out = jax.jit(jax.grad(loss))(spacing, jnp.ones(2, dtype=jnp.float32))
    np.testing.assert_allclose(out, [-0.25, -0.0625], rtol=1e-6)
