"""Tests for the `array_namespace` dispatch shim."""

from __future__ import annotations

import pytest
from pipekit_array._namespace import array_namespace


def test_array_namespace_single_input(backend, make_array):
    x = make_array(backend, shape=(4,), seed=0)
    xp = array_namespace(x)
    # numpy → 'numpy' or 'array_api_compat.numpy';
    # jax   → 'jax.numpy' or 'array_api_compat.numpy.jax';
    # torch → 'array_api_compat.torch'
    name = xp.__name__
    assert backend in name or "numpy" in name


def test_array_namespace_multi_input_same_backend(backend, make_array):
    x = make_array(backend, shape=(4,), seed=0)
    y = make_array(backend, shape=(4,), seed=1)
    xp = array_namespace(x, y)
    assert xp is not None


def test_array_namespace_no_array_input_raises():
    with pytest.raises(TypeError, match="No input implements the Array API"):
        array_namespace("not an array")


def test_array_namespace_mixed_backends_raises():
    np = pytest.importorskip("numpy")
    jnp = pytest.importorskip("jax.numpy")
    x_np = np.array([1.0, 2.0])
    x_jax = jnp.array([1.0, 2.0])
    with pytest.raises(TypeError, match="multiple Array API backends"):
        array_namespace(x_np, x_jax)


def test_array_namespace_non_array_among_arrays_raises(backend, make_array):
    """A non-array input alongside a real array must raise, not be ignored.

    Regression: the non-compat fallback used to silently skip inputs
    without `__array_namespace__`, masking type bugs at the boundary.
    """
    x = make_array(backend, shape=(4,), seed=0)
    with pytest.raises(TypeError):
        array_namespace(x, "not an array")
