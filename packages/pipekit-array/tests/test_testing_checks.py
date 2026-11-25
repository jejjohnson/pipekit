"""Each check of `pipekit_array.testing` passes on a right function and fails on
a wrong one."""

from __future__ import annotations

import importlib.util
import itertools
import subprocess
import sys
from typing import Any

import pytest


pytest.importorskip("array_api_compat")
strict = pytest.importorskip("array_api_strict")

import numpy as np
from pipekit_array.functions import elementwise, floating_inputs, like
from pipekit_array.testing import (
    array_kinds,
    assert_adjoint,
    assert_backends_agree,
    assert_finite_gradient,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_mean,
    assert_refuses_dataarrays,
    assert_traces,
    assert_uniform,
    assert_variance,
    ks_critical_value,
    ks_statistic,
)


def _missing(library: str) -> bool:
    return importlib.util.find_spec(library) is None


needs_jax = pytest.mark.skipif(_missing("jax"), reason="JAX is not installed")
needs_xarray = pytest.mark.skipif(_missing("xarray"), reason="xarray is not installed")
SLOPE = 0.3
X = np.linspace(0.1, 2.0, 6).reshape(2, 3)
VALID = np.asarray([[True, False, True], [True, True, False]])


@pytest.fixture(autouse=True)
def x64():
    """JAX in its 64-bit mode, if it is installed."""
    if _missing("jax"):
        yield
        return
    import jax

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@elementwise(n_outputs=2)
def toy(values, offset=0.0) -> Any:
    """sin(0.3 x) + o, and x²: an elementwise function."""
    xp, (values, offset) = floating_inputs(values, offset)
    return xp.sin(like(SLOPE, values, xp) * values) + offset, values**2


def toy_mean(values, valid) -> Any:
    """The mean of the valid samples: a function along an axis."""
    xp, (values,) = floating_inputs(values, params=(valid,))
    kept = xp.where(valid, values, xp.zeros_like(values))
    return xp.sum(kept, axis=-1) / xp.sum(xp.astype(valid, values.dtype), axis=-1)


def test_the_module_imports_neither_pytest_nor_another_library() -> None:
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from pipekit_array.testing import assert_matches_oracle, check_adapter\n"
        "assert_matches_oracle(np.ones(2), np.ones(2), rtol=0.0)\n"
        "names = {'pytest', 'jax', 'torch', 'dask', 'xarray', 'array_api_compat'}\n"
        "assert not names & set(sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_the_checks_run_on_numpy_alone() -> None:
    code = (
        "import sys\n"
        "for name in ('jax', 'torch', 'dask', 'xarray', 'array_api_strict'):\n"
        "    sys.modules[name] = None\n"
        "import importlib.util\n"
        "importlib.util.find_spec = lambda name: None\n"
        "import unittest\n"
        "import numpy as np\n"
        "import pipekit_array.testing as t\n"
        "assert [kind.name for kind in t.array_kinds()] == ['numpy']\n"
        "t.assert_backends_agree(np.sqrt, np.ones(2), labelled=True)\n"
        "for check in (\n"
        "    lambda: t.assert_traces(np.sqrt, np.ones(2)),\n"
        "    lambda: t.assert_grad_matches_fd(np.sqrt, np.ones(2)),\n"
        "    lambda: t.assert_finite_gradient(np.sqrt, np.ones(2)),\n"
        "    lambda: t.assert_refuses_dataarrays(np.sqrt, np.ones(2)),\n"
        "):\n"
        "    try:\n"
        "        check()\n"
        "    except unittest.SkipTest:\n"
        "        continue\n"
        "    raise SystemExit('a check did not skip')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --- right functions pass ---


def test_every_check_on_a_right_function() -> None:
    assert_matches_oracle(toy(X)[0], np.sin(SLOPE * X), rtol=1e-15)
    assert_backends_agree(toy, X, np.asarray([1.0, 2.0, 3.0]), labelled=True)
    assert_backends_agree(toy_mean, X, VALID)


@needs_jax
def test_every_check_of_jax_on_a_right_function() -> None:
    assert_traces(toy, X, np.asarray([1.0, 2.0]))
    assert_traces(toy_mean, X, VALID)
    assert_traces(lambda tree: {"out": tree["x"] * 2.0}, {"x": X})
    assert_grad_matches_fd(lambda x: toy(x)[0], X)
    assert_grad_matches_fd(lambda tree: tree["x"] * tree["y"], {"x": X, "y": X})
    assert_finite_gradient(lambda x: toy_mean(x, VALID), X)


@needs_xarray
def test_refusing_dataarrays() -> None:
    assert_refuses_dataarrays(toy_mean, X, VALID)
    with pytest.raises(AssertionError, match="argument 0 accepted"):
        assert_refuses_dataarrays(lambda x: x, X)


# --- the oracle and the adjoint ---


def test_an_oracle_names_the_worst_index() -> None:
    with pytest.raises(AssertionError, match=r"worst at index \(1,\)"):
        assert_matches_oracle([1.0, 2.5], [1.0, 2.0], rtol=1e-12)


def test_an_oracle_matches_nan_and_infinity_only_with_themselves() -> None:
    assert_matches_oracle([np.nan, np.inf], [np.nan, np.inf], rtol=0.0)
    bad = [([np.nan], [1.0]), ([1.0], [np.nan]), ([np.inf], [1.0]), ([1.0], [np.inf])]
    for actual, expected in bad:
        for rtol in (0.0, 1.0):
            with pytest.raises(AssertionError):
                assert_matches_oracle(actual, expected, rtol=rtol)


def test_an_oracle_masks_and_compares_integers_exactly() -> None:
    assert_matches_oracle([1.0, 9.0], [1.0, 2.0], rtol=0.0, where=[True, False])
    with pytest.raises(AssertionError):
        assert_matches_oracle([1], [2], rtol=10.0)
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_matches_oracle([1.0], [[1.0]], rtol=0.0)


def test_the_adjoint_of_the_wrong_matrix() -> None:
    h = np.random.default_rng(0).normal(size=(3, 4))
    assert_adjoint(lambda x: h @ x, lambda r: h.T @ r, 4, 3)
    with pytest.raises(AssertionError, match="dot-product"):
        assert_adjoint(lambda x: h @ x, lambda r: 1.1 * h.T @ r, 4, 3)
    with pytest.raises(AssertionError, match="r has shape"):
        assert_adjoint(lambda x: h @ x, lambda r: r, 4, 3)
    with pytest.raises(AssertionError, match="H x has shape"):
        assert_adjoint(lambda x: x, lambda r: h.T @ r, 4, 3)


# --- the array libraries ---


def test_the_kinds_have_unique_names_and_numpy_comes_first() -> None:
    names = [kind.name for kind in array_kinds()]
    assert len(set(names)) == len(names)
    assert names[:3] == ["numpy", "strict", "second device"]


@pytest.mark.parametrize("kind", array_kinds(), ids=lambda kind: kind.name)
def test_a_kind_reads_what_it_makes(kind) -> None:
    np.testing.assert_array_equal(kind.read(kind.make(X)), X)
    with pytest.raises(AssertionError, match=f"{kind.name} in must give"):
        kind.read("text")


def test_a_function_that_drops_the_kind() -> None:
    with pytest.raises(AssertionError, match="strict float64: strict in must give"):
        assert_backends_agree(lambda x: np.sin(np.asarray(x)), X, kinds=["strict"])


def test_a_function_that_promotes_float32() -> None:
    def promotes(x: Any) -> Any:
        xp, (x,) = floating_inputs(x)
        return x * xp.asarray(SLOPE, dtype=xp.float64)

    with pytest.raises(AssertionError, match="numpy float32: float64 came back"):
        assert_backends_agree(promotes, X)


def test_a_function_that_leaves_the_device() -> None:
    def misplaces(x: Any) -> Any:
        xp, (x,) = floating_inputs(x)
        return xp.zeros(x.shape, dtype=x.dtype)

    with pytest.raises(AssertionError, match="second device float64: the result is on"):
        assert_backends_agree(misplaces, X)


def test_a_function_that_is_wrong_on_one_library_names_it_and_the_index() -> None:
    def wrong(x: Any) -> Any:
        xp, (x,) = floating_inputs(x)
        return x + (0.0 if xp.__name__.endswith("numpy") else 1.0)

    with pytest.raises(AssertionError, match=r"strict float64: .* index \(0, 0\)"):
        assert_backends_agree(wrong, X)
    assert_backends_agree(wrong, X, kinds=["numpy", "jax"])


@needs_jax
def test_a_function_that_cannot_be_compiled() -> None:
    import jax

    def branches(x: Any) -> Any:
        xp, (x,) = floating_inputs(x)
        return x if xp.any(x > 0.0) else -x

    with pytest.raises(jax.errors.TracerBoolConversionError):
        assert_backends_agree(branches, X, kinds=["jax"])


def test_a_result_that_crosses_zero_needs_an_absolute_tolerance() -> None:
    def rounds(x: Any) -> Any:
        xp, (x,) = floating_inputs(x)
        return xp.sin(x) + (0.0 if xp.__name__.endswith("numpy") else 1e-14)

    zero = np.zeros(2)
    with pytest.raises(AssertionError, match="strict float64"):
        assert_backends_agree(rounds, zero, kinds=["strict"])
    assert_backends_agree(rounds, zero, atol=1e-13, atol32=1e-13, kinds=["strict"])


def test_counts_must_match_exactly() -> None:
    def count(valid: Any) -> Any:
        xp, _ = floating_inputs(0.0, params=(valid,))
        return xp.sum(xp.astype(valid, xp.int32), axis=-1)

    assert_backends_agree(count, VALID)


def test_a_result_that_is_missing_on_one_library() -> None:
    def fewer(x: Any) -> Any:
        xp, (x,) = floating_inputs(x)
        return (x, x) if xp.__name__.endswith("numpy") else x

    with pytest.raises(AssertionError, match="strict float64: 1 results, 2 on NumPy"):
        assert_backends_agree(fewer, X, kinds=["strict"])


def test_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="no such kind of array: numpi"):
        assert_backends_agree(toy, X, kinds=["numpi"])


@needs_xarray
def test_dataarrays_are_checked_only_on_request() -> None:
    def refuses(x: Any) -> Any:
        return floating_inputs(x)[1][0]

    assert_backends_agree(refuses, X)
    for options in ({"labelled": True}, {"kinds": ["xarray"]}):
        with pytest.raises(Exception, match="DataArray"):
            assert_backends_agree(refuses, X, **options)


@needs_xarray
def test_dataarrays_broadcast_as_numpy_does() -> None:
    kind = next(kind for kind in array_kinds() if kind.name == "xarray")
    out = kind.make(np.ones((2, 3))) * kind.make(np.arange(3.0))
    assert out.shape == (2, 3)


# --- JAX's transforms ---


@needs_jax
def test_a_gradient_that_is_cut() -> None:
    import jax

    with pytest.raises(AssertionError, match="out of tolerance"):
        assert_grad_matches_fd(lambda x: jax.lax.stop_gradient(x) * x, X)


@needs_jax
def test_a_gradient_that_is_not_finite() -> None:
    import jax.numpy as jnp

    with pytest.raises(AssertionError, match="of output 0 by input 0"):
        assert_grad_matches_fd(lambda x: jnp.sqrt(jnp.abs(x)), np.zeros(2))
    # the derivative and the difference are NaN together
    with pytest.raises(AssertionError, match="not finite"):
        assert_grad_matches_fd(lambda x: x * jnp.nan, np.ones(2))


@needs_jax
def test_the_gradient_of_a_single_where() -> None:
    import jax.numpy as jnp

    values = np.asarray([0.0, 2.0])

    def single(x: Any) -> Any:
        return jnp.sum(jnp.where(x > 0.0, 1.0 / x, 0.0))

    def double(x: Any) -> Any:
        safe = jnp.where(x > 0.0, x, 1.0)
        return jnp.sum(jnp.where(x > 0.0, 1.0 / safe, 0.0))

    assert_finite_gradient(double, values)
    assert_finite_gradient(single, values, where=values > 0.0)
    with pytest.raises(AssertionError, match="of output 0 by input 0"):
        assert_finite_gradient(single, values)


@needs_jax
def test_an_impure_function_under_a_transform() -> None:
    calls = itertools.count()
    with pytest.raises(AssertionError, match=r"jit\(fn\), output 0"):
        assert_traces(lambda x: x + next(calls), X)


@needs_jax
@pytest.mark.parametrize("other", [np.ones(5), np.asarray(1.0)])
def test_a_transform_needs_one_batch_axis(other) -> None:
    with pytest.raises(AssertionError, match="same leading batch axis"):
        assert_traces(toy, X, other)


@needs_jax
def test_a_check_of_jax_needs_the_64_bit_mode() -> None:
    import jax

    jax.config.update("jax_enable_x64", False)
    for check in (
        lambda: assert_traces(toy, X),
        lambda: assert_grad_matches_fd(toy, X),
        lambda: assert_backends_agree(toy, X, kinds=["jax"]),
    ):
        with pytest.raises(AssertionError, match="needs float64"):
            check()


# --- samples ---


def test_samples_of_the_right_law() -> None:
    rng = np.random.default_rng(0)
    assert_mean(rng.normal(2.0, 1.0, size=1000), 2.0)
    assert_variance(rng.normal(0.0, 2.0, size=1000), 4.0)
    assert_uniform(rng.uniform(2.0, 5.0, size=1000), 2.0, 5.0)


def test_samples_of_another_law() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError, match="standard errors"):
        assert_mean(rng.normal(2.0, 1.0, size=1000), 2.2)
    with pytest.raises(AssertionError, match="standard errors"):
        assert_variance(rng.normal(0.0, 2.0, size=1000), 8.0)
    with pytest.raises(AssertionError, match="KS statistic"):
        assert_uniform(rng.uniform(0.0, 1.0, size=1000) ** 2, 0.0, 1.0)
    with pytest.raises(AssertionError, match="outside"):
        assert_uniform(rng.uniform(0.0, 2.0, size=1000), 0.0, 1.0)


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (np.zeros(5), "too few"),
        (np.zeros(50), "no spread"),
        (np.append(np.arange(50.0), np.nan), "not finite"),
    ],
)
def test_a_sample_that_cannot_be_bounded(sample, message) -> None:
    with pytest.raises(AssertionError, match=message):
        assert_mean(sample, 0.0)


def test_the_kolmogorov_smirnov_statistic() -> None:
    # the quantiles of the uniform law are half a step from its ends
    sample = (np.arange(50) + 0.5) / 50
    np.testing.assert_allclose(ks_statistic(sample, 0.0, 1.0), 0.01)
    np.testing.assert_allclose(ks_critical_value(100, 1e-3), 0.195, atol=5e-4)
