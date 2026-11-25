"""`pipekit_array.hooks`: indexed updates, loops and padding on any array library."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import Any, NamedTuple

import pytest


pytest.importorskip("array_api_compat")
strict = pytest.importorskip("array_api_strict")

import numpy as np
from pipekit_array.hooks import (
    add_at,
    multiply_at,
    optimization_barrier,
    pad,
    scan,
    set_at,
    stop_gradient,
    vmap,
)
from pipekit_ops import BackendError


needs_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="JAX is not installed"
)
needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)
SECOND = strict.Device("device1")
X = np.arange(12.0).reshape(3, 4)


def host(array: Any) -> np.ndarray:
    """An array as NumPy, from any device."""
    if getattr(array, "device", None) == SECOND:
        array = array.to_device(strict.Device("CPU_DEVICE"))
    return np.asarray(array)


class Pair(NamedTuple):
    left: Any
    right: Any


@pytest.mark.parametrize("library", ["numpy", "array_api_compat"])
def test_the_import_error_names_the_extras(library: str) -> None:
    code = f"import sys; sys.modules[{library!r}] = None\nimport pipekit_array.hooks\n"
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "pipekit-array[numpy,compat]" in result.stderr


def test_numpy_data_imports_neither_jax_nor_torch() -> None:
    code = (
        "import sys, numpy as np\n"
        "from pipekit_array.hooks import add_at, pad, scan, stop_gradient, vmap\n"
        "x = np.arange(3.0)\n"
        "add_at(x, 0, 1.0), pad(x, ((1, 1),)), stop_gradient(x)\n"
        "vmap(np.sin)(x), scan(lambda c, v: (c + v, c), 0.0, x)\n"
        "assert not {'jax', 'torch', 'dask', 'xarray'} & set(sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --- indexed updates ---

UPDATES = {"set": set_at, "add": add_at, "multiply": multiply_at}
MASK = X % 3 == 0
# indices that every library takes in an assignment
BASIC = {
    "integer": (1, ...),
    "negative": (-1, ...),
    "slice": (slice(1, 3), ...),
    "step": (slice(None, None, 2), ...),
    "ellipsis": ...,
    "column": (..., 2),
    "block": (slice(0, 2), slice(1, 3)),
    "mask": MASK,
}
# an integer array, of distinct entries
ADVANCED = {
    "rows": np.asarray([0, 2]),
    "entries": (np.asarray([0, 2]), np.asarray([1, 3])),
    "mixed": (slice(None), np.asarray([3, 0])),
}


def expected(name: str, index: Any, values: Any) -> np.ndarray:
    """The update by plain NumPy; the index names no entry twice."""
    y = X.copy()
    if name == "set":
        y[index] = values
    elif name == "add":
        y[index] += values
    else:
        y[index] *= values
    return y


@pytest.mark.parametrize("name", UPDATES)
@pytest.mark.parametrize("index", BASIC.values(), ids=BASIC)
@pytest.mark.parametrize("device", [None, SECOND])
def test_a_basic_update_on_numpy_and_the_strict_library(name, index, device) -> None:
    want = expected(name, index, 2.5)
    x = X.astype(np.float32)
    out = UPDATES[name](x, index, 2.5)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, want)
    np.testing.assert_array_equal(x, X)

    x = strict.asarray(X, device=device)
    out = UPDATES[name](x, index, np.asarray(2.5))
    assert out.device == x.device
    np.testing.assert_array_equal(host(out), want)
    np.testing.assert_array_equal(host(x), X)


@needs_jax
@pytest.mark.parametrize("name", UPDATES)
@pytest.mark.parametrize("index", [*BASIC.values(), *ADVANCED.values()])
def test_an_update_on_jax_is_the_at_method(name, index) -> None:
    import jax
    import jax.numpy as jnp

    x = jnp.asarray(X)
    np.testing.assert_allclose(
        UPDATES[name](x, index, 2.5), expected(name, index, 2.5), rtol=1e-6
    )
    ours = jax.make_jaxpr(lambda x, v: UPDATES[name](x, index, v))(x, 2.5)
    theirs = jax.make_jaxpr(lambda x, v: getattr(x.at[index], name)(v))(x, 2.5)
    assert str(ours) == str(theirs)


@pytest.mark.parametrize("name", UPDATES)
@pytest.mark.parametrize("index", ADVANCED.values(), ids=ADVANCED)
def test_an_integer_array_on_numpy(name, index) -> None:
    np.testing.assert_array_equal(
        UPDATES[name](X, index, 2.5), expected(name, index, 2.5)
    )


@pytest.mark.parametrize("name", UPDATES)
def test_an_integer_array_is_refused_by_name_on_the_strict_library(name) -> None:
    with pytest.raises(BackendError, match=f"{name}_at .* array_api_strict"):
        UPDATES[name](strict.asarray(X), np.asarray([0, 2]), 1.0)


@pytest.mark.parametrize("xp", [np, pytest.param("jax", marks=needs_jax)])
def test_a_repeated_index_counts_each_time(xp) -> None:
    if xp == "jax":
        import jax.numpy as xp
    index = xp.asarray([0, 2, 2])
    out = add_at(xp.zeros(4), index, xp.asarray([1.0, 10.0, 100.0]))
    np.testing.assert_array_equal(out, [1.0, 0.0, 110.0, 0.0])
    out = multiply_at(xp.ones(4), index, xp.asarray([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(out, [2.0, 1.0, 12.0, 1.0])


@pytest.mark.parametrize("update", UPDATES.values())
def test_an_index_out_of_bounds(update) -> None:
    for x in (np.zeros(3), strict.zeros(3)):
        with pytest.raises(IndexError):
            update(x, 7, 1.0)


@needs_jax
def test_jax_does_not_report_an_index_out_of_bounds() -> None:
    import jax.numpy as jnp

    np.testing.assert_array_equal(set_at(jnp.zeros(3), 7, 1.0), np.zeros(3))


@needs_jax
@pytest.mark.parametrize("update", UPDATES.values())
def test_numpy_joins_jax_in_an_update(update) -> None:
    import jax
    import jax.numpy as jnp

    assert isinstance(update(np.ones(3), 0, jnp.asarray(2.0)), jax.Array)
    assert isinstance(update(jnp.ones(3), 0, np.asarray(2.0)), jax.Array)
    traced = jax.jit(lambda k: update(np.ones(3), k, 2.0))(jnp.asarray(1))
    np.testing.assert_array_equal(traced, update(np.ones(3), 1, 2.0))


@needs_jax
def test_a_gradient_flows_to_the_array_and_the_values() -> None:
    import jax
    import jax.numpy as jnp

    def loss(x, v):
        return (set_at(x, 1, v) ** 2).sum()

    by_x, by_v = jax.grad(loss, (0, 1))(jnp.asarray([1.0, 2.0]), 3.0)
    np.testing.assert_allclose(by_x, [2.0, 0.0])
    np.testing.assert_allclose(by_v, 6.0)


@needs_torch
@pytest.mark.parametrize("update", UPDATES.values())
def test_a_gradient_flows_on_torch(update) -> None:
    import torch

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    v = torch.tensor(4.0, requires_grad=True)
    (update(x, slice(1, 2), v) ** 2).sum().backward()
    assert x.grad[0] == 2.0
    assert v.grad != 0.0
    np.testing.assert_array_equal(x.detach(), [1.0, 2.0, 3.0])
    with pytest.raises(BackendError, match=r"add_at .* torch"):
        add_at(x, torch.tensor([0, 0]), v)


def test_dask_stays_lazy() -> None:
    da = pytest.importorskip("dask.array")
    x = da.from_array(X, chunks=2)
    for name, update in UPDATES.items():
        out = update(x, (slice(0, 2), slice(1, 3)), 2.5)
        assert isinstance(out, da.Array)
        np.testing.assert_array_equal(out, expected(name, np.s_[0:2, 1:3], 2.5))


def test_a_labelled_array_is_refused() -> None:
    xr = pytest.importorskip("xarray")
    with pytest.raises(BackendError, match="DataArray"):
        set_at(xr.DataArray(X), 0, 1.0)


# --- vmap ---


@pytest.mark.parametrize("xp", [np, strict, pytest.param("jax", marks=needs_jax)])
def test_vmap_over_some_arguments(xp) -> None:
    if xp == "jax":
        import jax.numpy as xp

    def row(a, b, scale):
        return xp.sum(a * b) * scale, a

    a, b = xp.asarray(X), xp.asarray(X.T)
    inner, first = vmap(row, in_axes=(0, 1, None))(a, b, 2.0)
    np.testing.assert_allclose(np.asarray(inner), 2.0 * (X * X).sum(axis=1))
    np.testing.assert_array_equal(np.asarray(first), X)


def test_vmap_of_nested_arrays() -> None:
    def fn(pair):
        return {"sum": pair.left + pair.right[0], "none": None}

    out = vmap(fn)(Pair(X, [X]))
    np.testing.assert_array_equal(out["sum"], 2 * X)
    assert out["none"] is None


@needs_jax
def test_vmap_on_jax_is_jax_vmap() -> None:
    import jax
    import jax.numpy as jnp

    def fn(a, b):
        return jnp.outer(a, b), a.sum()

    args = jnp.asarray(X), jnp.arange(2.0)
    ours = jax.make_jaxpr(vmap(fn, in_axes=(1, None)))(*args)
    theirs = jax.make_jaxpr(jax.vmap(fn, in_axes=(1, None)))(*args)
    assert str(ours) == str(theirs)


@pytest.mark.parametrize(
    ("args", "in_axes", "match"),
    [
        ((np.ones(2), np.ones(3)), 0, r"one length .* \[2, 3\]"),
        ((np.ones(2),), None, "one length .* none"),
        ((np.ones((0, 2)),), 0, "no steps"),
        ((np.asarray(1.0),), 0, "axis"),
    ],
)
def test_vmap_refuses(args, in_axes, match) -> None:
    with pytest.raises(ValueError, match=match):
        vmap(lambda *xs: xs[0], in_axes=in_axes)(*args)


# --- scan ---


def both(f, init, xs=None, **options) -> None:
    """Check `scan` on NumPy against ``jax.lax.scan``."""
    import jax

    arrays = jax.tree.map(jax.numpy.asarray, (init, xs))
    got = scan(f, init, xs, **options)
    want = jax.lax.scan(f, *arrays, **options)
    assert jax.tree.structure(got) == jax.tree.structure(want)
    assert not any(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(got))
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6), got, want)


def fibonacci(carry, _):
    a, b = carry
    return (b, a + b), b


@needs_jax
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_agrees_with_lax_scan(reverse) -> None:
    both(lambda c, x: (c + x, 2 * c), np.zeros(4), X, reverse=reverse)


@needs_jax
def test_scan_with_a_length_and_a_tuple_carry() -> None:
    both(fibonacci, (np.asarray(0), np.asarray(1)), None, length=7)


@needs_jax
def test_scan_with_nested_carries_inputs_and_outputs() -> None:
    def step(carry, x):
        total = carry["total"] + x.left * x.right[0]
        return {"total": total, "count": carry["count"] + 1}, Pair(total, None)

    init = {"total": np.zeros(4), "count": np.asarray(0)}
    both(step, init, Pair(X, [np.arange(3.0)]))


@needs_jax
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_on_jax_is_lax_scan(reverse) -> None:
    import jax
    import jax.numpy as jnp

    def step(c, x):
        return c + x, c * x

    args = jnp.zeros(4), jnp.asarray(X)
    ours = jax.make_jaxpr(lambda c, x: scan(step, c, x, reverse=reverse))(*args)
    theirs = jax.make_jaxpr(lambda c, x: jax.lax.scan(step, c, x, reverse=reverse))(
        *args
    )
    assert str(ours) == str(theirs)


@needs_jax
def test_scan_joins_a_numpy_table_to_a_jax_carry() -> None:
    import jax
    import jax.numpy as jnp

    total, _ = scan(lambda c, x: (c + x, None), jnp.zeros(4), X)
    assert isinstance(total, jax.Array)


def test_scan_takes_any_carry_off_jax() -> None:
    carry, out = scan(lambda c, x: (f"{c}{int(x)}", None), "", np.arange(3.0))
    assert (carry, out) == ("012", None)


def test_scan_on_the_strict_library() -> None:
    x = strict.asarray(X, device=SECOND)
    total, partial = scan(lambda c, v: (c + v, c), x[0, ...], x)
    assert partial.device == total.device == SECOND
    np.testing.assert_array_equal(host(total), X[0] + X.sum(axis=0))


@pytest.mark.parametrize(
    ("xs", "length", "match"),
    [
        (None, None, "xs or a length"),
        (np.ones((0, 2)), None, "no steps"),
        (None, 0, "no steps"),
        (np.ones(3), 4, r"\[3, 4\]"),
        ((np.ones(3), np.ones(2)), None, r"\[2, 3\]"),
        (np.asarray(1.0), None, "axis"),
    ],
)
def test_scan_refuses(xs, length, match) -> None:
    with pytest.raises(ValueError, match=match):
        scan(lambda c, x: (c, x), 0.0, xs, length)


@needs_torch
def test_a_gradient_flows_through_scan_and_vmap_on_torch() -> None:
    import torch

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    _, out = scan(lambda c, v: (c + v, c * v), torch.tensor(0.0), x)
    (out.sum() + vmap(lambda v: v * v)(x).sum()).backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad != 0).all()


# --- padding, and identities that steer JAX ---


@pytest.mark.parametrize("device", [None, SECOND])
@pytest.mark.parametrize("dtype", ["float32", "int64", "bool"])
def test_pad(device, dtype) -> None:
    widths = ((1, 0), (2, 3))
    x = X.astype(dtype)
    want = np.pad(x, widths)
    np.testing.assert_array_equal(pad(x, widths), want)
    out = pad(strict.asarray(x, device=device), widths)
    assert out.dtype == getattr(strict, dtype)
    np.testing.assert_array_equal(host(out), want)


@needs_torch
def test_pad_on_torch() -> None:
    import torch

    x = torch.ones((2, 2), dtype=torch.float32, requires_grad=True)
    out = pad(x, ((1, 0), (0, 2)))
    assert (tuple(out.shape), out.dtype) == ((3, 4), torch.float32)
    out.sum().backward()
    np.testing.assert_array_equal(x.grad, np.ones((2, 2)))


@pytest.mark.parametrize("widths", [(), ((1, 1),), ((1, 1),) * 3])
def test_pad_needs_a_pair_per_axis(widths) -> None:
    for x in (X, strict.asarray(X)):
        with pytest.raises(ValueError, match="one pair of widths per axis"):
            pad(x, widths)


@pytest.mark.parametrize("fn", [stop_gradient, optimization_barrier])
def test_an_identity_off_jax(fn) -> None:
    for x in (X, strict.asarray(X), {"a": (X, 1.0)}, None):
        assert fn(x) is x


@pytest.mark.parametrize("fn", [stop_gradient, optimization_barrier])
def test_an_identity_refuses_what_is_no_array(fn) -> None:
    with pytest.raises(BackendError):
        fn(object())


@needs_jax
def test_the_jaxprs_are_those_of_jax() -> None:
    import jax
    import jax.numpy as jnp

    x = jnp.asarray(X)
    pairs = [
        (stop_gradient, jax.lax.stop_gradient),
        (optimization_barrier, jax.lax.optimization_barrier),
        (lambda x: pad(x, ((1, 0), (0, 2))), lambda x: jnp.pad(x, ((1, 0), (0, 2)))),
        (lambda x: stop_gradient((x, 1.0)), lambda x: jax.lax.stop_gradient((x, 1.0))),
    ]
    for ours, theirs in pairs:
        assert str(jax.make_jaxpr(ours)(x)) == str(jax.make_jaxpr(theirs)(x))


@needs_jax
def test_no_gradient_goes_through_stop_gradient() -> None:
    import jax

    out = jax.grad(lambda x: x * stop_gradient({"x": x})["x"])(3.0)
    np.testing.assert_allclose(out, 3.0)


@needs_torch
def test_no_gradient_goes_through_stop_gradient_on_torch() -> None:
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    (x * stop_gradient((x, 1.0))[0]).backward()
    assert x.grad == 3.0
