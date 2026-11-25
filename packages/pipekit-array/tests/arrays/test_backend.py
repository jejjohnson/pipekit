"""`pipekit_array.arrays`: the library, the dtype and the device of a call."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest
from pipekit_ops import (
    BackendError,
    BackendWarning,
    guard,
)


# the module under test needs both, and so do its tests
pytest.importorskip("array_api_compat")
strict = pytest.importorskip("array_api_strict")

import array_api_compat.numpy as numpy_namespace
import numpy as np
from _array_toys import SECOND, Unknown, affine, density, needs_jax, needs_torch, to_xyz
from hypothesis import given, strategies as st
from pipekit_array.arrays import (
    _backend,
    as_inexact,
    concrete,
    floating_inputs,
    like,
    namespace_of,
)


# --- the import ---


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )


def test_the_package_imports_without_numpy_and_array_api_compat() -> None:
    code = (
        "import sys; sys.modules['numpy'] = sys.modules['array_api_compat'] = None\n"
        "import pipekit_array\n"
        "assert 'pipekit_array.arrays' not in sys.modules\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("library", ["numpy", "array_api_compat"])
def test_the_import_error_names_the_extras(library: str) -> None:
    code = f"import sys; sys.modules[{library!r}] = None\nimport pipekit_array.arrays\n"
    result = _run(code)
    assert result.returncode == 1
    assert "pipekit-array[numpy,compat]" in result.stderr


def test_the_module_imports_neither_jax_nor_xarray() -> None:
    code = (
        "import sys, numpy as np\n"
        "from pipekit_array.arrays import floating_inputs\n"
        "from pipekit_array.kernels import elementwise\n"
        "elementwise()(lambda x: floating_inputs(x)[1][0])(np.zeros(2))\n"
        "assert not {'jax', 'xarray', 'torch', 'dask'} & set(sys.modules)\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


# --- the namespace ---


@pytest.mark.parametrize(
    "value", [1, 1.0, True, None, [1.0], (1.0, 2.0), np.zeros(2), np.float64(1.0)]
)
def test_host_values_alone_resolve_to_numpy(value: object) -> None:
    assert namespace_of(value) is numpy_namespace


def test_no_arguments_resolve_to_numpy() -> None:
    assert namespace_of() is numpy_namespace


def test_numpy_joins_the_other_library() -> None:
    assert namespace_of(np.zeros(2), strict.asarray([1.0])) is strict
    assert namespace_of(strict.asarray([1.0]), 2.0, None) is strict


@needs_jax
def test_jax_wins_over_numpy_and_meets_no_third_library() -> None:
    import jax
    import jax.numpy as jnp

    assert namespace_of(np.zeros(2), jnp.zeros(2)) is jnp
    with pytest.raises(BackendError, match=r"array_api_strict, jax\.numpy"):
        namespace_of(jnp.zeros(2), strict.asarray([1.0]))

    seen = []

    def probe(x):
        seen.append(namespace_of(np.zeros(2), x))
        return x

    jax.jit(probe)(jnp.ones(2))
    assert seen == [jnp]


@pytest.mark.parametrize("value", [Unknown(), object(), "text", {"a": 1}])
def test_an_unknown_array_raises_and_is_never_numpy(value: object) -> None:
    with pytest.raises(BackendError, match="not an array of a known library"):
        namespace_of(np.zeros(2), value)
    with pytest.raises(BackendError, match="not an array of a known library"):
        affine(value, 1.0)


def test_backend_error_is_a_type_error_and_a_value_error() -> None:
    for kind in (TypeError, ValueError):
        with pytest.raises(kind, match="known library"):
            namespace_of("text")


# --- the dtype ---


def test_float32_stays_float32_next_to_numbers_and_integers() -> None:
    x = np.zeros(2, dtype=np.float32)
    _, arrays = floating_inputs(x, 2, 1.5, True, np.asarray([1, 2]), np.asarray(True))
    assert [a.dtype for a in arrays] == [np.float32] * 6


def test_float32_and_float64_give_float64() -> None:
    _, arrays = floating_inputs(np.float32([1.0]), np.float64([1.0]))
    assert [a.dtype for a in arrays] == [np.float64] * 2


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint16, np.int64, np.bool_])
def test_integers_become_the_default_float(dtype: type) -> None:
    _, (x,) = floating_inputs(np.asarray([0, 1], dtype=dtype))
    assert x.dtype == np.float64


def test_numbers_alone_give_the_default_float() -> None:
    _, (x, y) = floating_inputs(1, 2.5)
    assert x.dtype == y.dtype == np.float64
    assert x.shape == ()


def test_a_numpy_scalar_carries_its_dtype() -> None:
    _, (x, y) = floating_inputs(np.float32([1.0]), np.float64(1.0))
    assert x.dtype == y.dtype == np.float64


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([1j]),
        3j,
        np.asarray(["2020-01-01"], dtype="datetime64[D]"),
        np.asarray(["a"]),
        np.asarray([object()]),
    ],
)
def test_values_that_are_not_real_numbers_raise(value: object) -> None:
    with pytest.raises(ValueError, match="real numbers"):
        floating_inputs(value, 1.0)


def test_params_choose_the_namespace_and_never_the_dtype() -> None:
    field = strict.asarray(1.0, dtype=strict.float64)
    xp, (x,) = floating_inputs(np.float32([1.0]), params=(field, 2.0))
    assert xp is strict
    assert x.dtype == strict.float32


def test_like_takes_the_dtype_of_the_reference() -> None:
    assert like(0.1, np.float32([1.0]), np).dtype == np.float32
    assert like(0.1, strict.asarray([1.0]), strict).dtype == strict.float64


@given(
    st.lists(st.floats(-10.0, 10.0), min_size=1, max_size=5),
    st.sampled_from([np.float32, np.float64]),
)
def test_every_input_comes_back_in_one_dtype(values: list[float], dtype: type) -> None:
    x = np.asarray(values, dtype=dtype)
    _, arrays = floating_inputs(x, 1, x[:1], values[0])
    assert {a.dtype for a in arrays} == {np.dtype(dtype)}
    np.testing.assert_array_equal(arrays[0], x)


# --- the device ---


def test_host_values_join_the_device_of_the_array() -> None:
    x = strict.asarray([1.0, 2.0], device=SECOND)
    _, arrays = floating_inputs(1.0, np.ones(2), x, [3.0])
    assert [a.device for a in arrays] == [SECOND] * 4
    assert like(0.1, x, strict).device == SECOND
    assert affine(x, 2.0, np.ones(2)).device == SECOND
    assert [a.device for a in to_xyz(x, np.ones(2))] == [SECOND] * 3


def test_a_parameter_can_choose_the_device() -> None:
    field = strict.asarray(2.0, device=SECOND)
    _, (x,) = floating_inputs([1.0, 2.0], params=(field,))
    assert x.device == SECOND


# --- arrays of the call's library are cast, never rebuilt ---


class Recorder(types.ModuleType):
    """A namespace that fails when ``asarray`` is called on one of its arrays."""

    def __getattr__(self, name: str) -> object:
        return getattr(strict, name)

    def asarray(self, value, **kwargs):
        assert isinstance(value, (float, np.ndarray)), "asarray on a library's array"
        return strict.asarray(value, **kwargs)


def test_an_array_of_the_library_is_never_rebuilt(monkeypatch) -> None:
    # torch.asarray detaches its input in torch 2.5 and 2.10
    recorder = Recorder("recorder")
    monkeypatch.setattr(_backend, "namespace_of", lambda *values: recorder)
    x = strict.asarray([1.0, 2.0])
    field = strict.asarray(2.0)
    xp, (y, z) = floating_inputs(x, 1.5, params=(field,))
    like(field, y, xp)
    assert y is x
    assert z.dtype == x.dtype
    _, (single, _) = floating_inputs(strict.astype(x, strict.float32), field)
    assert single.dtype == strict.float64


@needs_torch
def test_torch_gradients_reach_every_input() -> None:
    import torch

    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([0.1, 0.2], dtype=dtype, requires_grad=True)
        slope = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        affine(x, slope, np.ones(2)).sum().backward()
        assert x.requires_grad and slope.requires_grad
        expected = 2.0 * np.cos([0.2, 0.4])
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-6)
        assert slope.grad is not None


# --- the standard and JAX ---


def test_the_toy_function_runs_on_the_standard_only() -> None:
    x = strict.asarray([0.1, 0.2])
    out = affine(x, strict.asarray(2, dtype=strict.int32), 1.0)
    assert out.dtype == strict.float64
    expected = np.sin(2.0 * np.asarray([0.1, 0.2])) + 1.0
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-15)


@needs_jax
def test_transforms_go_through_the_backend() -> None:
    import jax
    import jax.numpy as jnp

    p, t = jnp.asarray([1e5, 8e4]), jnp.asarray([280.0, 250.0])
    eager = density(p, t)
    np.testing.assert_allclose(jax.jit(density)(p, t), eager, rtol=1e-6)
    np.testing.assert_allclose(jax.vmap(density)(p, t), eager, rtol=1e-6)
    slope = jax.grad(density)(1e5, 280.0)
    np.testing.assert_allclose(slope, eager[0] / 1e5, rtol=1e-6)


@needs_jax
def test_a_traced_parameter_selects_jax() -> None:
    import jax

    def scaled(field):
        xp, (x,) = floating_inputs(np.asarray([1.0, 2.0]), params=(field,))
        return xp.sum(like(field, x, xp) ** 2 * x)

    # d/dr ∑ r² x = 2 r ∑ x
    np.testing.assert_allclose(jax.grad(scaled)(2.0), 12.0, rtol=1e-6)


# --- the guard on moves ---


@needs_jax
def test_guard_reports_a_numpy_array_that_joins_jax():
    import jax.numpy as jnp

    a, b = np.ones(3, np.float32), jnp.ones(3, np.float32)
    floating_inputs(a, b)
    for values in [(a, b), (b, a), (b, [1.0, 2.0, 3.0])]:
        with guard("raise"), pytest.raises(BackendError, match="NumPy data joined jax"):
            floating_inputs(*values)


@needs_jax
def test_guard_lets_numbers_and_constants_of_a_trace_pass():
    import jax
    import jax.numpy as jnp

    a, b = np.ones(3, np.float32), jnp.ones(3, np.float32)
    with guard("raise"):
        floating_inputs(b, 2.0)
        floating_inputs(b, np.float32(2.0), np.asarray(1.0, np.float32))
        floating_inputs(a, a)
        floating_inputs(b, b)
        jax.jit(lambda x: floating_inputs(a, x)[1][1])(b)


@needs_jax
def test_guard_reports_a_wider_dtype():
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True), guard("warn"), pytest.warns(BackendWarning) as got:
        floating_inputs(np.ones(3), jnp.ones(3, jnp.float32))
    assert [str(w.message).split(": ")[1] for w in got] == [
        "NumPy data joined jax",
        "NumPy data widened float32 to float64",
    ]


def test_guard_names_the_library_that_was_joined():
    with guard("raise"), pytest.raises(BackendError, match="joined array_api_strict"):
        floating_inputs(np.ones(3), strict.ones(3))


# --- integers to the default float ---


@pytest.mark.parametrize("value", [np.arange(3), np.asarray([True]), 2, [1, 2], True])
def test_as_inexact_takes_the_default_float_of_numpy(value) -> None:
    out = as_inexact(value)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, value)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_as_inexact_returns_floating_and_complex_arrays(dtype) -> None:
    for xp in (np, strict):
        x = xp.zeros(2, dtype=getattr(xp, dtype))
        assert as_inexact(x) is x


def test_as_inexact_stays_on_the_device() -> None:
    out = as_inexact(strict.ones(2, dtype=strict.int32, device=SECOND))
    assert (out.dtype, out.device) == (strict.float64, SECOND)


def test_as_inexact_refuses_dates() -> None:
    with pytest.raises(ValueError, match="numeric"):
        as_inexact(np.asarray(["2026-11-25"], dtype="datetime64[D]"))


@needs_jax
@pytest.mark.parametrize("x64", [False, True])
def test_as_inexact_follows_jax_x64(x64) -> None:
    import jax
    import jax.numpy as jnp

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64)
    try:
        want = jnp.float64 if x64 else jnp.float32
        assert as_inexact(jnp.arange(3)).dtype == want
        assert jax.jit(as_inexact)(jnp.asarray([True])).dtype == want
        assert as_inexact([1, 2], jnp).dtype == want
    finally:
        jax.config.update("jax_enable_x64", previous)


# --- concrete ---


@pytest.mark.parametrize("value", [1.5, [1.5], np.asarray([1.5]), strict.ones(1) * 1.5])
def test_concrete_gives_the_values_on_the_host(value) -> None:
    out = concrete(value)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.full(out.shape, 1.5))


@needs_jax
def test_concrete_is_none_under_a_trace() -> None:
    import jax
    import jax.numpy as jnp

    seen = []

    def f(x):
        seen.append((concrete(x), concrete(2.0)))
        return x.sum()

    for transformed in (jax.jit(f), jax.grad(f), jax.vmap(f)):
        transformed(jnp.ones((2, 2)))
    assert [(x, float(two)) for x, two in seen] == [(None, 2.0)] * 3
    np.testing.assert_array_equal(concrete(jnp.arange(3)), [0, 1, 2])


@needs_torch
def test_concrete_of_a_tensor_with_a_gradient() -> None:
    import torch

    out = concrete(torch.ones(2, requires_grad=True))
    np.testing.assert_array_equal(out, np.ones(2))


def test_concrete_of_a_cupy_array_copies_it_explicitly(monkeypatch) -> None:
    class OnGpu:
        """What CuPy's arrays do: ``get`` copies, and an implicit copy raises."""

        def get(self) -> np.ndarray:
            return np.asarray([1.0, 2.0])

        def __array__(self, dtype=None, copy=None):
            raise TypeError("Implicit conversion to a NumPy array is not allowed")

    array_api_compat = _backend.array_api_compat
    monkeypatch.setattr(array_api_compat, "is_cupy_array", lambda x: True)
    np.testing.assert_array_equal(concrete(OnGpu()), [1.0, 2.0])
