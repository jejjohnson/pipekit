"""`pipekit_array.functions`: the array backend of plain functions."""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
import types

import pytest
from pipekit_ops import BackendError, get_adapter


# the module under test needs both, and so do its tests
pytest.importorskip("array_api_compat")
strict = pytest.importorskip("array_api_strict")

import array_api_compat.numpy as numpy_namespace
import numpy as np
import pipekit_array.functions as functions
from hypothesis import given, strategies as st
from pipekit_array import DimArray, array_namespace
from pipekit_array.functions import (
    elementwise,
    floating_inputs,
    is_dataarray,
    like,
    namespace_of,
)


SECOND = strict.Device("device1")


def _missing(library: str) -> bool:
    return importlib.util.find_spec(library) is None


needs_jax = pytest.mark.skipif(_missing("jax"), reason="JAX is not installed")
needs_torch = pytest.mark.skipif(_missing("torch"), reason="torch is not installed")


class Unknown:
    """An array-like of no known library, as a pandas Series is."""

    shape = (2,)
    ndim = 1
    dtype = np.dtype(np.float64)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.zeros(2)


def affine(values, slope, offset=0.0):
    """The toy function: sin(slope · values) + offset, through the backend."""
    xp, (values, slope, offset) = floating_inputs(values, slope, offset)
    return xp.sin(slope * values) + offset


@elementwise()
def density(pressure, temperature, *, constant=287.0):
    """Density = p / (R T)."""
    xp, (p, t) = floating_inputs(pressure, temperature)
    return p / (like(constant, t, xp) * t)


@elementwise(n_outputs=3)
def to_xyz(lon, lat, degrees=False):
    """Unit vectors; z does not depend on the longitude."""
    xp, (lon, lat) = floating_inputs(lon, lat)
    if degrees:
        lon, lat = lon * xp.pi / 180.0, lat * xp.pi / 180.0
    return xp.cos(lat) * xp.cos(lon), xp.cos(lat) * xp.sin(lon), xp.sin(lat)


# --- the import ---


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )


def test_the_package_imports_without_numpy_and_array_api_compat() -> None:
    code = (
        "import sys; sys.modules['numpy'] = sys.modules['array_api_compat'] = None\n"
        "import pipekit_array\n"
        "assert 'pipekit_array.functions' not in sys.modules\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("library", ["numpy", "array_api_compat"])
def test_the_import_error_names_the_extras(library: str) -> None:
    code = (
        f"import sys; sys.modules[{library!r}] = None\nimport pipekit_array.functions\n"
    )
    result = _run(code)
    assert result.returncode == 1
    assert "pipekit-array[numpy,compat]" in result.stderr


def test_the_module_imports_neither_jax_nor_xarray() -> None:
    code = (
        "import sys, numpy as np\n"
        "from pipekit_array.functions import elementwise, floating_inputs\n"
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


@needs_jax
def test_the_operators_still_refuse_numpy_next_to_jax() -> None:
    import jax.numpy as jnp

    with pytest.raises(TypeError, match="multiple Array API backends"):
        array_namespace(np.zeros(2), jnp.zeros(2))


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
    monkeypatch.setattr(functions, "namespace_of", lambda *values: recorder)
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


# --- DataArrays ---


@pytest.fixture
def xr():
    return pytest.importorskip("xarray")


@pytest.fixture
def temperature(xr):
    return xr.DataArray(
        np.asarray([[280.0, 290.0], [250.0, 260.0]], dtype=np.float32),
        dims=("level", "x"),
        coords={
            "level": [1000, 800],
            "x": [1, 2],
            "height": ("level", [100.0, 2000.0]),
        },
        name="t",
        attrs={"units": "K"},
    )


@pytest.fixture
def pressure(xr):
    values = np.asarray([1e5, 8e4], dtype=np.float32)
    return xr.DataArray(values, {"level": [1000, 800]}, ("level",), name="p")


def test_plain_arrays_pass_through() -> None:
    out = density(np.asarray([1e5, 8e4]), 280.0)
    assert type(out) is np.ndarray
    assert not is_dataarray(out)


def test_a_square_grid_aligns_by_name(pressure, temperature) -> None:
    out = density(pressure, temperature)
    expected = density(pressure.values[:, None], temperature.values)
    assert out.dims == ("level", "x")
    np.testing.assert_array_equal(out.values, expected)
    by_position = density(pressure.values, temperature.values)
    assert np.max(np.abs(by_position / expected - 1.0)) > 0.2


def test_coordinates_stay_and_the_name_and_attributes_go(pressure, temperature) -> None:
    for out in (density(pressure, temperature), density(1e5, temperature)):
        assert set(out.coords) == {"level", "x", "height"}
        assert out.name is None
        assert out.attrs == {}
        assert out.dtype == np.float32


def test_every_output_has_the_dimensions_of_all_the_inputs(xr) -> None:
    lat = xr.DataArray([10.0, 20.0], dims="lat")
    lon = xr.DataArray([0.0, 5.0, 10.0], dims="lon")
    for out in to_xyz(lon, lat):
        assert is_dataarray(out)
        assert out.sizes == {"lon": 3, "lat": 2}


def test_mismatched_coordinates_raise(pressure, temperature) -> None:
    with pytest.raises(ValueError, match="level"):
        density(pressure.assign_coords(level=[1000, 850]), temperature)


@pytest.mark.parametrize("plain", [np.ones(2), [1e5, 8e4], (1e5, 8e4)])
def test_an_array_without_names_is_not_guessed(temperature, plain) -> None:
    match = "'pressure' has dimensions but no names.*DataArray 'temperature'"
    with pytest.raises(ValueError, match=match):
        density(plain, temperature)
    named = DimArray(temperature.values, ("level", "x"))
    with pytest.raises(ValueError, match="DimArray 'temperature'"):
        density(plain, named)


def test_keywords_align_and_options_pass_through(xr, pressure, temperature) -> None:
    constant = xr.DataArray([280.0, 290.0], {"x": [1, 2]}, ("x",))
    out = density(temperature=temperature, pressure=pressure, constant=constant)
    expected = pressure.values[:, None] / (constant.values * temperature.values)
    np.testing.assert_allclose(out.values, expected, rtol=1e-6)
    x, _, _ = to_xyz(xr.DataArray([0.0, 90.0], dims="x"), np.float64(0.0), degrees=True)
    np.testing.assert_allclose(x.values, [1.0, 0.0], atol=1e-15)


def test_dask_stays_lazy(xr, pressure, temperature) -> None:
    da = pytest.importorskip("dask.array")
    out = density(pressure, temperature.chunk({"x": 1}))
    assert isinstance(out.data, da.Array)
    np.testing.assert_array_equal(out, density(pressure, temperature))


@needs_jax
def test_a_dataarray_of_jax_holds_jax_and_numpy_joins(xr) -> None:
    import jax
    import jax.numpy as jnp

    on_jax = xr.DataArray(jnp.arange(1.0, 4.0), dims="x")
    on_numpy = xr.DataArray(np.arange(1.0, 4.0), dims="x")
    assert isinstance(density(on_jax, on_jax).data, jax.Array)
    assert isinstance(density(on_numpy, on_jax).data, jax.Array)


def test_a_dataset_is_not_an_array(xr) -> None:
    with pytest.raises(BackendError, match="Dataset"):
        density(xr.Dataset({"p": ("x", [1.0])}), 280.0)


# --- carriers with an adapter ---


def test_a_dimarray_equals_the_function_on_the_data() -> None:
    p = DimArray(np.asarray([1e5, 8e4]), ("level",), {"level": (1000, 800)})
    t = DimArray(np.asarray([[280.0, 250.0], [290.0, 260.0]]), ("x", "level"))
    out = density(p, t)
    assert out.dims == ("level", "x")
    np.testing.assert_array_equal(out.data, density(p.data[:, None], t.data.T))
    assert out.labels["level"] == (1000, 800)
    np.testing.assert_array_equal(density(1e5, t).data, density(1e5, t.data))


def test_every_dimarray_output_has_the_dimensions_of_all_the_inputs() -> None:
    lon = DimArray(np.asarray([0.0, 5.0, 10.0]), ("lon",))
    lat = DimArray(np.asarray([10.0, 20.0]), ("lat",))
    outs = to_xyz(lon, lat)
    assert [out.dims for out in outs] == [("lon", "lat")] * 3
    assert [out.shape for out in outs] == [(3, 2)] * 3
    for out, raw in zip(outs, to_xyz(lon.data[:, None], lat.data), strict=True):
        np.testing.assert_array_equal(out.data, np.broadcast_to(raw, (3, 2)))


def test_a_dimarray_of_another_library_stays_on_its_device() -> None:
    x = DimArray(strict.asarray([1.0, 2.0], device=SECOND), ("x",))
    out = density(1e5, x)
    assert out.data.device == SECOND


@needs_jax
def test_dimarrays_trace_and_refuse_mixed_backends() -> None:
    import jax
    import jax.numpy as jnp

    on_jax = DimArray(jnp.asarray([280.0, 250.0]), ("x",))
    on_numpy = DimArray(np.asarray([280.0, 250.0]), ("x",))
    out = jax.jit(density)(1e5, on_jax)
    np.testing.assert_allclose(out.data, density(1e5, on_jax).data, rtol=1e-6)
    with pytest.raises(BackendError, match="inputs on"):
        density(on_numpy, on_jax)
    # a traced number joins the carrier of NumPy
    slope = jax.grad(lambda p: density(p, on_numpy).data.sum())(1e5)
    expected = np.sum(1.0 / (287.0 * np.asarray([280.0, 250.0])))
    np.testing.assert_allclose(slope, expected, rtol=1e-6)


def test_a_dataarray_next_to_a_dimarray_raises(temperature) -> None:
    other = DimArray(np.asarray([1e5, 8e4]), ("level",))
    with pytest.raises(BackendError, match="'pressure' is a DimArray"):
        density(other, temperature)


def test_a_geoarray_keeps_its_grid() -> None:
    geoarray = pytest.importorskip("geoarray")
    data = np.linspace(250.0, 300.0, 24).reshape(2, 3, 4)
    scene = geoarray.GeoArray(data, (20, 0, 0, 0, -20, 0), 32613, dims="band y x")
    out = density(1e5, scene)
    assert type(out) is geoarray.GeoArray
    assert out.dims == scene.dims
    np.testing.assert_array_equal(out.data, density(1e5, data))
    assert get_adapter(out).check_compatible(out, scene)


# --- the decorator ---


def test_the_decorator_keeps_the_signature_and_docstring() -> None:
    assert density.__name__ == "density"
    assert density.__doc__ == "Density = p / (R T)."
    assert list(inspect.signature(density).parameters) == [
        "pressure",
        "temperature",
        "constant",
    ]


def test_a_missing_argument_is_reported_as_python_does(temperature) -> None:
    for pressure in (1e5, temperature):
        with pytest.raises(TypeError, match="temperature"):
            density(pressure)
