"""`pipekit_array.kernels.elementwise`: labelled arrays in a function of plain ones."""

from __future__ import annotations

import inspect

import pytest
from pipekit_ops import (
    BackendError,
    MissingValueAdapter,
    backend_of,
    get_adapter,
    guard,
    register_adapter,
    unregister_adapter,
)


# the module under test needs both, and so do its tests
pytest.importorskip("array_api_compat")
strict = pytest.importorskip("array_api_strict")

import numpy as np
from _array_toys import SECOND, density, needs_jax, to_xyz
from pipekit_array import DimArray, DimArrayAdapter, expand_dims, transpose
from pipekit_array.arrays import is_dataarray, namespace_of
from pipekit_array.kernels import elementwise


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
            "level": ("level", [1000, 800], {"units": "hPa"}),
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
        assert out.level.attrs == {"units": "hPa"}
        assert out.dtype == np.float32
    assert temperature.attrs == {"units": "K"}


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


def _total(first, *others, scale=1.0, **more):
    for other in (*others, *more.values()):
        first = first + other
    return scale * first


def test_variadic_arguments_align_by_name_too(xr) -> None:
    total = elementwise()(_total)
    a = xr.DataArray([1.0, 2.0], dims="x")
    b = xr.DataArray([10.0, 20.0, 30.0], dims="y")
    got = total(1.0, 2.0, a, b, scale=2.0, c=a)
    assert got.dims == ("x", "y")
    np.testing.assert_array_equal(got, 2.0 * (3.0 + (a + b) + a))
    with pytest.raises(ValueError, match="'others' has dimensions"):
        total(a, 1.0, np.ones(2))
    with pytest.raises(ValueError, match="'scale' has dimensions"):
        total(a, scale=np.ones(2))
    with pytest.raises(ValueError, match="'c' has dimensions"):
        total(a, c=np.ones(2))


def test_variadic_dimarrays_align_by_name_too() -> None:
    a = DimArray(np.asarray([1.0, 2.0]), ("x",))
    b = DimArray(np.asarray([10.0, 20.0, 30.0]), ("y",))
    got = elementwise()(_total)(1.0, a, b, c=a)
    assert got.dims == ("x", "y")
    np.testing.assert_array_equal(
        got.data, 1.0 + 2.0 * a.data[:, None] + b.data[None, :]
    )


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
def test_dimarrays_trace_and_join_backends() -> None:
    import jax
    import jax.numpy as jnp

    on_jax = DimArray(jnp.asarray([280.0, 250.0]), ("x",))
    on_numpy = DimArray(np.asarray([280.0, 250.0]), ("x",))
    out = jax.jit(density)(1e5, on_jax)
    np.testing.assert_allclose(out.data, density(1e5, on_jax).data, rtol=1e-6)
    mixed = density(on_numpy, on_jax)
    assert backend_of(mixed) == "jax"
    with (
        guard("raise"),
        pytest.raises(BackendError, match="density: NumPy"),
    ):
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


# --- missing elements ---


class Filled(DimArray):
    """A `DimArray` whose fill value marks missing elements."""

    fill = -9999.0


class FilledAdapter(DimArrayAdapter):
    def rewrap(self, x, data, dims):
        return Filled(data, dims)

    def keep_missing(self, out, *inputs):
        xp = namespace_of(out.data)
        missing = xp.zeros(out.shape, dtype=bool)
        for x in inputs:
            lacking = [d for d in out.dims if d not in x.dims]
            full = expand_dims(x, lacking) if lacking else x
            missing = missing | (transpose(full, out.dims).data == Filled.fill)
        return Filled(xp.where(missing, Filled.fill, out.data), out.dims)


@pytest.fixture
def filled():
    register_adapter(Filled, FilledAdapter())
    yield
    unregister_adapter(Filled)


def test_missing_elements_stay_missing(filled) -> None:
    fill = Filled.fill
    pressure = Filled(np.asarray([[1e5, fill], [9e4, 8e4]]), ("y", "x"))
    temperature = Filled(np.asarray([300.0, fill]), ("y",))
    out = density(pressure, 300.0)
    assert isinstance(out, Filled)
    np.testing.assert_array_equal(out.data == fill, [[False, True], [False, False]])
    np.testing.assert_allclose(out.data[0, 0], 1e5 / (287.0 * 300.0))
    both = density(pressure, temperature)
    np.testing.assert_array_equal(both.data == fill, [[False, True], [True, True]])


def test_missing_elements_stay_missing_in_every_output(filled) -> None:
    @elementwise(n_outputs=2)
    def twice_and_half(x):
        return 2 * x, x / 2

    given = Filled(np.asarray([1.0, Filled.fill]), ("x",))
    for out, value in zip(twice_and_half(given), (2.0, 0.5), strict=True):
        np.testing.assert_array_equal(out.data, [value, Filled.fill])


def test_the_adapter_protocol_for_missing_values() -> None:
    assert isinstance(FilledAdapter(), MissingValueAdapter)
    assert not isinstance(DimArrayAdapter(), MissingValueAdapter)
