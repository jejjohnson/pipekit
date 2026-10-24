"""Tests for the ``xr.DataArray`` adapter and the apply_ufunc path (#104)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
import pipekit_array as pa
import pytest
from pipekit import NamedArrayAdapter, get_adapter
from pipekit_array import BackendError, DimArray, KernelOp, KernelSpec


xr = pytest.importorskip("xarray")
dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")

from pipekit_array.xarray import XarrayAdapter, from_xarray


BANDS = ["B1", "B2", "B3", "B4"]
TARGET = np.array([0.1, -0.2, 0.3, 0.05])


def normalized_difference(a: Any, b: Any, eps: float) -> Any:
    return (a - b) / (a + b + eps)


def background_stats(x: Any) -> tuple[Any, Any]:
    mean = x.mean(axis=0)
    centred = x - mean
    covariance = centred.T @ centred / (x.shape[0] - 1)
    return mean, covariance + 1e-6 * np.eye(x.shape[1])


def matched_filter_score(x: Any, mu: Any, sigma: Any, *, t: Any) -> Any:
    w = np.linalg.solve(sigma, t)
    return ((x - mu) @ w) / (t @ w)


STATS = KernelOp(
    background_stats,
    KernelSpec("(r,c)->(c),(c,d)", vectorize=True, sizes=(("d", "c"),)),
    core={"r": "y", "c": "band", "d": "band2"},
)
SCORE = KernelOp(
    matched_filter_score,
    KernelSpec("(r,c),(c),(c,d)->(r)", vectorize=True),
    core={"r": "y", "c": "band", "d": "band2"},
    t=TARGET,
)
ND = KernelOp(normalized_difference, "(),()->()", core={}, eps=1e-10)


def _cube(lazy: bool = False, seed: int = 0) -> Any:
    """(time, band, y, x) = (3, 4, 16, 12) with band labels and x/y coordinates."""
    values = np.random.default_rng(seed).random((3, 4, 16, 12))
    data = da.from_array(values, chunks=(1, 4, 16, 6)) if lazy else values
    coords = {"band": BANDS, "y": np.arange(16.0), "x": 100.0 + np.arange(12.0)}
    return xr.DataArray(data, dims=("time", "band", "y", "x"), coords=coords)


def _counting_compute() -> Any:
    """A context counting graph executions (dask's 'compute' count)."""

    class Count:
        calls = 0

        def __call__(self, dsk: Any, keys: Any, **kwargs: Any) -> Any:
            Count.calls += 1
            return dask.get(dsk, keys, **kwargs)

    return Count()


def test_registered_lazily() -> None:
    adapter = get_adapter(_cube())
    assert isinstance(adapter, XarrayAdapter)
    assert isinstance(adapter, NamedArrayAdapter)


def test_import_without_xarray() -> None:
    """``import pipekit_array`` works with xarray blocked, and does not import it."""
    code = textwrap.dedent(
        """
        import importlib.abc, sys

        class NoXarray(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "xarray" or name.startswith("xarray."):
                    raise ImportError(name)
                return None

        sys.meta_path.insert(0, NoXarray())
        import numpy as np
        import pipekit_array as pa
        d = pa.DimArray(np.ones((2, 3)), ("y", "x"))
        assert pa.reduce(d, "sum", "x").dims == ("y",)
        assert "xarray" not in sys.modules
        print("ok")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "ok"


# -- KernelOp ---------------------------------------------------------------------


def test_kernel_parity_with_raw_arrays() -> None:
    cube = _cube()
    nir = cube.sel(band="B4")
    red = cube.sel(band="B3")
    out = ND(nir, red)
    assert isinstance(out, xr.DataArray)
    assert out.dims == ("time", "y", "x")
    np.testing.assert_allclose(
        out.values, normalized_difference(nir.values, red.values, 1e-10)
    )
    # coordinates of surviving dims come along
    np.testing.assert_array_equal(out.coords["x"].values, cube.coords["x"].values)


def test_matched_filter_parity_with_dimarray() -> None:
    """The same KernelOps on a DataArray and on a DimArray agree (PK-K1)."""
    scene = _cube().isel(time=0)
    mu, sigma = STATS(scene)
    assert mu.dims == ("band", "x")
    assert sigma.dims == ("band", "x", "band2")
    alpha = SCORE(scene, mu, sigma)
    assert alpha.dims == ("y", "x")
    named = DimArray(scene.values, ("band", "y", "x"), {"band": tuple(BANDS)})
    mu_n, sigma_n = STATS(named)
    alpha_n = SCORE(named, mu_n, sigma_n)
    np.testing.assert_allclose(alpha.values, alpha_n.data, rtol=1e-12)


def test_kernel_is_lazy_on_dask() -> None:
    """PK-K2: a graph is built, and nothing computes until .compute()."""
    count = _counting_compute()
    cube = _cube(lazy=True)
    with dask.config.set(scheduler=count):
        nir = cube.sel(band="B4")
        red = cube.sel(band="B3")
        out = ND(nir, red)
        assert isinstance(out.data, da.Array)
        assert count.calls == 0
        values = out.compute().values
        assert count.calls >= 1
    eager = _cube()
    expected = normalized_difference(
        eager.sel(band="B4").values, eager.sel(band="B3").values, 1e-10
    )
    np.testing.assert_allclose(values, expected)


def test_output_only_core_dim_is_planned_on_dask() -> None:
    """F5: band2's size comes from spec.sizes, so dask can build the graph."""
    scene = _cube(lazy=True).isel(time=0)
    _, sigma = STATS(scene)
    assert isinstance(sigma.data, da.Array)
    assert sigma.sizes["band2"] == 4
    np.testing.assert_allclose(
        sigma.compute().values, STATS(_cube().isel(time=0))[1].values
    )


def test_offset_coordinates_raise_instead_of_inner_joining() -> None:
    a = _cube().sel(band="B4")
    b = a.assign_coords(x=a.coords["x"] + 1e-7)
    # plain xarray drops every pixel
    assert (a + b).sizes["x"] == 0
    with pytest.raises(ValueError, match="not aligned"):
        ND(a, b)
    with pytest.raises(ValueError, match="not aligned"):
        pa.combine(np.add, a, b)


def test_kwargs_reach_the_kernel() -> None:
    seen: dict[str, Any] = {}

    def spy(a: Any, *, eps: float, scale: Any) -> Any:
        seen.update(eps=eps, scale=scale)
        return a * scale

    op = KernelOp(spy, "()->()", core={}, eps=0.5, scale=2.0)
    out = op(_cube().sel(band="B1"))
    assert seen == {"eps": 0.5, "scale": 2.0}
    np.testing.assert_allclose(out.values, _cube().sel(band="B1").values * 2)


def test_spec_without_xarray_backend() -> None:
    op = KernelOp(
        normalized_difference,
        KernelSpec("(),()->()", backends=("numpy",)),
        core={},
        eps=0.0,
    )
    a = _cube().sel(band="B1")
    with pytest.raises(BackendError, match="xarray"):
        op(a, a)


def test_dataarray_and_dimarray_do_not_mix() -> None:
    a = _cube().sel(band="B1").isel(time=0)
    b = DimArray(a.values, ("y", "x"))
    with pytest.raises(BackendError, match="mix"):
        ND(a, b)


# -- dim-level fast paths ------------------------------------------------------------


def test_fast_paths_equal_the_dimarray_results() -> None:
    cube = _cube()
    named = from_xarray(cube)
    assert named.labels == {
        "band": tuple(BANDS),
        "y": tuple(np.arange(16.0)),
        "x": tuple(100.0 + np.arange(12.0)),
    }
    cases = [
        (lambda c: pa.reduce(c, "mean", "time"), "reduce"),
        (lambda c: pa.reduce(c, np.max, ("time", "y")), "reduce callable"),
        (lambda c: pa.transpose(c, ("x", "y", "band", "time")), "transpose"),
        (lambda c: pa.argreduce(pa.sel(c, band="B2"), "argmax", "time"), "argreduce"),
        (lambda c: pa.isel(c, time=1, x=slice(2, 8)), "isel"),
        (lambda c: pa.sel(c, band=["B4", "B1"]), "sel"),
        (lambda c: pa.stack([pa.isel(c, time=0), pa.isel(c, time=1)], "run"), "stack"),
        (
            lambda c: pa.concat(
                [pa.isel(c, time=slice(0, 1)), pa.isel(c, time=slice(1, None))], "time"
            ),
            "concat",
        ),
        (
            lambda c: pa.combine(
                lambda a, b: a - b, pa.sel(c, band="B1"), pa.sel(c, band="B2")
            ),
            "combine",
        ),
    ]
    for fn, name in cases:
        on_xarray = fn(cube)
        on_dimarray = fn(named)
        assert tuple(on_xarray.dims) == on_dimarray.dims, name
        np.testing.assert_allclose(
            np.asarray(on_xarray.values), np.asarray(on_dimarray.data), err_msg=name
        )


def test_take_along_composite() -> None:
    cube = _cube()
    best = pa.argreduce(pa.sel(cube, band="B2"), "argmax", "time")
    composite = pa.take_along(cube, best, "time")
    assert composite.dims == ("band", "y", "x")
    named = from_xarray(cube)
    expected = pa.take_along(
        named, pa.argreduce(pa.sel(named, band="B2"), "argmax", "time"), "time"
    )
    np.testing.assert_allclose(composite.values, expected.data)


def test_reduce_stays_lazy() -> None:
    count = _counting_compute()
    with dask.config.set(scheduler=count):
        mean = pa.reduce(_cube(lazy=True), "mean", "time")
        assert isinstance(mean.data, da.Array)
        assert count.calls == 0


def test_rewrap_keeps_coords_of_surviving_dims() -> None:
    cube = _cube().isel(time=0)
    adapter = XarrayAdapter()
    summed = adapter.rewrap(cube, cube.values.sum(axis=0), ("y", "x"))
    assert set(summed.coords) == {"y", "x"}
    padded = adapter.rewrap(cube, np.zeros((4, 16, 13)), ("band", "y", "x"))
    assert "x" not in padded.coords


def test_dimarray_round_trip() -> None:
    cube = _cube()
    back = from_xarray(cube).to_xarray()
    xr.testing.assert_identical(back, cube.drop_vars([]))


def test_lazy_mask_kernel_declares_its_dtype() -> None:
    """dask's metadata matches the computed dtype for a bool-valued kernel."""
    op = KernelOp(lambda z: z > 0.5, "(c)->(c)", core={"c": "band"})
    out = op(_cube(lazy=True))
    assert isinstance(out.data, da.Array)
    assert out.dtype == bool
    assert out.compute().dtype == bool


# -- review fixes (Codex, #112) -------------------------------------------------------


def test_vectorized_kernel_keeps_its_output_dtype() -> None:
    ints = xr.DataArray(np.array([[1, 2], [3, 4]]), dims=("y", "c"))
    op = KernelOp(
        lambda v: v.mean(), KernelSpec("(c)->()", vectorize=True), core={"c": "c"}
    )
    np.testing.assert_allclose(op(ints).values, [1.5, 3.5])


def test_reduce_names_follow_numpy_semantics() -> None:
    x = xr.DataArray(np.array([[1.0, np.nan], [3.0, 4.0]]), dims=("t", "x"))
    assert np.isnan(pa.reduce(x, "mean", "t").values[1])
    np.testing.assert_allclose(pa.reduce(x, "nanmean", "t").values, [2.0, 4.0])
    lazy = pa.reduce(_cube(lazy=True), "nanmean", "time")
    assert isinstance(lazy.data, da.Array)


def test_datetime_labels_stay_datetimes() -> None:
    times = np.array(["2026-06-01", "2026-06-02"], dtype="datetime64[ns]")
    x = xr.DataArray(np.arange(2.0), dims=("time",), coords={"time": times})
    assert pa.sel(x, time=times[1]).values == 1.0
    assert from_xarray(x).to_xarray()["time"].dtype == times.dtype


def test_stack_fast_path_refuses_an_existing_dim() -> None:
    a = _cube().isel(time=0)
    with pytest.raises(ValueError, match="already a dim"):
        pa.stack([a, a], "band")


def test_lazy_combine_declares_the_result_dtype() -> None:
    x = xr.DataArray(da.from_array(np.array([1, 3, 5]), chunks=1), dims=("x",))
    halved = pa.combine(lambda a: a / 2, x)
    assert halved.dtype == np.float64
    assert float(halved.sum().compute()) == pytest.approx(4.5)


# -- review fixes (Codex, #112, second round) -----------------------------------------


def test_take_along_rejects_reordered_index_labels() -> None:
    x = xr.DataArray(
        np.arange(6.0).reshape(3, 2), dims=("t", "x"), coords={"x": ["A", "B"]}
    )
    index = xr.DataArray([0, 1], dims=("x",), coords={"x": ["B", "A"]})
    with pytest.raises(ValueError, match="labels"):
        pa.take_along(x, index, "t")


def test_stack_rejects_different_dims() -> None:
    with pytest.raises(ValueError, match="dims"):
        pa.stack([xr.DataArray([1.0], dims="x"), xr.DataArray([1.0], dims="y")], "run")
