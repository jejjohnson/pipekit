"""`pipekit_array.einx`: a named-dims einx bridge on any carrier (#131, OP-O2)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from pipekit_array import DimArray


einx = pytest.importorskip("einx")
from pipekit import BackendError
from pipekit_array import einx as px


def _cube() -> DimArray:
    data = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
    return DimArray(data, ("time", "band", "y", "x"), {"band": ("a", "b", "c")})


def test_rearrange_transposes_by_name_and_keeps_labels() -> None:
    cube = _cube()
    out = px.rearrange("time band y x -> time y x band", cube)
    assert out.dims == ("time", "y", "x", "band")
    np.testing.assert_array_equal(out.data, np.transpose(cube.data, (0, 2, 3, 1)))
    assert out.labels["band"] == ("a", "b", "c")
    # a flat slot in another order is matched by name
    again = px.rearrange("band time x y -> time band y x", cube)
    np.testing.assert_array_equal(again.data, cube.data)


def test_rearrange_splits_and_merges_dims() -> None:
    field = DimArray(np.arange(16.0).reshape(4, 4), ("y", "x"))
    patches = px.rearrange("(yb yi) (xb xi) -> (yb xb) yi xi", field, yi=2, xi=2)
    assert patches.dims == ("yb_xb", "yi", "xi")
    want = einx.id("(yb yi) (xb xi) -> (yb xb) yi xi", field.data, yi=2, xi=2)
    np.testing.assert_array_equal(patches.data, want)


@pytest.mark.parametrize("op", ["mean", "sum", "max", np.median])
def test_reduce_matches_einx(op) -> None:
    cube = _cube()
    out = px.reduce("time band y x -> band y x", cube, op=op)
    assert out.dims == ("band", "y", "x")
    fn = getattr(np, op) if isinstance(op, str) else op
    np.testing.assert_allclose(out.data, fn(cube.data, axis=0))


def test_einsum_contracts_by_name_whatever_the_input_order() -> None:
    cube = _cube()
    w = DimArray(np.arange(3.0 * 2).reshape(2, 3), ("k", "band"))  # slot says "band k"
    out = px.einsum("time band y x, band k -> time k y x", cube, w)
    assert out.dims == ("time", "k", "y", "x")
    want = np.einsum("tbyx,bk->tkyx", cube.data, w.data.T)
    np.testing.assert_allclose(out.data, want)


def test_repeat_broadcasts_along_a_new_dim() -> None:
    field = DimArray(np.arange(20.0).reshape(4, 5), ("y", "x"))
    out = px.repeat("y x -> member y x", field, member=3)
    assert out.dims == ("member", "y", "x") and out.shape == (3, 4, 5)
    np.testing.assert_array_equal(out.data[2], field.data)


def test_the_operator_serialises_and_runs() -> None:
    op = px.Einx("reduce", "time band y x -> band y x", op="mean")
    np.testing.assert_allclose(op(_cube()).data, _cube().data.mean(axis=0))
    assert op.get_config()["op"] == "mean"
    with pytest.raises(ValueError, match="op is given"):
        px.Einx("reduce", "a -> ")
    with pytest.raises(px.PatternError):
        px.Einx("rearrange", "a b")


def test_jax_under_jit_keeps_its_backend() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    dims = ("time", "band", "y", "x")

    @jax.jit
    def step(data):
        cube = DimArray(data, dims)
        return px.reduce("time band y x -> band y x", cube, op="mean").data

    out = step(jnp.asarray(_cube().data))
    assert isinstance(out, jax.Array)
    np.testing.assert_allclose(np.asarray(out), _cube().data.mean(axis=0))


def test_xarray_keeps_the_coordinates_of_surviving_dims() -> None:
    xr = pytest.importorskip("xarray")
    da = xr.DataArray(
        _cube().data,
        dims=("time", "band", "y", "x"),
        coords={"band": list("abc"), "x": np.arange(5.0) * 10},
    )
    out = px.reduce("time band y x -> band y x", da, op="mean")
    assert isinstance(out, xr.DataArray)
    assert list(out["band"].values) == list("abc")
    np.testing.assert_array_equal(out["x"].values, np.arange(5.0) * 10)


def test_dask_is_refused_not_computed() -> None:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask")
    da = xr.DataArray(_cube().data, dims=("time", "band", "y", "x")).chunk({"time": 1})
    with pytest.raises(BackendError, match=r"\.compute\(\)"):
        px.reduce("time band y x -> band y x", da, op="mean")


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: px.reduce("time band y -> band y", _cube(), op="mean"), "has dims"),
        (lambda: px.reduce("time band y x", _cube(), op="mean"), "exactly one '->'"),
        (
            lambda: px.reduce("[time] band y x -> band y x", _cube(), op="mean"),
            "not supported",
        ),
        (lambda: px.rearrange("(time band) y x -> y x", _cube()), "in order"),
        (
            lambda: px.reduce("time band y x -> y x", _cube(), op="median2"),
            "op must be",
        ),
        (lambda: px.einsum("time band y x -> time", _cube(), _cube()), "input"),
    ],
)
def test_bad_patterns_name_the_problem(call, match) -> None:
    with pytest.raises(px.PatternError, match=match):
        call()


def test_importing_pipekit_array_does_not_import_einx() -> None:
    code = "import sys, pipekit_array, pipekit_array.einx; print('einx' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
