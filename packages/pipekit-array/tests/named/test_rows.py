"""`to_rows` / `from_rows`: carriers to sample tables and back (#129, OP-O3)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from pipekit_array import DimArray, RowLayout, from_rows, to_rows


DIMS = ("time", "band", "y", "x")


def _cube() -> DimArray:
    data = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
    return DimArray(data, DIMS, {"band": ("B2", "B3", "B4")})


def _all_sample_choices() -> list[tuple[str, ...]]:
    return [
        combo
        for n in range(1, len(DIMS) + 1)
        for combo in itertools.combinations(DIMS, n)
    ]


@pytest.mark.parametrize("sample_dims", _all_sample_choices())
def test_round_trip_for_every_choice_of_sample_dims(sample_dims) -> None:
    cube = _cube()
    rows, layout = to_rows(cube, sample_dims)
    assert rows.shape == (layout.n_samples, layout.n_features)
    back = from_rows(rows, layout)
    assert back.dims == cube.dims
    np.testing.assert_array_equal(back.data, cube.data)


def test_rows_are_c_order_over_the_carrier_dims() -> None:
    """Row i, column j is the cube at sample i and feature j, both C-ordered."""
    cube = _cube()
    rows, layout = to_rows(cube, ("x", "time"))  # given out of order on purpose
    assert layout.sample_dims == ("time", "x")
    assert layout.feature_dims == ("band", "y")
    want = np.transpose(cube.data, (0, 3, 1, 2)).reshape(2 * 5, 3 * 4)
    np.testing.assert_array_equal(rows, want)


def test_a_new_feature_dim_takes_the_place_of_the_features() -> None:
    cube = _cube()
    rows, layout = to_rows(cube, ("time", "y", "x"))
    scores = rows[:, :2] * 2.0
    out = from_rows(scores, layout, feature_dim="component")
    assert out.dims == ("time", "component", "y", "x")
    assert out.shape == (2, 2, 4, 5)
    np.testing.assert_array_equal(out.data[:, 0], cube.data[:, 0] * 2.0)
    assert "band" not in out.labels  # the band labels went with the band dim


def test_one_value_per_sample_gives_the_sample_dims() -> None:
    cube = _cube()
    rows, layout = to_rows(cube, ("y", "x"))
    labels = rows.sum(axis=1)
    out = from_rows(labels, layout)
    assert out.dims == ("y", "x")
    np.testing.assert_array_equal(out.data, cube.data.sum(axis=(0, 1)))


def test_no_feature_dims_gives_one_column() -> None:
    cube = _cube()
    rows, layout = to_rows(cube, DIMS)
    assert rows.shape == (120, 1) and layout.feature_dims == ()
    out = from_rows(rows * 0 + 1, layout, feature_dim="score")
    assert out.dims == ("score", *DIMS)


def test_layout_equality_ignores_the_template() -> None:
    _, a = to_rows(_cube(), ("y", "x"))
    _, b = to_rows(DimArray(np.zeros((2, 3, 4, 5)), DIMS), ("y", "x"))
    assert a == b
    assert isinstance(a, RowLayout) and "template" not in repr(a)


@pytest.mark.parametrize(
    ("sample_dims", "match"),
    [((), "at least one"), (("y", "y"), "repeat"), (("lat",), "not in")],
)
def test_bad_sample_dims_name_the_problem(sample_dims, match) -> None:
    with pytest.raises(ValueError, match=match):
        to_rows(_cube(), sample_dims)


def test_rows_that_do_not_fit_name_the_problem() -> None:
    rows, layout = to_rows(_cube(), ("y", "x"))
    with pytest.raises(ValueError, match="do not fit"):
        from_rows(rows[:-1], layout)
    with pytest.raises(ValueError, match="feature_dim"):
        from_rows(rows[:, :2], layout)
    with pytest.raises(ValueError, match="already a dim"):
        from_rows(rows[:, :2], layout, feature_dim="y")


def test_a_raw_array_is_refused() -> None:
    with pytest.raises(TypeError, match="named carriers"):
        to_rows(np.zeros((2, 3)), ("y",))


def test_jax_under_jit_keeps_its_backend() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    cube = DimArray(jnp.asarray(_cube().data), DIMS)

    @jax.jit
    def centred(data):
        rows, layout = to_rows(DimArray(data, DIMS), ("time", "y", "x"))
        return from_rows(rows - rows.mean(axis=0), layout).data

    out = centred(cube.data)
    assert isinstance(out, jax.Array)
    want = cube.data - cube.data.mean(axis=(0, 2, 3), keepdims=True)
    np.testing.assert_allclose(np.asarray(out), np.asarray(want), rtol=1e-12)


def test_xarray_keeps_its_coordinates_and_dask_stays_lazy() -> None:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask")
    da = xr.DataArray(
        _cube().data,
        dims=DIMS,
        coords={"band": ["B2", "B3", "B4"], "x": np.arange(5) * 10.0},
    )
    rows, layout = to_rows(da, ("y", "x"))
    back = from_rows(rows, layout)
    assert isinstance(back, xr.DataArray)
    xr.testing.assert_identical(back, da)

    lazy = da.chunk({"time": 1})
    lazy_rows, lazy_layout = to_rows(lazy, ("time", "y", "x"))
    assert hasattr(lazy_rows, "dask")
    out = from_rows(lazy_rows, lazy_layout)
    assert hasattr(out.data, "dask")
    xr.testing.assert_identical(out.compute(), da)
