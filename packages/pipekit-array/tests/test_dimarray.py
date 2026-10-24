"""Tests for `pipekit_array.DimArray` and its reference adapter (#101)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pipekit import NamedArrayAdapter, Signature, get_adapter
from pipekit_array import DimArray, DimArrayAdapter


BANDS = ("B11", "B12")


def _stats() -> DimArray:
    """μ(col, band): three columns, two bands, labelled bands."""
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    return DimArray(data, ("col", "band"), {"band": BANDS})


def test_construction() -> None:
    mu = _stats()
    assert mu.dims == ("col", "band")
    assert mu.shape == (3, 2)
    assert mu.labels == {"band": BANDS}
    # label values are stored as tuples
    listed = DimArray(np.zeros(2), ("band",), {"band": ["B11", "B12"]})
    assert listed.labels["band"] == BANDS
    assert "DimArray[col band](3, 2)" in repr(mu)


@pytest.mark.parametrize(
    ("dims", "labels", "match"),
    [
        (("col",), {}, "shape"),
        (("col", "col"), {}, "unique"),
        (("col", "band"), {"band": ("B11",)}, "1 labels"),
        (("col", "band"), {"level": (1, 2)}, "not in dims"),
    ],
)
def test_construction_errors(
    dims: tuple[str, ...], labels: dict[str, tuple], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        DimArray(np.zeros((3, 2)), dims, labels)


def test_registered_on_import() -> None:
    adapter = get_adapter(_stats())
    assert isinstance(adapter, DimArrayAdapter)
    assert isinstance(adapter, NamedArrayAdapter)


def test_isel_keeps_and_slices_labels() -> None:
    adapter = DimArrayAdapter()
    mu = _stats()
    # a slice keeps the dim and slices its labels
    last_band = adapter.isel(mu, {"band": slice(1, None)})
    assert last_band.dims == ("col", "band")
    assert last_band.labels == {"band": ("B12",)}
    np.testing.assert_array_equal(last_band.data, mu.data[:, 1:])
    # an integer drops the dim and its labels
    one_col = adapter.isel(mu, {"col": 0})
    assert one_col.dims == ("band",)
    assert one_col.labels == {"band": BANDS}
    one_band = adapter.isel(mu, {"band": np.int64(0)})
    assert one_band.dims == ("col",)
    assert one_band.labels == {}
    # an index array keeps the dim, with one label per position
    swapped = adapter.isel(mu, {"band": np.array([1, 0])})
    assert swapped.labels == {"band": ("B12", "B11")}
    np.testing.assert_array_equal(swapped.data, mu.data[:, [1, 0]])


def test_isel_unknown_dim() -> None:
    with pytest.raises(KeyError, match="unknown dim"):
        DimArrayAdapter().isel(_stats(), {"row": 0})


def test_rewrap_drops_labels_of_dropped_or_resized_dims() -> None:
    adapter = DimArrayAdapter()
    mu = _stats()
    # band kept at its size: labels kept
    scaled = adapter.rewrap(mu, mu.data * 2, mu.dims)
    assert scaled.labels == {"band": BANDS}
    # band consumed: no labels
    total = adapter.rewrap(mu, mu.data.sum(axis=1), ("col",))
    assert total.dims == ("col",)
    assert total.labels == {}
    # band resized: its labels no longer fit
    padded = adapter.rewrap(mu, np.zeros((3, 3)), ("col", "band"))
    assert padded.labels == {}
    # a new dim gets none
    covariance = adapter.rewrap(mu, np.zeros((3, 2, 2)), ("col", "band", "band2"))
    assert covariance.labels == {"band": BANDS}


def test_check_compatible() -> None:
    adapter = DimArrayAdapter()
    mu = _stats()
    other = DimArray(np.ones((3, 2)), ("col", "band"), {"band": BANDS})
    assert adapter.check_compatible(mu, other) == (mu, other)
    # different dims combine if the shared ones agree
    per_col = DimArray(np.ones(3), ("col",))
    assert adapter.check_compatible(mu, per_col) == (mu, per_col)
    with pytest.raises(ValueError, match="size"):
        adapter.check_compatible(mu, DimArray(np.ones(4), ("col",)))
    relabelled = DimArray(np.ones((3, 2)), ("col", "band"), {"band": ("B8", "B4")})
    with pytest.raises(ValueError, match="labels"):
        adapter.check_compatible(mu, relabelled)


def test_signature() -> None:
    sig = DimArrayAdapter().signature(_stats())
    assert sig == Signature(dims=(("col", 3), ("band", 2)), dtype="float32")
    assert sig.matches("col band")


def test_jax_pytree() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mu = DimArray(jnp.arange(6.0).reshape(3, 2), ("col", "band"), {"band": BANDS})
    leaves = jax.tree_util.tree_leaves(mu)
    assert len(leaves) == 1
    assert leaves[0] is mu.data
    traces = 0

    def double(d: DimArray) -> DimArray:
        nonlocal traces
        traces += 1
        return DimArrayAdapter().rewrap(d, d.data * 2, d.dims)

    f = jax.jit(double)
    out = f(mu)
    assert out.dims == mu.dims
    assert out.labels == mu.labels
    np.testing.assert_array_equal(np.asarray(out.data), np.asarray(mu.data) * 2)
    # new values, same dims and labels: no retrace
    f(DimArray(jnp.ones((3, 2)), ("col", "band"), {"band": BANDS}))
    assert traces == 1
    # new labels are static: one more trace
    f(DimArray(jnp.ones((3, 2)), ("col", "band"), {"band": ("B8", "B4")}))
    assert traces == 2


def test_to_xarray() -> None:
    xr = pytest.importorskip("xarray")
    da = _stats().to_xarray()
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("col", "band")
    assert tuple(da.coords["band"].values.tolist()) == BANDS
    assert "col" not in da.coords


def test_to_xarray_without_xarray(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "xarray", None)
    with pytest.raises(ImportError, match=r"pipekit-array\[xarray\]"):
        _stats().to_xarray()


def test_bare_vmap_drops_and_restores_the_leading_dim() -> None:
    jax = pytest.importorskip("jax")
    cube = DimArray(np.ones((3, 2, 4)), ("time", "band", "x"), {"band": ("a", "b")})
    seen: list[tuple[str, ...]] = []

    def per_time(c: DimArray) -> DimArray:
        seen.append(c.dims)
        assert c.labels == {"band": ("a", "b")}
        return DimArray(c.data.sum(axis=0), ("x",))

    out = jax.vmap(per_time)(cube)
    assert seen == [("band", "x")]
    assert out.dims == ("batch", "x")
    assert out.shape == (3, 4)


def test_vmapped_dims_skips_taken_names() -> None:
    from pipekit_array.dimarray import vmapped_dims

    assert vmapped_dims(("batch", "x"), (2, 3, 4)) == ("batch_1", "batch", "x")
    assert vmapped_dims(("y", "x"), None) == ("y", "x")


def test_isel_index_arrays_select_independently() -> None:
    d = DimArray(np.arange(12).reshape(3, 4), ("row", "col"), {"row": ("a", "b", "c")})
    out = DimArrayAdapter().isel(d, {"row": [0, 2], "col": [1, 3]})
    assert out.dims == ("row", "col") and out.shape == (2, 2)
    np.testing.assert_array_equal(out.data, [[1, 3], [9, 11]])
    assert out.labels == {"row": ("a", "c")}


def test_isel_boolean_mask_filters_labels() -> None:
    d = DimArray(np.arange(3.0), ("x",), {"x": ("a", "b", "c")})
    for mask in ([True, False, True], np.array([True, False, True])):
        out = DimArrayAdapter().isel(d, {"x": mask})
        np.testing.assert_array_equal(out.data, [0.0, 2.0])
        assert out.labels == {"x": ("a", "c")}


def test_isel_rejects_a_mask_of_the_wrong_length() -> None:
    """Codex #109: a short mask is an error, not a partial selection."""
    d = DimArray(np.arange(3.0), ("x",))
    with pytest.raises(IndexError, match=r"\(3,\)"):
        DimArrayAdapter().isel(d, {"x": [True, False]})
    with pytest.raises(IndexError):
        DimArrayAdapter().isel(d, {"x": np.ones((3, 1), dtype=bool)})


@pytest.mark.parametrize("labelled", [False, True])
def test_isel_with_a_0d_integer_array_drops_the_dim(labelled: bool) -> None:
    """Codex #109: ``np.array(1)`` selects like ``1``."""
    labels = {"x": ("a", "b", "c")} if labelled else {}
    d = DimArray(np.arange(6.0).reshape(2, 3), ("y", "x"), labels)
    out = DimArrayAdapter().isel(d, {"x": np.array(1)})
    assert out.dims == ("y",)
    np.testing.assert_array_equal(out.data, [1.0, 4.0])
    assert out.labels == {}


def test_isel_with_a_traced_scalar_index_drops_the_dim() -> None:
    jax = pytest.importorskip("jax")
    d = DimArray(np.arange(6.0).reshape(2, 3), ("y", "x"))

    @jax.jit
    def pick(i: Any) -> Any:
        out = DimArrayAdapter().isel(
            DimArray(jax.numpy.asarray(d.data), d.dims), {"x": i}
        )
        assert out.dims == ("y",)
        return out.data

    np.testing.assert_array_equal(pick(jax.numpy.asarray(2)), [2.0, 5.0])


def test_isel_rejects_multidimensional_positions() -> None:
    """Codex #109: a 2-D position array would add an axis ``x`` cannot name."""
    d = DimArray(np.arange(3.0), ("x",))
    with pytest.raises(IndexError, match="1-D"):
        DimArrayAdapter().isel(d, {"x": np.array([[0, 1]])})
    with pytest.raises(IndexError, match="1-D"):
        DimArrayAdapter().isel(d, {"x": [[0, 1]]})
