"""Tests for the dim-level functions and their operators (#102)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pipekit_array as pa
import pytest
from pipekit import Signature, get_adapter
from pipekit_array import (
    ArgReduce,
    Combine,
    ConcatenateAlong,
    DimArray,
    DimArrayAdapter,
    ISel,
    MapOver,
    Reduce,
    Sel,
    StackAlong,
    TakeAlong,
    Transpose,
)


BANDS = ("B4", "B8")


def _cube(xp: Any = np) -> DimArray:
    """(time, band, x) = (3, 2, 4), labelled bands."""
    data = xp.asarray(np.arange(24.0).reshape(3, 2, 4))
    return DimArray(data, ("time", "band", "x"), {"band": BANDS})


@pytest.fixture(params=["numpy", "jax"])
def xp(request: pytest.FixtureRequest) -> Any:
    if request.param == "jax":
        return pytest.importorskip("jax.numpy")
    return np


def _signature(d: DimArray) -> Signature:
    adapter = get_adapter(d)
    assert adapter is not None
    return adapter.signature(d)


# -- functions ---------------------------------------------------------------


def test_isel(xp: Any) -> None:
    cube = _cube(xp)
    first = pa.isel(cube, time=0)
    assert first.dims == ("band", "x")
    assert first.labels == {"band": BANDS}
    np.testing.assert_array_equal(np.asarray(first.data), np.asarray(cube.data)[0])
    window = pa.isel(cube, x=slice(1, 3), band=1)
    assert window.dims == ("time", "x")
    assert window.shape == (3, 2)
    with pytest.raises(KeyError, match="unknown dim"):
        pa.isel(cube, y=0)


def test_sel(xp: Any) -> None:
    cube = _cube(xp)
    nir = pa.sel(cube, band="B8")
    assert nir.dims == ("time", "x")
    np.testing.assert_array_equal(np.asarray(nir.data), np.asarray(cube.data)[:, 1])
    swapped = pa.sel(cube, band=["B8", "B4"])
    assert swapped.labels == {"band": ("B8", "B4")}
    with pytest.raises(KeyError, match="not a label"):
        pa.sel(cube, band="B11")
    with pytest.raises(KeyError, match="has no labels"):
        pa.sel(cube, time=0)


def test_transpose(xp: Any) -> None:
    cube = _cube(xp)
    moved = pa.transpose(cube, ("x", "time", "band"))
    assert moved.dims == ("x", "time", "band")
    assert moved.labels == {"band": BANDS}
    np.testing.assert_array_equal(
        np.asarray(moved.data), np.transpose(np.asarray(cube.data), (2, 0, 1))
    )
    with pytest.raises(ValueError, match="permutation"):
        pa.transpose(cube, ("x", "time"))


def test_reduce(xp: Any) -> None:
    cube = _cube(xp)
    mean = pa.reduce(cube, "mean", "time")
    assert mean.dims == ("band", "x")
    assert mean.labels == {"band": BANDS}
    np.testing.assert_allclose(np.asarray(mean.data), np.asarray(cube.data).mean(0))
    total = pa.reduce(cube, xp.sum, ("time", "x"))
    assert total.dims == ("band",)
    np.testing.assert_allclose(
        np.asarray(total.data), np.asarray(cube.data).sum((0, 2))
    )
    with pytest.raises(ValueError, match="no 'mode'"):
        pa.reduce(cube, "mode", "time")


def test_argreduce_and_take_along_make_a_composite(xp: Any) -> None:
    """Max-NDVI composite: the band values at each pixel's best time."""
    rng = np.random.default_rng(0)
    values = rng.random((5, 2, 4))
    cube = DimArray(xp.asarray(values), ("time", "band", "x"), {"band": BANDS})
    score = pa.sel(cube, band="B8")
    best = pa.argreduce(score, "argmax", "time")
    assert best.dims == ("x",)
    composite = pa.take_along(cube, best, "time")
    assert composite.dims == ("band", "x")
    expected = values[np.argmax(values[:, 1], axis=0), :, np.arange(4)].T
    np.testing.assert_allclose(np.asarray(composite.data), expected)


def test_take_along_accepts_index_dims_in_any_order(xp: Any) -> None:
    rng = np.random.default_rng(1)
    values = rng.random((5, 2, 4))
    cube = DimArray(xp.asarray(values), ("time", "band", "x"))
    index = rng.integers(0, 5, size=(4, 2))
    out = pa.take_along(cube, DimArray(xp.asarray(index), ("x", "band")), "time")
    assert out.dims == ("band", "x")
    expected = np.take_along_axis(values, index.T[None], axis=0)[0]
    np.testing.assert_allclose(np.asarray(out.data), expected)
    with pytest.raises(ValueError, match="not among"):
        pa.take_along(cube, DimArray(xp.asarray(index), ("x", "level")), "time")


def test_map_over(xp: Any) -> None:
    cube = _cube(xp)

    def band_sum(item: DimArray) -> DimArray:
        assert item.dims == ("band", "x")
        return pa.reduce(item, "sum", "band")

    out = pa.map_over(cube, band_sum, "time")
    assert out.dims == ("time", "x")
    np.testing.assert_allclose(np.asarray(out.data), np.asarray(cube.data).sum(1))
    # a non-leading dim comes back first
    by_band = pa.map_over(cube, lambda item: pa.reduce(item, "max", "x"), "band")
    assert by_band.dims == ("band", "time")


def test_stack_then_isel_gives_the_inputs_back(xp: Any) -> None:
    a = pa.isel(_cube(xp), time=0)
    b = pa.isel(_cube(xp), time=1)
    stacked = pa.stack([a, b], "time")
    assert stacked.dims == ("time", "band", "x")
    assert stacked.labels == {"band": BANDS}
    for i, original in enumerate((a, b)):
        back = pa.isel(stacked, time=i)
        assert back.dims == original.dims
        np.testing.assert_array_equal(np.asarray(back.data), np.asarray(original.data))
    with pytest.raises(ValueError, match="already a dim"):
        pa.stack([a, b], "band")
    with pytest.raises(ValueError, match="size"):
        pa.stack([a, pa.isel(a, x=slice(0, 2))], "time")


def test_concat_then_isel_gives_the_inputs_back(xp: Any) -> None:
    cube = _cube(xp)
    head = pa.isel(cube, time=slice(0, 1))
    tail = pa.isel(cube, time=slice(1, None))
    joined = pa.concat([head, tail], "time")
    assert joined.dims == cube.dims
    np.testing.assert_array_equal(np.asarray(joined.data), np.asarray(cube.data))
    back = pa.isel(joined, time=slice(1, None))
    np.testing.assert_array_equal(np.asarray(back.data), np.asarray(tail.data))
    relabelled = DimArray(np.asarray(head.data), head.dims, {"band": ("B11", "B12")})
    with pytest.raises(ValueError, match="labels"):
        pa.concat([pa.isel(_cube(np), time=slice(0, 1)), relabelled], "time")


def test_combine(xp: Any) -> None:
    nir = pa.sel(_cube(xp), band="B8")
    red = pa.sel(_cube(xp), band="B4")
    nd = pa.combine(lambda a, b: (a - b) / (a + b), nir, red)
    assert nd.dims == ("time", "x")
    expected = (np.asarray(nir.data) - np.asarray(red.data)) / (
        np.asarray(nir.data) + np.asarray(red.data)
    )
    np.testing.assert_allclose(np.asarray(nd.data), expected)
    # dims in another order are put in the first one's order by name
    flipped = pa.transpose(red, ("x", "time"))
    np.testing.assert_allclose(
        np.asarray(pa.combine(lambda a, b: a - b, nir, flipped).data),
        np.asarray(nir.data) - np.asarray(red.data),
    )
    with pytest.raises(ValueError, match="differ"):
        pa.combine(lambda a, b: a + b, nir, pa.isel(_cube(xp), band=0, x=0))


def test_raw_arrays_name_dimarray() -> None:
    with pytest.raises(TypeError, match="DimArray"):
        pa.reduce(np.zeros((2, 3)), "mean", "x")


def test_reduce_under_jit() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    cube = _cube(jnp)
    out = jax.jit(lambda c: pa.reduce(c, "mean", "time"))(cube)
    assert out.dims == ("band", "x")
    np.testing.assert_allclose(
        np.asarray(out.data), np.arange(24.0).reshape(3, 2, 4).mean(0)
    )


def test_dask_backed_dimarray_stays_lazy() -> None:
    da = pytest.importorskip("dask.array")
    lazy = DimArray(da.ones((3, 2, 4), chunks=(1, 2, 4)), ("time", "band", "x"))
    mean = pa.reduce(lazy, "mean", "time")
    assert type(mean.data).__module__.startswith("dask.")
    np.testing.assert_allclose(np.asarray(mean.data.compute()), np.ones((2, 4)))


# -- adapter fast paths --------------------------------------------------------


def test_an_adapter_method_overrides_the_generic_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = get_adapter(_cube())
    calls: list[tuple] = []

    def fast_reduce(x: DimArray, fn: Any, dims: tuple[str, ...]) -> str:
        calls.append((fn, dims))
        return "fast"

    monkeypatch.setattr(adapter, "reduce", fast_reduce, raising=False)
    assert pa.reduce(_cube(), "mean", "time") == "fast"
    assert calls == [("mean", ("time",))]


# -- operators ---------------------------------------------------------------


def _ops() -> list[tuple[Any, tuple[Any, ...]]]:
    cube = _cube()
    best = pa.argreduce(pa.sel(cube, band="B8"), "argmax", "time")
    return [
        (ISel(dim="time", index=0), (cube,)),
        (ISel(dim="x", index=slice(1, None, 2)), (cube,)),
        (Sel(dim="band", label="B8"), (cube,)),
        (Sel(dim="band", label=["B8", "B4"]), (cube,)),
        (Transpose(order=("x", "time", "band")), (cube,)),
        (Reduce("mean", dim="time"), (cube,)),
        (Reduce("max", dim=("time", "x")), (cube,)),
        (ArgReduce("argmax", dim="time"), (cube,)),
        (TakeAlong(dim="time"), (cube, best)),
        (Combine(lambda a: a * 2), (cube,)),
    ]


@pytest.mark.parametrize("case", range(10))
def test_operator_output_signature_equals_the_actual_one(case: int) -> None:
    op, args = _ops()[case]
    out = op(*args)
    predicted = op.compute_output_signature(_signature(args[0]))
    actual = _signature(out)
    assert predicted.dims == actual.dims, type(op).__name__
    if predicted.dtype is not None:
        assert predicted.dtype == actual.dtype


def test_map_over_operator() -> None:
    op = MapOver(lambda item: pa.reduce(item, "sum", "band"), dim="time")
    assert op(_cube()).dims == ("time", "x")
    assert op.forbid_in_yaml
    assert op.compute_output_signature(_signature(_cube())) is None


def test_reduce_takes_names_only() -> None:
    with pytest.raises(TypeError, match="name"):
        Reduce(np.mean, dim="time")  # type: ignore[arg-type]


def test_combinators_take_dim() -> None:
    a = pa.isel(_cube(), time=0)
    b = pa.isel(_cube(), time=1)
    stacked = StackAlong(dim="time")([a, b])
    assert stacked.dims == ("time", "band", "x")
    joined = ConcatenateAlong(dim="time")([stacked, stacked])
    assert joined.shape == (4, 2, 4)
    assert StackAlong(dim="time").get_config() == {"dim": "time"}
    assert StackAlong().get_config() == {"axis": 0}
    with pytest.raises(ValueError, match="not both"):
        StackAlong(axis=0, dim="time")


# -- review fixes ---------------------------------------------------------------------


def test_take_along_never_broadcasts_a_size_one_index_dim() -> None:
    cube = DimArray(np.arange(12.0).reshape(3, 4), ("time", "x"))
    index = DimArray(np.array([2]), ("x",))
    with pytest.raises(ValueError, match="never broadcast"):
        pa.take_along(cube, index, "time")


def test_sel_takes_an_array_of_labels_and_tuple_labels() -> None:
    scene = DimArray(
        np.arange(6.0).reshape(3, 2), ("band", "x"), {"band": ("a", "b", "c")}
    )
    picked = pa.sel(scene, band=np.array(["a", "c"]))
    assert picked.labels["band"] == ("a", "c")
    pairs = DimArray(
        np.arange(4.0), ("cell",), {"cell": ((0, 0), (0, 1), (1, 0), (1, 1))}
    )
    assert pa.sel(pairs, cell=(0, 1)).dims == ()
    assert pa.sel(pairs, cell=[(0, 1), (1, 1)]).shape == (2,)


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, ("x",)),
        ([0, 2], ("t", "x")),
        (np.array([1, 2]), ("t", "x")),
        (slice(1, 3), ("t", "x")),
    ],
)
def test_isel_operator_predicts_the_run(
    index: object, expected: tuple[str, ...]
) -> None:
    cube = DimArray(np.ones((4, 3)), ("t", "x"))
    op = pa.ISel(dim="t", index=index)
    sig = op.compute_output_signature(DimArrayAdapter().signature(cube))
    out = op(cube)
    assert out.dims == expected
    assert tuple(n for _, n in sig.dims) == out.shape


@pytest.mark.parametrize(
    ("fn", "dtype"),
    [
        ("all", np.float32),
        ("any", np.int8),
        ("sum", np.int8),
        ("sum", np.float32),
        ("max", np.int8),
        ("mean", np.float32),
        ("mean", np.int16),
        ("prod", np.bool_),
    ],
)
def test_reduce_operator_predicts_the_dtype(fn: str, dtype: type) -> None:
    cube = DimArray(np.ones((4, 3), dtype=dtype), ("t", "x"))
    op = pa.Reduce(fn, dim="t")
    predicted = op.compute_output_signature(DimArrayAdapter().signature(cube)).dtype
    actual = str(op(cube).data.dtype)
    assert predicted in (None, actual), (fn, dtype, predicted, actual)
    if fn in ("all", "any", "max") or "float" in str(np.dtype(dtype)):
        assert predicted == actual


# -- review fixes (Codex, #110) -------------------------------------------------------


def test_meanscalar_states_saved_before_the_rename_load() -> None:
    from pipekit import Operator

    state = pa.MeanScalar(axis=0).state
    state = {**state, "module": "pipekit_array.reduce"}
    assert isinstance(Operator.from_state(state), pa.MeanScalar)


def test_dask_generic_paths_stay_lazy() -> None:
    da = pytest.importorskip("dask.array")
    cube = DimArray(
        da.ones((3, 2, 4), chunks=1), ("time", "band", "x"), {"band": BANDS}
    )
    moved = pa.transpose(cube, ("x", "band", "time"))
    joined = pa.concat(
        [pa.isel(cube, time=slice(0, 1)), pa.isel(cube, time=slice(1, None))], "time"
    )
    best = DimArray(
        da.zeros((2, 4), dtype=int, chunks=1), ("band", "x"), {"band": BANDS}
    )
    picked = pa.take_along(cube, best, "time")
    for out in (moved, joined, picked):
        assert isinstance(out.data, da.Array)
    assert moved.shape == (4, 2, 3) and joined.shape == (3, 2, 4)
    values = np.random.default_rng(0).random((3, 2, 4))
    idx = np.random.default_rng(1).integers(0, 3, (2, 4))
    lazy = pa.take_along(
        DimArray(da.from_array(values, chunks=1), ("time", "band", "x")),
        DimArray(da.from_array(idx, chunks=1), ("band", "x")),
        "time",
    )
    eager = pa.take_along(
        DimArray(values, ("time", "band", "x")), DimArray(idx, ("band", "x")), "time"
    )
    np.testing.assert_array_equal(lazy.data.compute(), eager.data)


def test_concat_accepts_an_empty_part() -> None:
    cube = _cube()
    joined = pa.concat([pa.isel(cube, time=slice(0, 0)), cube], "time")
    np.testing.assert_array_equal(joined.data, cube.data)


def test_map_over_keeps_the_mapped_labels() -> None:
    cube = _cube()
    out = pa.map_over(cube, lambda item: item, "band")
    assert out.labels["band"] == BANDS
    assert pa.sel(out, band="B8").dims == ("time", "x")


def test_take_along_rejects_reordered_labels() -> None:
    data = DimArray(np.ones((3, 2)), ("time", "x"), {"x": ("left", "right")})
    index = DimArray(np.zeros(2, dtype=int), ("x",), {"x": ("right", "left")})
    with pytest.raises(ValueError, match="by position"):
        pa.take_along(data, index, "time")


def test_mixed_backends_never_stack_silently() -> None:
    jnp = pytest.importorskip("jax.numpy")
    a = DimArray(np.ones((2, 3)), ("y", "x"))
    b = DimArray(jnp.ones((2, 3)), ("y", "x"))
    for join in (
        lambda: pa.stack([a, b], "run"),
        lambda: pa.concat([a, b], "y"),
        lambda: pa.combine(np.add, a, b),
    ):
        with pytest.raises(pa.BackendError, match="to_jax"):
            join()


def test_sel_operator_keeps_a_tuple_label() -> None:
    pairs = DimArray(
        np.arange(4.0), ("cell",), {"cell": ((0, 0), (0, 1), (1, 0), (1, 1))}
    )
    assert Sel(dim="cell", label=(0, 1))(pairs).dims == ()
    assert Sel(dim="cell", label=[(0, 1), (1, 1)])(pairs).shape == (2,)


@pytest.mark.parametrize("label", [(0, 1), [(0, 1), (1, 1)], ["a", "b"], "a"])
def test_sel_tuple_labels_survive_dumps_loads(label: Any) -> None:
    """Codex #110: JSON turns tuples into lists; the config keeps them apart."""
    from pipekit import dumps, loads

    again = loads(dumps(Sel(dim="cell", label=label)))
    assert again.label == label
    assert type(again.label) is type(label)


def test_numpy_conversion_honours_copy() -> None:
    d = DimArray(np.zeros(3), ("x",))
    copied = np.array(d, copy=True)
    copied[0] = 1.0
    assert d.data[0] == 0.0
    with pytest.raises(ValueError):
        np.array(d, dtype=np.float32, copy=False)


def test_sel_with_a_label_array_keeps_the_dim_in_its_signature() -> None:
    """Codex #110: a 1-D label array is several labels, in the signature too."""
    from pipekit import dumps, loads

    cube = DimArray(np.zeros((3, 2)), ("band", "x"), {"band": ("a", "b", "c")})
    op = Sel(dim="band", label=np.array(["a", "c"]))
    sig = op.compute_output_signature(Signature(dims=(("band", 3), ("x", 2))))
    assert sig.dims == (("band", 2), ("x", 2))
    assert op(cube).dims == ("band", "x")
    assert loads(dumps(op)).label == ["a", "c"]


def test_take_along_refuses_a_jax_index_on_numpy_data() -> None:
    """Codex #110: mixed backends raise instead of converting the index."""
    jnp = pytest.importorskip("jax.numpy")
    x = DimArray(np.zeros((3, 2)), ("t", "x"))
    index = DimArray(jnp.zeros(2, dtype=jnp.int32), ("x",))
    with pytest.raises(pa.BackendError, match="take_along"):
        pa.take_along(x, index, "t")
