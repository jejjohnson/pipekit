"""Tests for the phase-3 dim functions (`dim_windows`) and their operators.

Gates of ``docs/design/api/dim_functions.md`` §7 for ``window``,
``coarsen``, ``ApplyAlong`` and ``BatchAlong``: G1 xarray parity, G2
window parity, G3 adapter agreement, G4 jit, G5 vmap, G7 closed forms,
G8 dtype, G9 operators, G10 backends; and `BatchAlong`'s closed forms.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pipekit_array as pa
import pytest
from pipekit_array import (
    ApplyAlong,
    BatchAlong,
    Coarsen,
    DimArray,
    MapOver,
    Reduce,
    Window,
)
from pipekit_array.testing import check_adapter, values_of
from pipekit_ops import (
    Operator,
    check_pickleable,
    dumps,
    get_adapter,
    loads,
    pipeline_hash,
    signature_of,
)
from pipekit_ops.testing import assert_round_trip


# Shapes: T = 12 scenes; B = 13 bands; Y = 4, X = 5 pixels (the running
# example of §2 at a small grid)
BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11")
BANDS = (*BANDS, "B12")
SCENES = tuple(range(100, 112))
DIMS = ("time", "band", "y", "x")
SHAPE = (12, 13, 4, 5)
BACKENDS = ["numpy", "jax", "dask"]


def _data(seed: int = 0, shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def _on(backend: str, data: np.ndarray) -> Any:
    if backend == "jax":
        return pytest.importorskip("jax.numpy").asarray(data)
    if backend == "dask":
        return pytest.importorskip("dask.array").from_array(data, chunks=(4, 13, 2, 5))
    return data


def _cube(backend: str = "numpy", seed: int = 0, data: Any = None) -> DimArray:
    """``time band y x`` = (12, 13, 4, 5), float32, labelled scenes and bands."""
    values = _data(seed) if data is None else data
    return DimArray(_on(backend, values), DIMS, {"time": SCENES, "band": BANDS})


def _labels(x: Any) -> dict[str, tuple[Any, ...]]:
    """Every labelled dim of a carrier, through its adapter."""
    adapter = get_adapter(x)
    assert adapter is not None
    found = {d: adapter.labels(x, d) for d in adapter.dims(x)}
    return {d: tuple(v) for d, v in found.items() if v is not None}


def _assert_same(actual: Any, expected: Any, *, rtol: float = 0.0) -> None:
    """Same dims, same labels, same dtype; identical values unless ``rtol``."""
    adapter, reference = get_adapter(actual), get_adapter(expected)
    assert adapter is not None and reference is not None
    assert tuple(adapter.dims(actual)) == tuple(reference.dims(expected))
    assert _labels(actual) == _labels(expected)
    a, e = values_of(actual), values_of(expected)
    assert a.dtype == e.dtype
    if rtol:
        np.testing.assert_allclose(a, e, rtol=rtol)
    else:
        np.testing.assert_array_equal(a, e)


#: each function on the running example, with static arguments
FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "window": lambda c: pa.window(c, time=3, stride=2),
    "window_band": lambda c: pa.window(c, window_dim="w", band=4, stride=3),
    "coarsen": lambda c: pa.coarsen(c, "mean", time=3),
    "coarsen_trim": lambda c: pa.coarsen(c, "max", boundary="trim", y=3, x=2),
}


# -- G1 xarray parity, G2 window parity ------------------------------------------


@pytest.fixture
def xr() -> Any:
    return pytest.importorskip("xarray")


@pytest.mark.parametrize(
    ("boundary", "windows"),
    [("exact", {"time": 3}), ("exact", {"time": 4, "y": 2}), ("trim", {"x": 2})],
)
def test_g1_coarsen_equals_xarray(xr: Any, boundary: str, windows: dict) -> None:
    cube = _cube(data=_data().astype(np.float64))
    ours = pa.coarsen(cube, "mean", boundary=boundary, **windows)
    theirs = cube.to_xarray().coarsen(windows, boundary=boundary).reduce(np.mean)
    assert ours.dims == tuple(map(str, theirs.dims))
    np.testing.assert_allclose(values_of(ours), theirs.values, rtol=1e-12)
    # the coarsened dims lose their labels; the others keep theirs
    assert "time" not in ours.labels or "time" not in windows
    assert ours.labels["band"] == BANDS


def test_g2_window_equals_rolling_construct(xr: Any) -> None:
    cube = _cube()
    theirs = cube.to_xarray().rolling(time=3).construct("window")
    theirs = theirs.isel(time=slice(2, None, 2))
    ours = pa.window(cube, time=3, stride=2)
    assert ours.dims == tuple(map(str, theirs.dims))
    np.testing.assert_array_equal(values_of(ours), theirs.values)
    assert ours.labels["time"] == tuple(theirs.indexes["time"]) == SCENES[2::2]
    assert ours.labels["band"] == tuple(theirs.indexes["band"])
    # the xarray carrier takes the fast path: the same DataArray, coords and all
    xr.testing.assert_identical(pa.window(cube.to_xarray(), time=3, stride=2), theirs)


def test_g2_running_example_has_five_windows() -> None:
    cube = DimArray(_data(shape=(12, 13, 64, 64)), DIMS, {"band": BANDS})
    windows = pa.window(cube, time=3, stride=2)
    # n = ⌊(12 - 3)/2⌋ + 1 = 5
    assert windows.shape == (5, 13, 64, 64, 3)
    assert windows.dtype == np.float32


# -- G3 adapter agreement --------------------------------------------------------


def test_g3_xarray_fast_path_stays_lazy_on_dask(xr: Any) -> None:
    da = pytest.importorskip("dask.array")
    from pipekit_array.xarray import XarrayAdapter

    def make(data: np.ndarray, dims: tuple[str, ...], labels: dict) -> Any:
        values = da.from_array(data, chunks=(1, 3, 4, 5))
        return xr.DataArray(
            values, dims=dims, coords={k: list(v) for k, v in labels.items()}
        )

    # every window and coarsen case of the kit, on lazy data
    check_adapter(XarrayAdapter(), make)
    lazy = make(_data(shape=(3, 3, 4, 5)), DIMS, {"band": ("B2", "B4", "B8")})
    assert isinstance(pa.window(lazy, time=2).data, da.Array)
    assert isinstance(pa.coarsen(lazy, "mean", y=2).data, da.Array)


# -- G4 jit, G5 vmap -------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g4_jit_equals_eager(name: str) -> None:
    jax = pytest.importorskip("jax")
    fn = FUNCTIONS[name]
    cube = _cube("jax")
    rtol = 1e-6 if name.startswith("coarsen") else 0.0
    _assert_same(jax.jit(fn)(cube), fn(cube), rtol=rtol)


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g5_vmap_equals_a_stack_of_calls(name: str) -> None:
    jax = pytest.importorskip("jax")
    fn = FUNCTIONS[name]
    cubes = [_cube("jax", seed) for seed in range(4)]
    batched = jax.vmap(fn)(pa.stack(cubes, "batch"))
    _assert_same(batched, pa.stack([fn(c) for c in cubes], "batch"))


# -- G7 closed forms -------------------------------------------------------------


def test_g7_coarsen_is_split_dims_then_reduce() -> None:
    cube = DimArray(_data(shape=(12, 13, 64, 64)), DIMS, {"band": BANDS})
    split = pa.split_dims(cube, time={"time": 4, "window": 3})
    _assert_same(pa.coarsen(cube, "mean", time=3), pa.reduce(split, "mean", "window"))


def test_window_entries_are_the_strided_positions() -> None:
    cube = _cube()
    windows = pa.window(cube, time=4, stride=3)
    # w_{n, r} = x_{n·s + r}, n < ⌊(12 - 4)/3⌋ + 1 = 3
    assert windows.shape[0] == 3
    for n in range(3):
        for r in range(4):
            np.testing.assert_array_equal(
                values_of(pa.isel(windows, time=n, window=r)),
                values_of(pa.isel(cube, time=n * 3 + r)),
            )


def test_window_then_reduce_is_rolling_mean_without_the_leading_nans(xr: Any) -> None:
    cube = _cube(data=_data().astype(np.float64))
    ours = pa.reduce(pa.window(cube, time=3), "mean", "window")
    theirs = cube.to_xarray().rolling(time=3).mean().isel(time=slice(2, None))
    np.testing.assert_allclose(values_of(ours), theirs.values, rtol=1e-12)


# -- G8 dtype, G10 backends ------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g8_g10_float32_on_every_backend(name: str, backend: str) -> None:
    fn = FUNCTIONS[name]
    out = fn(_cube(backend))
    assert out.dtype == np.float32
    rtol = 1e-6 if name.startswith("coarsen") else 0.0
    _assert_same(out, fn(_cube()), rtol=rtol)


@pytest.mark.parametrize("backend", BACKENDS)
def test_g8_window_keeps_integers(backend: str) -> None:
    counts = DimArray(_on_2d(backend, np.arange(12 * 3).reshape(12, 3)), ("time", "x"))
    windows = pa.window(counts, time=3)
    # JAX without x64 holds int32: the input's dtype, whichever it is
    assert values_of(windows).dtype == values_of(counts).dtype
    assert values_of(windows)[0, 0].tolist() == [0, 3, 6]


def test_g8_window_keeps_integers_on_xarray(xr: Any) -> None:
    from pipekit_array.xarray import from_xarray

    counts = xr.DataArray(np.arange(12 * 3).reshape(12, 3), dims=("time", "x"))
    for x in (counts, counts.astype(bool)):
        windows = pa.window(x, time=3, stride=2)
        assert windows.dtype == x.dtype
        np.testing.assert_array_equal(
            windows.values, pa.window(from_xarray(x), time=3, stride=2).data
        )


def _on_2d(backend: str, data: np.ndarray) -> Any:
    if backend == "dask":
        return pytest.importorskip("dask.array").from_array(data, chunks=(4, 3))
    return _on(backend, data)


# -- G9 operators ----------------------------------------------------------------


@pytest.fixture(scope="module")
def running_example() -> DimArray:
    """§2's cube with a 64-by-64 grid: (12, 13, 64, 64) float32."""
    return DimArray(_data(shape=(12, 13, 64, 64)), DIMS, {"band": BANDS})


def _operators() -> list[Operator]:
    return [
        Window(stride=2, time=3),
        Window(window_dim="w", stride=1, x=5),
        Coarsen("mean", time=3),
        Coarsen("nanmax", boundary="trim", y=5, x=3),
        ApplyAlong(Reduce("mean", dim="band"), dim="time"),
        BatchAlong(Reduce("mean", dim="band"), size=5, dim="time", pad="last"),
        BatchAlong(Reduce("sum", dim="x"), size=5, dim="time", pad="drop"),
    ]


@pytest.mark.parametrize("case", range(7))
def test_g9_operators_round_trip_with_a_stable_hash(case: int) -> None:
    op = _operators()[case]
    back = assert_round_trip(op, _cube())
    rebuilt = loads(dumps(back))
    assert pipeline_hash(rebuilt) == pipeline_hash(op)
    assert dumps(rebuilt) == dumps(op)


@pytest.mark.parametrize("case", range(7))
def test_g9_signature_predicts_the_run(case: int, running_example: DimArray) -> None:
    op = _operators()[case]
    predicted = op.compute_output_signature(signature_of(running_example))
    assert predicted == signature_of(op(running_example)), repr(op)


def test_g9_apply_along_serialises_only_when_inner_does() -> None:
    assert not ApplyAlong.forbid_in_yaml
    assert not BatchAlong.forbid_in_yaml
    closure = MapOver(lambda item: item, dim="x")
    for op in (ApplyAlong(closure, "time"), BatchAlong(closure, 4, "time")):
        # the walker finds the flagged child, and dumps refuses it, as for
        # a Sequential holding a closure
        assert check_pickleable(op) == [closure]
        with pytest.raises(TypeError, match="not JSON serializable"):
            dumps(op)


# -- ApplyAlong ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_apply_along_equals_map_over(backend: str) -> None:
    cube = _cube(backend)
    inner = Reduce("max", dim="band")
    _assert_same(ApplyAlong(inner, "time")(cube), pa.map_over(cube, inner, "time"))


# -- BatchAlong closed forms -----------------------------------------------------


@pytest.mark.parametrize("pad", ["none", "last"])
@pytest.mark.parametrize("size", [3, 5, 12, 20])
@pytest.mark.parametrize("backend", BACKENDS)
def test_batch_along_equals_inner_on_the_whole(
    backend: str, size: int, pad: str
) -> None:
    cube = _cube(backend)
    inner = Reduce("mean", dim="band")
    _assert_same(BatchAlong(inner, size, "time", pad=pad)(cube), inner(cube))


def test_batch_along_drop_leaves_out_the_short_chunk() -> None:
    cube = _cube()
    inner = Reduce("mean", dim="band")
    dropped = BatchAlong(inner, 5, "time", pad="drop")(cube)
    _assert_same(dropped, inner(pa.isel(cube, time=slice(0, 10))))
    with pytest.raises(ValueError, match="drops every entry"):
        BatchAlong(inner, 20, "time", pad="drop")(cube)


def test_batch_along_on_xarray(xr: Any) -> None:
    cube = _cube().to_xarray()
    inner = Reduce("mean", dim="band")
    out = BatchAlong(inner, 5, "time", pad="last")(cube)
    xr.testing.assert_identical(out, inner(cube))


class _CountingMean(Operator):
    """A jitted band mean that counts its traces."""

    forbid_in_yaml = True

    def __init__(self) -> None:
        import jax

        self.traces = 0
        self._run = jax.jit(self._mean)

    def _mean(self, x: DimArray) -> DimArray:
        self.traces += 1
        return pa.reduce(x, "mean", "band")

    def _apply(self, x: DimArray) -> DimArray:
        return self._run(x)


@pytest.mark.parametrize(("pad", "traces"), [("last", 1), ("none", 2)])
def test_batch_along_pad_last_compiles_once(pad: str, traces: int) -> None:
    pytest.importorskip("jax")
    cube = DimArray(_on("jax", _data()), DIMS, {"band": BANDS})
    inner = _CountingMean()
    out = BatchAlong(inner, 5, "time", pad=pad)(cube)
    # chunks of 5, 5 and 2: "last" pads the 2 to 5, so one shape is traced
    assert inner.traces == traces
    _assert_same(out, pa.reduce(cube, "mean", "band"), rtol=1e-6)


# -- errors and fast paths -------------------------------------------------------


def test_errors_name_the_function_and_the_dim() -> None:
    cube = _cube()
    with pytest.raises(ValueError, match="give one dim=size"):
        pa.window(cube, time=3, x=2)
    with pytest.raises(KeyError, match="window: unknown dim"):
        pa.window(cube, pixel=3)
    with pytest.raises(ValueError, match="already a dim"):
        pa.window(cube, window_dim="band", time=3)
    with pytest.raises(ValueError, match="must be positive"):
        pa.window(cube, time=3, stride=0)
    with pytest.raises(ValueError, match="does not fit 'time' of size 12"):
        pa.window(cube, time=13)
    with pytest.raises(ValueError, match="5 does not divide 'time' of size 12"):
        pa.coarsen(cube, "mean", time=5)
    with pytest.raises(ValueError, match="boundary must be one of"):
        pa.coarsen(cube, "mean", boundary="pad", time=3)
    with pytest.raises(ValueError, match="time=0 must be positive"):
        pa.coarsen(cube, "mean", time=0)
    with pytest.raises(ValueError, match="at least one"):
        pa.coarsen(cube, "mean")
    with pytest.raises(ValueError, match="give one dim=size"):
        Window(time=3, x=2)
    with pytest.raises(TypeError, match="reduction's name"):
        Coarsen(np.mean, time=3)
    with pytest.raises(TypeError, match=r"ApplyAlong\.inner must be an Operator"):
        ApplyAlong(np.mean, "time")
    with pytest.raises(TypeError, match=r"BatchAlong\.inner must be an Operator"):
        BatchAlong(np.mean, 4, "time")
    with pytest.raises(ValueError, match="size must be positive"):
        BatchAlong(Reduce("mean", "band"), 0, "time")
    with pytest.raises(ValueError, match="pad must be one of"):
        BatchAlong(Reduce("mean", "band"), 4, "time", pad="first")


def test_coarsen_block_names_avoid_existing_dims() -> None:
    cube = DimArray(np.arange(8.0).reshape(4, 2), ("x", "x_block"))
    out = pa.coarsen(cube, "sum", x=2, x_block=2)
    assert out.dims == ("x", "x_block")
    assert out.data.tolist() == [[1.0 + 2.0 + 3.0 + 0.0], [4.0 + 5.0 + 6.0 + 7.0]]


def test_an_adapter_method_overrides_the_generic_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = get_adapter(_cube())
    calls: list[Any] = []

    def fast(x: Any, dim: str, size: int, *, window_dim: str, stride: int) -> None:
        calls.append((dim, size, window_dim, stride))

    monkeypatch.setattr(adapter, "window", fast, raising=False)
    pa.window(_cube(), time=3)
    assert calls == [("time", 3, "window", 1)]
