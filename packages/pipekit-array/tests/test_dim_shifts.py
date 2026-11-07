"""Tests for the phase-2 dim functions (`dim_shifts`) and their operators.

Gates of ``docs/design/api/dim_functions.md`` §7 for ``pad``, ``roll``,
``shift``, ``diff``, ``cumreduce`` and ``where``: G1 xarray parity, G3
adapter agreement, G4 jit, G5 vmap, G6 round trips, G7 closed forms, G8
dtype, G9 operators, G10 backends.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pipekit_array as pa
import pytest
from pipekit_array import (
    CumReduce,
    Diff,
    DimArray,
    DimArrayAdapter,
    Pad,
    Roll,
    Shift,
    Where,
)
from pipekit_array.testing import check_adapter, values_of
from pipekit_ops import dumps, get_adapter, loads, pipeline_hash, signature_of
from pipekit_ops.testing import assert_round_trip


# Shapes: T = 12 scenes; B = 13 bands; Y = 4, X = 5 pixels (the running
# example of §2 at a small grid)
BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11")
BANDS = (*BANDS, "B12")
DIMS = ("time", "band", "y", "x")
SHAPE = (12, 13, 4, 5)
T = SHAPE[0]


def _data(seed: int = 0, shape: tuple[int, ...] = SHAPE, dtype: Any = np.float32):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=dtype)


def _on(backend: str, data: np.ndarray, time_chunk: int = 4) -> Any:
    if backend == "jax":
        return pytest.importorskip("jax.numpy").asarray(data)
    if backend == "dask":
        dask_array = pytest.importorskip("dask.array")
        return dask_array.from_array(data, chunks=(time_chunk, 13, 2, 5))
    return data


def _cube(backend: str = "numpy", seed: int = 0, dtype: Any = np.float32) -> DimArray:
    """``time band y x`` = (12, 13, 4, 5), labelled bands."""
    return DimArray(_on(backend, _data(seed, dtype=dtype)), DIMS, {"band": BANDS})


def _valid(shape: tuple[int, ...] = (12, 4, 5)) -> DimArray:
    """A ``time y x`` cloud mask v, True = valid, about two thirds True."""
    return DimArray(
        np.arange(np.prod(shape)).reshape(shape) % 3 > 0, ("time", "y", "x")
    )


def _labels(x: Any) -> dict[str, tuple[Any, ...]]:
    """Every labelled dim of a carrier, through its adapter."""
    adapter = get_adapter(x)
    assert adapter is not None
    found = {d: adapter.labels(x, d) for d in adapter.dims(x)}
    return {d: tuple(v) for d, v in found.items() if v is not None}


def _assert_same(actual: Any, expected: Any, rtol: float | None = None) -> None:
    """Same dims, labels and dtype; identical values (NaN included) or within rtol."""
    adapter, reference = get_adapter(actual), get_adapter(expected)
    assert adapter is not None and reference is not None
    assert tuple(adapter.dims(actual)) == tuple(reference.dims(expected))
    assert _labels(actual) == _labels(expected)
    a, e = values_of(actual), values_of(expected)
    assert a.dtype == e.dtype
    if rtol is None:
        np.testing.assert_array_equal(a, e)
    else:
        np.testing.assert_allclose(a, e, rtol=rtol)


#: each function on the running example, with static arguments
FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "pad_constant": lambda c: pa.pad(c, time=(2, 0), x=1),
    "pad_edge": lambda c: pa.pad(c, mode="edge", y=(0, 3)),
    "pad_reflect": lambda c: pa.pad(c, mode="reflect", band=2),
    "pad_wrap": lambda c: pa.pad(c, mode="wrap", time=(1, 4)),
    "roll": lambda c: pa.roll(c, time=5, x=-2),
    "shift": lambda c: pa.shift(c, time=1, band=-3),
    "diff": lambda c: pa.diff(c, "time", 2, label="lower"),
    "cumreduce": lambda c: pa.cumreduce(c, "cumulative_sum", "time"),
    "where": lambda c: pa.where(c, _valid()),
}

#: the functions whose values are compared within a tolerance, not exactly
INEXACT = {"cumreduce"}


# -- G1 xarray parity -----------------------------------------------------------


@pytest.fixture
def xr() -> Any:
    return pytest.importorskip("xarray")


def _assert_matches_xarray(ours: Any, theirs: Any, rtol: float | None = None) -> None:
    assert tuple(get_adapter(ours).dims(ours)) == tuple(map(str, theirs.dims))
    if rtol is None:
        # NaN fills in the same places
        np.testing.assert_array_equal(values_of(ours), theirs.values)
    else:
        np.testing.assert_allclose(values_of(ours), theirs.values, rtol=rtol)


@pytest.mark.parametrize("mode", ["constant", "edge", "reflect", "wrap"])
def test_g1_pad_equals_xarray(xr: Any, mode: str) -> None:
    cube = _cube()
    ours = pa.pad(cube, mode=mode, time=(2, 0), band=3)
    theirs = cube.to_xarray().pad(time=(2, 0), band=3, mode=mode)
    _assert_matches_xarray(ours, theirs)
    assert values_of(ours).dtype == theirs.dtype
    # a padded dim loses its labels; xarray pads its coordinate with NaN
    assert "band" not in _labels(ours)


def test_g1_pad_constant_values_equals_xarray(xr: Any) -> None:
    cube = _cube()
    ours = pa.pad(cube, constant_values=-1.0, x=(1, 2))
    theirs = cube.to_xarray().pad(x=(1, 2), constant_values=-1.0)
    _assert_matches_xarray(ours, theirs)


def test_g1_roll_equals_xarray(xr: Any) -> None:
    cube = _cube()
    ours = pa.roll(cube, time=5, band=-2)
    theirs = cube.to_xarray().roll(time=5, band=-2)
    _assert_matches_xarray(ours, theirs)
    # labels stay on their positions (roll_coords=False)
    assert ours.labels["band"] == tuple(theirs.indexes["band"]) == BANDS


@pytest.mark.parametrize("shifts", [{"time": 1}, {"time": -3, "x": 2}, {"band": 13}])
def test_g1_shift_equals_xarray(xr: Any, shifts: dict[str, int]) -> None:
    cube = _cube()
    ours = pa.shift(cube, **shifts)
    theirs = cube.to_xarray().shift(**shifts)
    _assert_matches_xarray(ours, theirs)
    assert values_of(ours).dtype == theirs.dtype == np.float32
    assert ours.labels["band"] == tuple(theirs.indexes["band"])


def test_g1_shift_fill_value_equals_xarray(xr: Any) -> None:
    cube = _cube()
    ours = pa.shift(cube, fill_value=0.5, time=2)
    theirs = cube.to_xarray().shift(time=2, fill_value=0.5)
    _assert_matches_xarray(ours, theirs)


@pytest.mark.parametrize("label", ["upper", "lower"])
@pytest.mark.parametrize("n", [0, 1, 2])
def test_g1_diff_equals_xarray(xr: Any, n: int, label: str) -> None:
    scenes = tuple(f"t{k}" for k in range(T))
    cube = DimArray(_data(), DIMS, {"time": scenes, "band": BANDS})
    ours = pa.diff(cube, "time", n, label=label)
    theirs = cube.to_xarray().diff("time", n=n, label=label)
    _assert_matches_xarray(ours, theirs)
    # "lower" keeps the subtrahend's labels at every step: positions
    # 0 … N-n-1; xarray's recursion drops label after its first step
    expected = scenes[n:] if label == "upper" else scenes[: T - n]
    assert ours.labels["time"] == expected
    if n < 2 or label == "upper":
        assert ours.labels["time"] == tuple(theirs.indexes["time"])
    assert ours.labels["band"] == BANDS


@pytest.mark.parametrize(
    ("fn", "method"), [("cumulative_sum", "cumsum"), ("cumulative_prod", "cumprod")]
)
def test_g1_cumreduce_equals_xarray(xr: Any, fn: str, method: str) -> None:
    cube = _cube(dtype=np.float64)
    ours = pa.cumreduce(cube, fn, "time")
    theirs = getattr(cube.to_xarray(), method)("time")
    _assert_matches_xarray(ours, theirs, rtol=1e-12)


def test_g1_nancumsum_equals_xarray_cumsum_with_skipna(xr: Any) -> None:
    data = _data(dtype=np.float64)
    data[data < 0.2] = np.nan
    cube = DimArray(data, DIMS, {"band": BANDS})
    ours = pa.cumreduce(cube, "nancumsum", "time")
    theirs = cube.to_xarray().cumsum("time", skipna=True)
    _assert_matches_xarray(ours, theirs, rtol=1e-12)
    # no implicit skipna: cumulative_sum propagates NaN
    assert np.isnan(values_of(pa.cumreduce(cube, "cumulative_sum", "time"))).any()


@pytest.mark.parametrize("other", [None, 0.0])
def test_g1_where_equals_xarray(xr: Any, other: float | None) -> None:
    cube, valid = _cube(), _valid()
    ours = pa.where(cube, valid, other)
    xa, cond = cube.to_xarray(), valid.to_xarray()
    theirs = xa.where(cond) if other is None else xa.where(cond, other)
    _assert_matches_xarray(ours, theirs)
    assert values_of(ours).dtype == theirs.dtype == np.float32


# -- G3 adapter agreement --------------------------------------------------------


def test_g3_kit_passes_on_dimarray_dask() -> None:
    dask_array = pytest.importorskip("dask.array")

    def make(data: np.ndarray, dims: tuple[str, ...], labels: dict) -> DimArray:
        return DimArray(dask_array.from_array(data, chunks=(1, 3, 2, 5)), dims, labels)

    check_adapter(DimArrayAdapter(), make)


# -- G4 jit, G5 vmap -------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g4_jit_equals_eager(name: str) -> None:
    jax = pytest.importorskip("jax")
    fn = FUNCTIONS[name]
    cube = _cube("jax")
    _assert_same(jax.jit(fn)(cube), fn(cube))


def test_g4_roll_takes_a_traced_shift() -> None:
    jax = pytest.importorskip("jax")
    cube = _cube("jax")
    rolled = jax.jit(lambda c, k: pa.roll(c, time=k))(cube, 5)
    _assert_same(rolled, pa.roll(cube, time=5))


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g5_vmap_equals_a_stack_of_calls(name: str) -> None:
    jax = pytest.importorskip("jax")
    fn = FUNCTIONS[name]
    cubes = [_cube("jax", seed) for seed in range(4)]
    batched = jax.vmap(fn)(pa.stack(cubes, "batch"))
    _assert_same(batched, pa.stack([fn(c) for c in cubes], "batch"))


# -- G6 round trips, G7 closed forms ---------------------------------------------


BACKENDS = ["numpy", "jax", "dask"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("k", [1, 5, 12])
def test_g6_roll_back_is_the_identity(backend: str, k: int) -> None:
    cube = _cube(backend)
    _assert_same(pa.roll(pa.roll(cube, time=k), time=-k), cube)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("k", [1, 5, 11])
def test_g7_shift_moves_the_data_by_k(backend: str, k: int) -> None:
    cube = _cube(backend)
    moved = pa.isel(pa.shift(cube, time=k), time=slice(k, None))
    _assert_same(moved, pa.isel(cube, time=slice(None, -k)))
    vacated = values_of(pa.isel(pa.shift(cube, time=k), time=slice(None, k)))
    assert np.isnan(vacated).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_g7_cumulative_sum_of_diff_telescopes(backend: str) -> None:
    jax = pytest.importorskip("jax") if backend == "jax" else None
    previous = jax.config.jax_enable_x64 if jax else None
    if jax:
        # the float64 oracle needs float64 on JAX too
        jax.config.update("jax_enable_x64", True)
    try:
        cube = _cube(backend, dtype=np.float64)
        drift = pa.cumreduce(pa.diff(cube, "time"), "cumulative_sum", "time")
    finally:
        if jax:
            jax.config.update("jax_enable_x64", previous)
    assert drift.dtype == np.float64
    # c_{T-2} = ∑_{t < T-1} (x_{t+1} - x_t) = x_{T-1} - x_0
    last = values_of(pa.isel(drift, time=-1))
    data = values_of(cube)
    np.testing.assert_allclose(last, data[T - 1] - data[0], rtol=1e-12)


# -- G8 dtype, G10 backends ------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g8_g10_float32_on_every_backend(name: str, backend: str) -> None:
    fn = FUNCTIONS[name]
    out = fn(_cube(backend))
    assert out.dtype == np.float32
    # dask accumulates chunk by chunk: the sums associate differently
    rtol = 1e-6 if name in INEXACT and backend == "dask" else None
    _assert_same(out, fn(_cube()), rtol=rtol)


@pytest.mark.parametrize("fn", ["cumulative_sum", "cumulative_prod", "nancumsum"])
def test_g10_cumreduce_on_dask_equals_numpy(fn: str) -> None:
    # time in one chunk: dask runs NumPy's accumulation on each block
    cube = DimArray(_on("dask", _data(dtype=np.float64), time_chunk=T), DIMS)
    reference = DimArray(_data(dtype=np.float64), DIMS)
    _assert_same(pa.cumreduce(cube, fn, "time"), pa.cumreduce(reference, fn, "time"))
    # across chunks of time, within float64 rounding
    chunked = DimArray(_on("dask", _data(dtype=np.float64)), DIMS)
    _assert_same(
        pa.cumreduce(chunked, fn, "time"),
        pa.cumreduce(reference, fn, "time"),
        rtol=1e-12,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_g8_integers_need_a_fill(backend: str) -> None:
    counts = DimArray(_on(backend, np.arange(np.prod(SHAPE)).reshape(SHAPE)), DIMS)
    with pytest.raises(TypeError, match=r"shift: int\d+ data has no NaN"):
        pa.shift(counts, time=1)
    with pytest.raises(TypeError, match=r"pad: .* pass constant_values="):
        pa.pad(counts, time=1)
    with pytest.raises(TypeError, match=r"where: .* pass other="):
        pa.where(counts, _valid())
    # with a fill, the dtype is kept
    assert pa.shift(counts, fill_value=-1, time=1).dtype == counts.dtype
    assert pa.pad(counts, constant_values=0, time=1).dtype == counts.dtype
    assert pa.pad(counts, mode="edge", time=1).dtype == counts.dtype
    assert pa.where(counts, _valid(), -1).dtype == counts.dtype


# -- G9 operators ----------------------------------------------------------------


@pytest.fixture(scope="module")
def running_example() -> DimArray:
    """§2's cube at full size: (12, 13, 256, 256) float32."""
    data = _data(shape=(12, 13, 256, 256))
    return DimArray(data, DIMS, {"band": BANDS})


def _operators(grid: int) -> list[tuple[Any, tuple[Any, ...]]]:
    """Each twin, and its inputs after the cube."""
    return [
        (Pad(mode="reflect", time=(2, 0), x=3), ()),
        (Pad(constant_values=0.0, band=1), ()),
        (Roll(time=5, x=-2), ()),
        (Shift(fill_value=0.0, time=1, band=-3), ()),
        (Diff("time", n=2, label="lower"), ()),
        (CumReduce("cumulative_sum", "time"), ()),
        (Where(other=-1.0), (_valid((12, grid, grid)),)),
    ]


@pytest.mark.parametrize("case", range(7))
def test_g9_operators_round_trip_with_a_stable_hash(case: int) -> None:
    cube = DimArray(_data(shape=(12, 13, 4, 4)), DIMS, {"band": BANDS})
    op, others = _operators(4)[case]
    back = assert_round_trip(op, cube, *others)
    rebuilt = loads(dumps(back))
    assert pipeline_hash(rebuilt) == pipeline_hash(op)
    assert dumps(rebuilt) == dumps(op)


@pytest.mark.parametrize("case", range(7))
def test_g9_signature_predicts_the_run(case: int, running_example: DimArray) -> None:
    op, others = _operators(256)[case]
    out = op(running_example, *others)
    predicted = op.compute_output_signature(signature_of(running_example))
    assert predicted == signature_of(out), type(op).__name__
    if others:
        pair = (signature_of(running_example), signature_of(others[0]))
        assert op.compute_output_signature(pair) == signature_of(out)


def test_g9_cumreduce_takes_a_name() -> None:
    with pytest.raises(TypeError, match="function's name"):
        CumReduce(np.cumsum, "time")  # type: ignore[arg-type]


# -- labels, errors, fast paths --------------------------------------------------


def test_labels_stay_on_their_positions() -> None:
    cube = _cube()
    assert pa.roll(cube, band=3).labels == {"band": BANDS}
    assert pa.shift(cube, band=3).labels == {"band": BANDS}
    assert pa.cumreduce(cube, "cumulative_sum", "band").labels == {"band": BANDS}
    assert pa.where(cube, _valid()).labels == {"band": BANDS}
    assert pa.pad(cube, band=1).labels == {}
    assert pa.diff(cube, "band").labels == {"band": BANDS[1:]}


def test_errors_name_the_function_and_the_dim() -> None:
    cube = _cube()
    with pytest.raises(KeyError, match="pad: unknown dim"):
        pa.pad(cube, pixel=1)
    with pytest.raises(ValueError, match="mode 'symmetric' is not one of"):
        pa.pad(cube, mode="symmetric", time=1)
    with pytest.raises(ValueError, match="constant_values is for mode='constant'"):
        pa.pad(cube, mode="edge", constant_values=0.0, time=1)
    with pytest.raises(ValueError, match="width ≥ 0 or a pair"):
        pa.pad(cube, time=(1, -1))
    with pytest.raises(KeyError, match="roll: unknown dim"):
        pa.roll(cube, pixel=1)
    with pytest.raises(KeyError, match="shift: unknown dim"):
        pa.shift(cube, pixel=1)
    with pytest.raises(ValueError, match="order n must be ≥ 0"):
        pa.diff(cube, "time", -1)
    with pytest.raises(ValueError, match="label must be 'upper' or 'lower'"):
        pa.diff(cube, "time", label="middle")
    with pytest.raises(ValueError, match="no 'cummean'"):
        pa.cumreduce(cube, "cummean", "time")
    with pytest.raises(ValueError, match="keeps the length of 'time'"):
        pa.cumreduce(cube, lambda d, axis: d.sum(axis=axis), "time")
    with pytest.raises(ValueError, match=r"where: where= has dim\(s\) \['member'\]"):
        pa.where(cube, DimArray(np.ones((2,), dtype=bool), ("member",)))


def test_an_adapter_method_overrides_the_generic_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = get_adapter(_cube())
    calls: list[Any] = []
    monkeypatch.setattr(
        adapter,
        "shift",
        lambda x, shifts, fill_value: calls.append((shifts, fill_value)),
        raising=False,
    )
    pa.shift(_cube(), fill_value=0.0, time=1)
    assert calls == [({"time": 1}, 0.0)]
