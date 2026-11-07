"""Tests for the phase-4 dim functions (`dim_algebra`) and their operators.

Gates of ``docs/design/api/dim_functions.md`` §7 for ``quantile`` and
``dot``: G1 xarray parity, G3 adapter agreement, G4 jit, G5 vmap, G8
dtype, G9 operators, G10 backends, G11 labels, G12 batched dot; and the
worked sketch of §4.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pipekit_array as pa
import pytest
from pipekit_array import DimArray, DimArrayAdapter, Dot, Quantile, dim_algebra
from pipekit_array.dim_algebra import METHODS
from pipekit_array.testing import check_adapter, values_of
from pipekit_ops import (
    dumps,
    get_adapter,
    loads,
    pipeline_hash,
    register_adapter,
    signature_of,
    unregister_adapter,
)
from pipekit_ops.adapters import BackendError
from pipekit_ops.testing import assert_round_trip


# Shapes: T = 12 scenes; B = 13 bands; Y = 4, X = 5 pixels; C = 3
# components (the running example of §2 at a small grid)
BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11")
BANDS = (*BANDS, "B12")
COMPONENTS = ("pc1", "pc2", "pc3")
DIMS = ("time", "band", "y", "x")
SHAPE = (12, 13, 4, 5)
BACKENDS = ["numpy", "jax", "dask"]


def _data(seed: int = 0, shape: tuple[int, ...] = SHAPE, dtype: Any = np.float32):
    rng = np.random.default_rng(seed)
    return rng.random(shape).astype(dtype)


def _on(backend: str, data: np.ndarray) -> Any:
    if backend == "jax":
        return pytest.importorskip("jax.numpy").asarray(data)
    if backend == "dask":
        # the leading dim over several chunks: quantile must rechunk it
        chunks = (4, *data.shape[1:])
        return pytest.importorskip("dask.array").from_array(data, chunks=chunks)
    return data


def _cube(backend: str = "numpy", seed: int = 0, dtype: Any = np.float32):
    """``time band y x`` = (12, 13, 4, 5), labelled bands."""
    return DimArray(_on(backend, _data(seed, dtype=dtype)), DIMS, {"band": BANDS})


def _weights(backend: str = "numpy", dtype: Any = np.float32) -> DimArray:
    """``band component`` = (13, 3), labelled components."""
    data = _on(backend, _data(7, (13, 3), dtype))
    return DimArray(data, ("band", "component"), {"component": COMPONENTS})


@contextlib.contextmanager
def _x64(backend: str) -> Iterator[None]:
    """float64 on JAX for the block; nothing elsewhere."""
    if backend != "jax":
        yield
        return
    jax = pytest.importorskip("jax")
    with jax.enable_x64(True):
        yield


def _labels(x: Any) -> dict[str, tuple[Any, ...]]:
    adapter = get_adapter(x)
    assert adapter is not None
    found = {d: adapter.labels(x, d) for d in adapter.dims(x)}
    return {d: tuple(v) for d, v in found.items() if v is not None}


def _dims(x: Any) -> tuple[str, ...]:
    adapter = get_adapter(x)
    assert adapter is not None
    return tuple(adapter.dims(x))


def _assert_close(actual: Any, expected: Any, rtol: float) -> None:
    """Same dims, labels and dtype; values within ``rtol``, NaN in place."""
    assert _dims(actual) == _dims(expected)
    assert _labels(actual) == _labels(expected)
    a, e = values_of(actual), values_of(expected)
    assert a.dtype == e.dtype
    np.testing.assert_allclose(a, e, rtol=rtol, atol=0)


#: each function on the running example, with static arguments
FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "quantile(list, time)": lambda c: pa.quantile(c, [0.1, 0.5, 0.9], "time"),
    "quantile(float, (time, y), skipna)": lambda c: pa.quantile(
        c, 0.3, ("time", "y"), method="midpoint", skipna=True
    ),
    "dot(band)": lambda c: pa.dot(
        c, DimArray(c.data[0, :, 0, :3], ("band", "component")), dim="band"
    ),
    "dot(x), batch over band y": lambda c: pa.dot(c, pa.isel(c, time=0), dim="x"),
}


# -- G1 xarray parity ------------------------------------------------------------


@pytest.fixture
def xr() -> Any:
    return pytest.importorskip("xarray")


def _with_nan(seed: int = 0) -> DimArray:
    """The cube in float64 with a few NaN, so skipna matters."""
    data = _data(seed, dtype=np.float64)
    data[0, 0, 0, 0] = np.nan
    data[3, 2, 1, :] = np.nan
    return DimArray(data, DIMS, {"band": BANDS})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("q", [0.3, [0.1, 0.5, 0.9]])
@pytest.mark.parametrize("dim", ["time", ("time", "y")])
def test_g1_quantile_equals_xarray(
    xr: Any, method: str, skipna: bool, q: Any, dim: Any
) -> None:
    cube = _with_nan()
    ours = pa.quantile(cube, q, dim, method=method, skipna=skipna)
    theirs = cube.to_xarray().quantile(q, dim=dim, method=method, skipna=skipna)
    assert _dims(ours) == tuple(map(str, theirs.dims))
    np.testing.assert_allclose(values_of(ours), theirs.values, rtol=1e-12, atol=0)
    if isinstance(q, list):
        assert ours.labels["quantile"] == tuple(theirs.indexes["quantile"])


def test_g1_dot_equals_xarray(xr: Any) -> None:
    cube = _cube(dtype=np.float64)
    weights = _weights(dtype=np.float64)
    ours = pa.dot(cube, weights, dim="band")
    theirs = xr.dot(cube.to_xarray(), weights.to_xarray(), dim="band")
    assert _dims(ours) == tuple(map(str, theirs.dims))
    np.testing.assert_allclose(values_of(ours), theirs.values, rtol=1e-12, atol=0)
    # several summed dims and three inputs
    third = DimArray(_data(3, (12, 3), np.float64), ("time", "component"))
    ours = pa.dot(cube, weights, third, dim=("band", "x", "component"))
    theirs = xr.dot(
        cube.to_xarray(),
        weights.to_xarray(),
        third.to_xarray(),
        dim=("band", "x", "component"),
    )
    assert _dims(ours) == tuple(map(str, theirs.dims))
    np.testing.assert_allclose(values_of(ours), theirs.values, rtol=1e-12, atol=0)


@pytest.mark.parametrize("seed", range(20))
def test_g1_dot_dim_order_equals_xarray(xr: Any, seed: int) -> None:
    # random inputs over six dims: batch dims, summed dims and dims of one
    # input all land in xr.dot's order, the order of first appearance
    rng = np.random.default_rng(seed)
    names = ("a", "b", "c", "d", "t", "k")
    sizes = dict(zip(names, (2, 3, 4, 2, 3, 2), strict=True))
    xs = []
    for _ in range(int(rng.integers(2, 4))):
        dims = tuple(rng.choice(names, int(rng.integers(1, 5)), replace=False))
        data = rng.random(tuple(sizes[d] for d in dims))
        xs.append(DimArray(data, tuple(map(str, dims))))
    union = list(dict.fromkeys(d for x in xs for d in x.dims))
    dim = tuple(rng.choice(union, int(rng.integers(0, len(union) + 1)), replace=False))
    dim = tuple(map(str, dim))
    ours = pa.dot(*xs, dim=dim)
    theirs = xr.dot(*(x.to_xarray() for x in xs), dim=list(dim))
    assert _dims(ours) == tuple(map(str, theirs.dims))
    np.testing.assert_allclose(values_of(ours), theirs.values, rtol=1e-12, atol=0)


# -- G3 adapter agreement ----------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Plain(DimArray):
    """A DimArray subclass whose adapter has no ``relabel``."""


class NoRelabelAdapter(DimArrayAdapter):
    """`DimArrayAdapter` without ``relabel``; keeps `Plain`."""

    #: ``relabel`` is detected by name: None hides the inherited one
    relabel = None  # type: ignore[assignment]

    def isel(self, x: Plain, indexers: dict[str, Any]) -> Plain:
        out = super().isel(x, indexers)
        return Plain(out.data, out.dims, out.labels)

    def rewrap(self, x: Plain, data: Any, dims: tuple[str, ...]) -> Plain:
        out = super().rewrap(x, data, dims)
        return Plain(out.data, out.dims, out.labels)


@pytest.fixture
def no_relabel() -> Iterator[NoRelabelAdapter]:
    adapter = NoRelabelAdapter()
    register_adapter(Plain, adapter)
    yield adapter
    unregister_adapter(Plain)


def test_g3_kit_passes_without_relabel(no_relabel: NoRelabelAdapter) -> None:
    # the cases that set no labels run on an adapter without relabel
    check_adapter(no_relabel, lambda data, dims, labels: Plain(data, dims, labels))


class WrongQuantile(DimArrayAdapter):
    """A quantile fast path that returns the median whatever ``q``."""

    def quantile(self, x: Plain, q: Any, dims: tuple[str, ...], **_: Any) -> Plain:
        out = pa.reduce(x, "median", dims)
        return Plain(out.data, out.dims, out.labels)

    def rewrap(self, x: Plain, data: Any, dims: tuple[str, ...]) -> Plain:
        out = super().rewrap(x, data, dims)
        return Plain(out.data, out.dims, out.labels)


def test_g3_kit_catches_a_wrong_quantile_fast_path() -> None:
    adapter = WrongQuantile()
    register_adapter(Plain, adapter)
    try:
        with pytest.raises(AssertionError, match=r"quantile\(0.25, \(time, y\)"):
            check_adapter(adapter, lambda data, dims, labels: Plain(data, dims, labels))
    finally:
        unregister_adapter(Plain)


# -- G4 jit, G5 vmap -------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g4_jit_equals_eager(name: str) -> None:
    jax = pytest.importorskip("jax")
    fn = FUNCTIONS[name]
    cube = _cube("jax")
    _assert_close(jax.jit(fn)(cube), fn(cube), rtol=1e-6)


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g5_vmap_equals_a_stack_of_calls(name: str) -> None:
    jax = pytest.importorskip("jax")
    fn = FUNCTIONS[name]
    cubes = [_cube("jax", seed) for seed in range(4)]
    batched = jax.vmap(fn)(pa.stack(cubes, "batch"))
    _assert_close(batched, pa.stack([fn(c) for c in cubes], "batch"), rtol=1e-6)


# -- G8 dtype, G10 backends ------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g8_g10_float32_on_every_backend(name: str, backend: str) -> None:
    fn = FUNCTIONS[name]
    out = fn(_cube(backend))
    assert out.dtype == np.float32
    _assert_close(out, fn(_cube()), rtol=1e-6)


def test_g10_quantile_stays_lazy_on_dask() -> None:
    cube = _cube("dask")
    out = pa.quantile(cube, [0.1, 0.9], "time")
    assert type(out.data).__module__.startswith("dask.")


def test_g10_mixed_backends_raise() -> None:
    with pytest.raises(BackendError, match="dot: inputs on"):
        pa.dot(_cube("numpy"), _weights("jax"), dim="band")


# -- G9 operators ----------------------------------------------------------------


@pytest.fixture(scope="module")
def running_example() -> DimArray:
    """§2's cube at full size: (12, 13, 256, 256) float32."""
    return DimArray(_data(shape=(12, 13, 256, 256)), DIMS, {"band": BANDS})


def _operators() -> list[tuple[Any, Callable[[DimArray], tuple[Any, ...]]]]:
    """Each twin, and how to make its inputs from the cube."""
    return [
        (Quantile([0.1, 0.5, 0.9], dim="time"), lambda c: (c,)),
        (
            Quantile(0.25, dim=("time", "x"), method="nearest", skipna=True),
            lambda c: (c,),
        ),
        (
            Dot(dim="band"),
            lambda c: (pa.merge_dims(c, pixel=("y", "x")), _weights()),
        ),
        (Dot(dim=("band", "x")), lambda c: (c, pa.isel(c, time=0))),
    ]


@pytest.mark.parametrize("case", range(4))
def test_g9_operators_round_trip_with_a_stable_hash(case: int) -> None:
    op, inputs = _operators()[case]
    back = assert_round_trip(op, *inputs(_cube()))
    rebuilt = loads(dumps(back))
    assert pipeline_hash(rebuilt) == pipeline_hash(op)
    assert dumps(rebuilt) == dumps(op)


@pytest.mark.parametrize("case", range(4))
def test_g9_signature_predicts_the_run(case: int, running_example: DimArray) -> None:
    op, inputs = _operators()[case]
    xs = inputs(running_example)
    out = op(*xs)
    sigs = tuple(signature_of(x) for x in xs)
    predicted = op.compute_output_signature(sigs if isinstance(op, Dot) else sigs[0])
    assert predicted == signature_of(out), type(op).__name__


def test_g9_q_round_trips_as_an_ordered_list() -> None:
    op = Quantile([0.9, 0.1], dim=("time", "y"))
    assert op.get_config()["q"] == [0.9, 0.1]
    assert loads(dumps(op)).q == (0.9, 0.1)
    assert pipeline_hash(op) != pipeline_hash(Quantile([0.1, 0.9], ("time", "y")))
    assert pipeline_hash(Quantile(0.5, "time")) != pipeline_hash(
        Quantile([0.5], "time")
    )


def test_g9_dot_needs_every_signature() -> None:
    sig = signature_of(_cube())
    assert Dot("band").compute_output_signature(sig) is None
    assert Dot("band").compute_output_signature((sig, None)) is None


# -- G11 labels ------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["dimarray", "xarray"])
def test_g11_labels(kind: str) -> None:
    cube: Any = _cube()
    weights: Any = _weights()
    if kind == "xarray":
        pytest.importorskip("xarray")
        cube, weights = cube.to_xarray(), weights.to_xarray()
    adapter = get_adapter(cube)
    assert adapter is not None
    spread = pa.quantile(cube, [0.1, 0.9], "time")
    assert tuple(adapter.labels(spread, "quantile")) == (0.1, 0.9)
    assert tuple(adapter.labels(spread, "band")) == BANDS
    scores = pa.dot(cube, weights, dim="band")
    assert tuple(adapter.labels(scores, "component")) == COMPONENTS


def test_g11_an_adapter_without_relabel_raises_on_a_sequence_q(
    no_relabel: NoRelabelAdapter,
) -> None:
    plain = Plain(_data(), DIMS, {"band": BANDS})
    with pytest.raises(TypeError, match=r"quantile: NoRelabelAdapter .*'quantile'"):
        pa.quantile(plain, [0.1, 0.9], "time")
    weights = Plain(_data(7, (13, 3)), ("band", "component"), {"component": COMPONENTS})
    with pytest.raises(TypeError, match=r"dot: NoRelabelAdapter .*'component'"):
        pa.dot(plain, weights, dim="band")
    # a float q sets no labels: the same data as the reference adapter
    reference = DimArray(_data(), DIMS, {"band": BANDS})
    _assert_close(
        pa.quantile(plain, 0.5, "time"), pa.quantile(reference, 0.5, "time"), 0.0
    )


def test_g11_a_reused_quantile_dim_takes_q_as_labels() -> None:
    # a reduced dim called quantile, of the new dim's size: rewrap would
    # copy its labels, relabel replaces them
    old = DimArray(np.zeros((2, 4)), ("quantile", "x"), {"quantile": ("a", "b")})
    out = pa.quantile(old, [0.2, 0.8], "quantile")
    assert out.dims == ("quantile", "x")
    assert out.labels == {"quantile": (0.2, 0.8)}


# -- G12 batched dot -------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_g12_batched_dot(backend: str, xr: Any) -> None:
    x_data = _data(1, (12, 13), np.float64)
    w_data = _data(2, (12, 13, 3), np.float64)
    with _x64(backend):
        x = DimArray(_on(backend, x_data), ("time", "band"))
        w = DimArray(_on(backend, w_data), ("time", "band", "component"))
        out = pa.dot(x, w, dim="band")
        assert out.dims == ("time", "component")
        assert values_of(out).dtype == np.float64
        # s_{tc} = ∑_b x_{tb} w_{tbc}
        loop = np.zeros((12, 3))
        for t in range(12):
            for c in range(3):
                loop[t, c] = sum(x_data[t, b] * w_data[t, b, c] for b in range(13))
        np.testing.assert_allclose(values_of(out), loop, rtol=1e-12, atol=0)
        theirs = xr.dot(
            xr.DataArray(x_data, dims=("time", "band")),
            xr.DataArray(w_data, dims=("time", "band", "component")),
            dim="band",
        )
        np.testing.assert_allclose(values_of(out), theirs.values, rtol=1e-12, atol=0)


# -- the no-einsum fallback ------------------------------------------------------


class NoEinsum:
    """An array namespace without ``einsum``, to force the fallback."""

    def __init__(self, xp: Any) -> None:
        self._xp = xp

    def __getattr__(self, name: str) -> Any:
        if name == "einsum":
            raise AttributeError(name)
        return getattr(self._xp, name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dot_fallback_equals_einsum(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _x64(backend):
        cube = _cube(backend, dtype=np.float64)
        weights = _weights(backend, dtype=np.float64)
        calls = {
            "pixels": lambda: pa.dot(cube, weights, dim="band"),
            "batch": lambda: pa.dot(cube, pa.isel(cube, time=0), dim=("x", "y")),
            "three": lambda: pa.dot(
                pa.isel(cube, y=0), weights, pa.isel(cube, band=0, x=0), dim="band"
            ),
            "total": lambda: pa.dot(cube, cube, dim=DIMS),
        }
        expected = {name: call() for name, call in calls.items()}
        original = dim_algebra.namespace_of
        monkeypatch.setattr(
            dim_algebra, "namespace_of", lambda data: NoEinsum(original(data))
        )
        for name, call in calls.items():
            actual = call()
            assert _dims(actual) == _dims(expected[name]), name
            assert _labels(actual) == _labels(expected[name]), name
            np.testing.assert_allclose(
                values_of(actual),
                values_of(expected[name]),
                rtol=1e-12,
                atol=0,
                err_msg=name,
            )


# -- worked sketch (§4) ----------------------------------------------------------


def test_worked_sketch_scores() -> None:
    clear = _cube(dtype=np.float64)
    weights = _weights(dtype=np.float64)
    # p = i·X + j
    pixels = pa.merge_dims(clear, pixel=("y", "x"))
    # s_{tcp} = ∑_b W_{bc} x̂_{tbp}
    scores = pa.dot(pixels, weights, dim="band")
    assert scores.dims == ("time", "pixel", "component")
    assert scores.labels == {"component": COMPONENTS}
    expected = np.einsum("tbp,bc->tpc", pixels.data, weights.data)
    np.testing.assert_allclose(scores.data, expected, rtol=1e-12, atol=0)
    score_map = pa.split_dims(scores, pixel={"y": 4, "x": 5})
    assert score_map.dims == ("time", "y", "x", "component")
    # Q_{qbij}, q ∈ {0.1, 0.9}
    spread = pa.quantile(clear, [0.1, 0.9], "time")
    assert spread.dims == ("quantile", "band", "y", "x")


# -- errors ----------------------------------------------------------------------


def test_errors_name_the_function_and_the_problem() -> None:
    cube = _cube()
    with pytest.raises(ValueError, match=r"'weibull' is not supported.*inverted_cdf"):
        pa.quantile(cube, 0.5, "time", method="weibull")
    with pytest.raises(ValueError, match="'weibull' is not supported"):
        Quantile(0.5, "time", method="weibull")
    with pytest.raises(ValueError, match=r"q must lie in \[0, 1\], got \[1.5\]"):
        pa.quantile(cube, [0.5, 1.5], "time")
    with pytest.raises(KeyError, match="quantile: unknown dim"):
        pa.quantile(cube, 0.5, "member")
    taken = pa.rename(cube, x="quantile")
    with pytest.raises(ValueError, match="'quantile' is already a dim"):
        pa.quantile(taken, [0.1, 0.9], "time")
    with pytest.raises(ValueError, match="dot needs at least one carrier"):
        pa.dot(dim="band")
    with pytest.raises(KeyError, match="dot: unknown dim"):
        pa.dot(cube, _weights(), dim="member")
    with pytest.raises(ValueError, match="size"):
        pa.dot(cube, DimArray(np.zeros((12, 3)), ("band", "c")), dim="band")
    relabelled = DimArray(np.zeros(13), ("band",), {"band": tuple("abcdefghijklm")})
    with pytest.raises(ValueError, match="labels"):
        pa.dot(cube, relabelled, dim="band")


def test_an_adapter_method_overrides_the_generic_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = get_adapter(_cube())
    calls: list[Any] = []
    monkeypatch.setattr(
        adapter,
        "quantile",
        lambda x, q, dims, **kw: calls.append((q, dims, kw)),
        raising=False,
    )
    pa.quantile(_cube(), [0.1, 0.9], ["time"], method="lower")
    assert calls == [((0.1, 0.9), ("time",), {"method": "lower", "skipna": False})]
