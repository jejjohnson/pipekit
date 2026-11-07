"""Tests for the phase-1 dim functions (`dim_shapes`) and their operators.

Gates of ``docs/design/api/dim_functions.md`` §7 for ``rename``,
``expand_dims``, ``squeeze``, ``broadcast_like``, ``merge_dims`` and
``split_dims``: G1 xarray parity, G3 adapter agreement, G4 jit, G5 vmap,
G6 round trips, G8 dtype, G9 operators, G10 backends, G11 labels.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pipekit_array as pa
import pytest
from pipekit_array import (
    BroadcastLike,
    DimArray,
    DimArrayAdapter,
    ExpandDims,
    MergeDims,
    Rename,
    SplitDims,
    Squeeze,
)
from pipekit_array.testing import check_adapter, values_of
from pipekit_ops import (
    Signature,
    dumps,
    get_adapter,
    loads,
    pipeline_hash,
    register_adapter,
    signature_of,
    unregister_adapter,
)
from pipekit_ops.testing import assert_round_trip


# Shapes: T = 12 scenes; B = 13 bands; Y = 4, X = 5 pixels (the running
# example of §2 at a small grid)
BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11")
BANDS = (*BANDS, "B12")
DIMS = ("time", "band", "y", "x")
SHAPE = (12, 13, 4, 5)


def _data(seed: int = 0, shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def _on(backend: str, data: np.ndarray) -> Any:
    if backend == "jax":
        return pytest.importorskip("jax.numpy").asarray(data)
    if backend == "dask":
        return pytest.importorskip("dask.array").from_array(data, chunks=(4, 13, 2, 5))
    return data


def _cube(backend: str = "numpy", seed: int = 0) -> DimArray:
    """``time band y x`` = (12, 13, 4, 5), float32, labelled bands."""
    return DimArray(_on(backend, _data(seed)), DIMS, {"band": BANDS})


def _labels(x: Any) -> dict[str, tuple[Any, ...]]:
    """Every labelled dim of a carrier, through its adapter."""
    adapter = get_adapter(x)
    assert adapter is not None
    found = {d: adapter.labels(x, d) for d in adapter.dims(x)}
    return {d: tuple(v) for d, v in found.items() if v is not None}


def _assert_same(actual: Any, expected: Any) -> None:
    """Same dims, same labels, identical values and dtype."""
    adapter, reference = get_adapter(actual), get_adapter(expected)
    assert adapter is not None and reference is not None
    assert tuple(adapter.dims(actual)) == tuple(reference.dims(expected))
    assert _labels(actual) == _labels(expected)
    a, e = values_of(actual), values_of(expected)
    assert a.dtype == e.dtype
    np.testing.assert_array_equal(a, e)


#: each function on the running example, with static arguments
FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "rename": lambda c: pa.rename(c, band="channel", x="col"),
    "expand_dims": lambda c: pa.expand_dims(c, ("member", "run"), axis=2),
    "squeeze": lambda c: pa.squeeze(pa.isel(c, time=slice(0, 1)), "time"),
    "broadcast_like": lambda c: pa.broadcast_like(
        pa.isel(c, time=0), DimArray(np.zeros((2, 13, 4, 5)), ("run", "band", "y", "x"))
    ),
    "merge_dims": lambda c: pa.merge_dims(c, pixel=("y", "x"), pb=("time", "band")),
    "split_dims": lambda c: pa.split_dims(
        c, time={"year": 3, "month": 4}, band={"band": 13, "one": 1}
    ),
}


# -- G1 xarray parity -----------------------------------------------------------


@pytest.fixture
def xr() -> Any:
    return pytest.importorskip("xarray")


def _assert_matches_xarray(ours: Any, theirs: Any) -> None:
    assert tuple(get_adapter(ours).dims(ours)) == tuple(map(str, theirs.dims))
    np.testing.assert_array_equal(values_of(ours), theirs.values)


def test_g1_rename_equals_xarray(xr: Any) -> None:
    cube = _cube()
    ours = pa.rename(cube, band="channel")
    theirs = cube.to_xarray().rename(band="channel")
    _assert_matches_xarray(ours, theirs)
    assert ours.labels["channel"] == tuple(theirs.indexes["channel"])


def test_g1_expand_dims_equals_xarray(xr: Any) -> None:
    cube = _cube()
    xa = cube.to_xarray()
    _assert_matches_xarray(
        pa.expand_dims(cube, "member", axis=2), xa.expand_dims("member", axis=2)
    )
    _assert_matches_xarray(
        pa.expand_dims(cube, "member", axis=-1), xa.expand_dims("member", axis=-1)
    )
    _assert_matches_xarray(
        pa.expand_dims(cube, ("member", "run")), xa.expand_dims(["member", "run"])
    )


def test_g1_squeeze_equals_xarray(xr: Any) -> None:
    first = pa.isel(_cube(), time=slice(0, 1))
    _assert_matches_xarray(pa.squeeze(first, "time"), first.to_xarray().squeeze("time"))


def test_g1_broadcast_like_equals_xarray(xr: Any) -> None:
    cube = _cube()
    median = pa.reduce(cube, "median", "time")
    ours = pa.broadcast_like(median, cube)
    theirs = median.to_xarray().broadcast_like(cube.to_xarray())
    _assert_matches_xarray(ours, theirs)
    # exclude: xarray neither adds nor aligns the excluded dim
    ours = pa.broadcast_like(median, cube, exclude=("time",))
    theirs = median.to_xarray().broadcast_like(cube.to_xarray(), exclude=["time"])
    _assert_matches_xarray(ours, theirs)


def test_g1_merge_dims_equals_xarray_stack(xr: Any) -> None:
    cube = _cube()
    ours = pa.merge_dims(cube, pixel=("y", "x"))
    theirs = cube.to_xarray().stack(pixel=("y", "x"))
    _assert_matches_xarray(ours, theirs)
    # tuple labels in the MultiIndex's order
    ours = pa.merge_dims(cube, pb=("band", "y"))
    theirs = cube.to_xarray().stack(pb=("band", "y"))
    _assert_matches_xarray(ours, theirs)
    assert ours.labels["pb"] == tuple(theirs.indexes["pb"])


def test_g1_split_dims_equals_xarray_unstack(xr: Any) -> None:
    cube = _cube()
    pixels = pa.merge_dims(cube, pixel=("y", "x"))
    ours = pa.split_dims(pixels, pixel={"y": 4, "x": 5})
    theirs = cube.to_xarray().stack(pixel=("y", "x")).unstack("pixel")
    _assert_matches_xarray(ours, theirs)


# -- G3 adapter agreement --------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Plain(DimArray):
    """A DimArray subclass whose adapter has no ``relabel``."""


class NoRelabelAdapter:
    """`DimArrayAdapter`'s seven methods without ``relabel``; keeps `Plain`."""

    def __init__(self) -> None:
        self._reference = DimArrayAdapter()

    def dims(self, x: Plain) -> tuple[str, ...]:
        return self._reference.dims(x)

    def data(self, x: Plain) -> Any:
        return self._reference.data(x)

    def labels(self, x: Plain, dim: str) -> tuple[Any, ...] | None:
        return self._reference.labels(x, dim)

    def isel(self, x: Plain, indexers: dict[str, Any]) -> Plain:
        out = self._reference.isel(x, indexers)
        return Plain(out.data, out.dims, out.labels)

    def rewrap(self, x: Plain, data: Any, dims: tuple[str, ...]) -> Plain:
        out = self._reference.rewrap(x, data, dims)
        return Plain(out.data, out.dims, out.labels)

    def check_compatible(self, *xs: Plain) -> tuple[Plain, ...]:
        return self._reference.check_compatible(*xs)

    def signature(self, x: Plain) -> Signature:
        return self._reference.signature(x)


@pytest.fixture
def no_relabel() -> Iterator[NoRelabelAdapter]:
    adapter = NoRelabelAdapter()
    register_adapter(Plain, adapter)
    yield adapter
    unregister_adapter(Plain)


def test_g3_kit_passes_without_relabel(no_relabel: NoRelabelAdapter) -> None:
    # only the cases on unlabelled dims run: an adapter without relabel
    # keeps working for every function that sets no labels
    check_adapter(no_relabel, lambda data, dims, labels: Plain(data, dims, labels))


# -- G4 jit, G5 vmap -------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g4_jit_equals_eager(name: str) -> None:
    jax = pytest.importorskip("jax")
    fn = FUNCTIONS[name]
    cube = _cube("jax")
    _assert_same(jax.jit(fn)(cube), fn(cube))


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g5_vmap_equals_a_stack_of_calls(name: str) -> None:
    jax = pytest.importorskip("jax")
    fn = FUNCTIONS[name]
    cubes = [_cube("jax", seed) for seed in range(4)]
    batched = jax.vmap(fn)(pa.stack(cubes, "batch"))
    _assert_same(batched, pa.stack([fn(c) for c in cubes], "batch"))


# -- G6 round trips --------------------------------------------------------------


BACKENDS = ["numpy", "jax", "dask"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_g6_split_undoes_merge(backend: str) -> None:
    cube = _cube(backend)
    pixels = pa.merge_dims(cube, pixel=("y", "x"))
    _assert_same(pa.split_dims(pixels, pixel={"y": 4, "x": 5}), cube)
    # a labelled source comes back with its labels, an unlabelled one without
    mixed = pa.merge_dims(cube, pb=("band", "y"))
    back = pa.split_dims(mixed, pb={"band": 13, "y": 4})
    _assert_same(pa.transpose(back, DIMS), cube)


@pytest.mark.parametrize("backend", BACKENDS)
def test_g6_rename_with_inverted_names_is_the_identity(backend: str) -> None:
    cube = _cube(backend)
    renamed = pa.rename(cube, band="channel", time="scene")
    _assert_same(pa.rename(renamed, channel="band", scene="time"), cube)
    # a swap of two names: the labels move with their dims
    swapped = pa.rename(cube, band="time", time="band")
    assert swapped.labels == {"time": BANDS}
    _assert_same(pa.rename(swapped, band="time", time="band"), cube)


@pytest.mark.parametrize("backend", BACKENDS)
def test_g6_squeeze_undoes_expand_dims(backend: str) -> None:
    cube = _cube(backend)
    _assert_same(pa.squeeze(pa.expand_dims(cube, "member"), "member"), cube)


# -- G8 dtype, G10 backends ------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_g8_g10_float32_on_every_backend(name: str, backend: str) -> None:
    fn = FUNCTIONS[name]
    out = fn(_cube(backend))
    assert out.dtype == np.float32
    _assert_same(out, fn(_cube()))


# -- G9 operators ----------------------------------------------------------------


@pytest.fixture(scope="module")
def running_example() -> DimArray:
    """§2's cube at full size: (12, 13, 256, 256) float32."""
    data = _data(shape=(12, 13, 256, 256))
    return DimArray(data, DIMS, {"band": BANDS})


def _operators() -> list[tuple[Any, tuple[Any, ...]]]:
    """Each twin, and its inputs after the cube."""
    members = DimArray(np.zeros(4, dtype=np.float32), ("member",))
    return [
        (Rename(band="channel", x="col"), ()),
        (ExpandDims(("member", "run"), axis=-2), ()),
        (Squeeze("member"), ()),
        (BroadcastLike(), (members,)),
        (MergeDims(pixel=("y", "x"), pb=("time", "band")), ()),
        (SplitDims(time={"year": 3, "month": 4}, band={"band": 13, "one": 1}), ()),
    ]


@pytest.mark.parametrize("case", range(6))
def test_g9_operators_round_trip_with_a_stable_hash(case: int) -> None:
    op, others = _operators()[case]
    cube = _cube()
    if isinstance(op, Squeeze):
        cube = pa.expand_dims(cube, "member")
    back = assert_round_trip(op, cube, *others)
    rebuilt = loads(dumps(back))
    assert pipeline_hash(rebuilt) == pipeline_hash(op)
    assert dumps(rebuilt) == dumps(op)


@pytest.mark.parametrize("case", range(6))
def test_g9_signature_predicts_the_run(case: int, running_example: DimArray) -> None:
    op, others = _operators()[case]
    cube = running_example
    if isinstance(op, Squeeze):
        cube = pa.expand_dims(cube, "member")
    out = op(cube, *others)
    if others:
        predicted = op.compute_output_signature(
            (signature_of(cube), signature_of(others[0]))
        )
    else:
        predicted = op.compute_output_signature(signature_of(cube))
    assert predicted == signature_of(out), type(op).__name__


def test_g9_order_of_merges_and_parts_is_in_the_hash() -> None:
    first = MergeDims(pixel=("y", "x"), pb=("time", "band"))
    second = MergeDims(pb=("time", "band"), pixel=("y", "x"))
    assert pipeline_hash(first) != pipeline_hash(second)
    yx = SplitDims(pixel={"y": 4, "x": 5})
    xy = SplitDims(pixel={"x": 5, "y": 4})
    assert pipeline_hash(yx) != pipeline_hash(xy)
    assert loads(dumps(xy)).dims == {"pixel": {"x": 5, "y": 4}}


def test_g9_broadcast_like_needs_both_signatures() -> None:
    sig = signature_of(_cube())
    assert BroadcastLike().compute_output_signature(sig) is None


# -- G11 labels ------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["dimarray", "xarray"])
def test_g11_labels(kind: str) -> None:
    cube: Any = _cube()
    if kind == "xarray":
        pytest.importorskip("xarray")
        cube = cube.to_xarray()
    adapter = get_adapter(cube)
    assert adapter is not None
    renamed = pa.rename(cube, band="channel")
    assert tuple(adapter.labels(renamed, "channel")) == BANDS
    merged = pa.merge_dims(cube, pb=("band", "y"))
    assert adapter.labels(merged, "pb")[0] == ("B1", 0)
    # broadcast_like carries other's labels onto the added dims
    members = {"member": ("a", "b", "c"), "band": BANDS}
    other = DimArray(np.zeros((3, *SHAPE)), ("member", *DIMS), members)
    if kind == "xarray":
        other = other.to_xarray()
    widened = pa.broadcast_like(pa.isel(cube, band=0), other)
    assert tuple(adapter.labels(widened, "member")) == ("a", "b", "c")
    assert tuple(adapter.labels(widened, "band")) == BANDS


def test_g11_an_adapter_without_relabel_raises_only_on_labels(
    no_relabel: NoRelabelAdapter,
) -> None:
    plain = Plain(_data(), DIMS, {"band": BANDS})
    with pytest.raises(TypeError, match=r"NoRelabelAdapter .*'channel'"):
        pa.rename(plain, band="channel")
    with pytest.raises(TypeError, match="NoRelabelAdapter"):
        pa.merge_dims(plain, pb=("band", "y"))
    # an unlabelled dim: the same data as the reference adapter
    reference = DimArray(_data(), DIMS, {"band": BANDS})
    _assert_same(pa.rename(plain, time="scene"), pa.rename(reference, time="scene"))
    _assert_same(
        pa.merge_dims(plain, pixel=("y", "x")),
        pa.merge_dims(reference, pixel=("y", "x")),
    )


@pytest.mark.parametrize("kind", ["dimarray", "xarray"])
def test_relabel_checks_lengths_and_dims(kind: str) -> None:
    cube: Any = _cube()
    if kind == "xarray":
        pytest.importorskip("xarray")
        cube = cube.to_xarray()
    adapter = get_adapter(cube)
    with pytest.raises(ValueError, match="2 labels for dim 'band' of size 13"):
        adapter.relabel(cube, {"band": ("a", "b")})
    with pytest.raises(ValueError, match="unknown dim"):
        adapter.relabel(cube, {"pixel": None})


# -- errors ----------------------------------------------------------------------


def test_errors_name_the_function_and_the_dim() -> None:
    cube = _cube()
    with pytest.raises(KeyError, match="rename: unknown dim"):
        pa.rename(cube, pixel="p")
    with pytest.raises(ValueError, match="repeat a name"):
        pa.rename(cube, band="time")
    with pytest.raises(ValueError, match="already dims"):
        pa.expand_dims(cube, "band")
    with pytest.raises(ValueError, match="axis 6 is out of range"):
        pa.expand_dims(cube, "member", axis=6)
    with pytest.raises(ValueError, match="'band' has size 13"):
        pa.squeeze(cube, "band")
    with pytest.raises(ValueError, match="size"):
        pa.broadcast_like(cube, DimArray(np.zeros((2, 3)), ("member", "band")))
    relabelled = DimArray(np.zeros(13), ("band",), {"band": tuple("abcdefghijklm")})
    with pytest.raises(ValueError, match="labels"):
        pa.broadcast_like(cube, relabelled)
    with pytest.raises(TypeError, match="sequence of dim names"):
        pa.merge_dims(cube, pixel="yx")
    with pytest.raises(ValueError, match="merged twice"):
        pa.merge_dims(cube, a=("y",), b=("y", "x"))
    with pytest.raises(ValueError, match="multiply to 21, but 'x' has size 5"):
        pa.split_dims(cube, x={"a": 3, "b": 7})
    with pytest.raises(TypeError, match="sequence of dim names"):
        MergeDims(pixel="yx")


def test_split_dims_reuses_the_split_dim_name() -> None:
    cube = _cube()
    windows = pa.split_dims(cube, time={"time": 4, "window": 3})
    assert windows.dims == ("time", "window", "band", "y", "x")
    np.testing.assert_array_equal(
        values_of(pa.isel(windows, time=1, window=2)), values_of(pa.isel(cube, time=5))
    )


def test_an_adapter_method_overrides_the_generic_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = get_adapter(_cube())
    calls: list[Any] = []
    monkeypatch.setattr(
        adapter, "merge_dims", lambda x, groups: calls.append(groups), raising=False
    )
    pa.merge_dims(_cube(), pixel=["y", "x"])
    assert calls == [{"pixel": ("y", "x")}]
