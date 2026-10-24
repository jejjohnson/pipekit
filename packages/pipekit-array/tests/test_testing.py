"""Self-tests for the adapter and KernelOp test kit, and gates PK-K3 to PK-K5 (#105)."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import pipekit_array as pa
import pytest
from pipekit import Sequential, Signature, register_adapter, unregister_adapter
from pipekit_array import DimArray, DimArrayAdapter, ISel, KernelOp, KernelSpec, Reduce
from pipekit_array.testing import check_adapter, check_kernel_op


def normalized_difference(a: Any, b: Any, eps: float) -> Any:
    return (a - b) / (a + b + eps)


ND = KernelOp(normalized_difference, "(),()->()", core={}, eps=1e-10)
BAND_SUM = KernelOp(
    lambda x: x.sum(-1), KernelSpec("(c)->()", dtypes=("same",)), core={"c": "band"}
)


def _dimarray(xp: Any = np) -> Any:
    def make(data: np.ndarray, dims: tuple[str, ...], labels: dict) -> DimArray:
        return DimArray(xp.asarray(data), dims, labels)

    return make


# -- check_adapter -------------------------------------------------------------------


def test_kit_passes_on_dimarray_numpy() -> None:
    check_adapter(DimArrayAdapter(), _dimarray())


def test_kit_passes_on_dimarray_jax() -> None:
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        check_adapter(DimArrayAdapter(), _dimarray(pytest.importorskip("jax.numpy")))
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("lazy", [False, True])
def test_kit_passes_on_xarray(lazy: bool) -> None:
    xr = pytest.importorskip("xarray")
    da = pytest.importorskip("dask.array")
    from pipekit_array.xarray import XarrayAdapter

    def make(data: np.ndarray, dims: tuple[str, ...], labels: dict) -> Any:
        values = da.from_array(data, chunks=(1, 3, 4, 5)) if lazy else data
        return xr.DataArray(
            values, dims=dims, coords={k: list(v) for k, v in labels.items()}
        )

    check_adapter(XarrayAdapter(), make)


@dataclasses.dataclass(frozen=True, eq=False)
class Broken(DimArray):
    """A DimArray subclass, so a broken adapter can be registered for it alone."""


class ForgetsLabels(DimArrayAdapter):
    """isel drops every label: a classic adapter bug."""

    def isel(self, x: DimArray, indexers: dict[str, Any]) -> DimArray:
        out = super().isel(x, indexers)
        return Broken(out.data, out.dims)

    def rewrap(self, x: DimArray, data: Any, dims: tuple[str, ...]) -> DimArray:
        out = super().rewrap(x, data, dims)
        return Broken(out.data, out.dims, out.labels)


class WrongFastPath(DimArrayAdapter):
    """A reduce fast path that sums instead of reducing as asked."""

    def reduce(self, x: DimArray, fn: Any, dims: tuple[str, ...]) -> DimArray:
        data = x.data.sum(axis=tuple(x.dims.index(d) for d in dims))
        kept = tuple(d for d in x.dims if d not in dims)
        return Broken(data, kept)

    def rewrap(self, x: DimArray, data: Any, dims: tuple[str, ...]) -> DimArray:
        out = super().rewrap(x, data, dims)
        return Broken(out.data, out.dims, out.labels)


def _make_broken(data: np.ndarray, dims: tuple[str, ...], labels: dict) -> Broken:
    return Broken(data, dims, labels)


@pytest.mark.parametrize(
    ("adapter", "message"),
    [
        (ForgetsLabels(), r"Broken: isel did not slice"),
        (WrongFastPath(), r"reduce\(mean, time\) on Broken"),
    ],
)
def test_kit_fails_on_a_broken_adapter(adapter: DimArrayAdapter, message: str) -> None:
    register_adapter(Broken, adapter)
    try:
        with pytest.raises(AssertionError, match=message):
            check_adapter(adapter, _make_broken)
    finally:
        unregister_adapter(Broken)


def test_kit_needs_the_registered_adapter() -> None:
    with pytest.raises(AssertionError, match="not the adapter under test"):
        check_adapter(ForgetsLabels(), _dimarray())


# -- check_kernel_op (PK-K1) -----------------------------------------------------------


def test_kernel_op_agrees_across_families() -> None:
    rng = np.random.default_rng(0)
    nir, red = rng.random((4, 5)), rng.random((4, 5))
    calls: list[Any] = [(DimArray(nir, ("y", "x")), DimArray(red, ("y", "x")))]
    xr = pytest.importorskip("xarray")
    calls.append(
        (xr.DataArray(nir, dims=("y", "x")), xr.DataArray(red, dims=("y", "x")))
    )
    check_kernel_op(ND, calls)
    scene = rng.random((3, 4, 5))
    check_kernel_op(
        BAND_SUM,
        [
            DimArray(scene, ("band", "y", "x")),
            xr.DataArray(scene, dims=("band", "y", "x")),
        ],
    )


def test_kernel_op_check_catches_a_wrong_size_or_dtype() -> None:
    """Codex #114: a kernel that breaks its declared size or dtype fails the check."""
    scene = DimArray(np.random.default_rng(0).random((3, 4)), ("band", "x"))
    short = KernelOp(
        lambda z: z[..., :1],
        KernelSpec("(c)->(d)", sizes=(("d", "c"),)),
        core={"c": "band", "d": "out"},
    )
    with pytest.raises(AssertionError, match="out=3"):
        check_kernel_op(short, [scene])
    liar = KernelOp(
        lambda z: z.sum(-1) > 0,
        KernelSpec("(c)->()", dtypes=("same",)),
        core={"c": "band"},
    )
    with pytest.raises(AssertionError, match="dtype"):
        check_kernel_op(liar, [scene])


def test_kernel_op_mismatch_names_the_carrier() -> None:
    xr = pytest.importorskip("xarray")
    scene = np.random.default_rng(0).random((3, 4, 5))
    swapped = xr.DataArray(scene.transpose(0, 2, 1).copy(), dims=("band", "x", "y"))
    with pytest.raises(AssertionError, match="DataArray: dims"):
        check_kernel_op(BAND_SUM, [DimArray(scene, ("band", "y", "x")), swapped])


# -- PK-K3: one trace for new transforms -----------------------------------------------


@dataclasses.dataclass(frozen=True)
class Gridded:
    """A toy geo carrier: data and a (6,) transform leaf; dims are static."""

    data: Any
    transform: Any
    dims: tuple[str, ...]


class GriddedAdapter(DimArrayAdapter):
    def labels(self, x: Any, dim: str) -> None:
        return None

    def rewrap(self, x: Gridded, data: Any, dims: tuple[str, ...]) -> Gridded:
        return Gridded(data, x.transform, dims)

    def check_compatible(self, *xs: Gridded) -> tuple[Gridded, ...]:
        return xs

    def signature(self, x: Gridded) -> Signature:
        return Signature(dims=tuple(zip(x.dims, x.data.shape, strict=True)))


def test_pk_k3_one_trace_for_new_transforms() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.tree_util.register_pytree_node(
        Gridded,
        lambda g: ((g.data, g.transform), g.dims),
        lambda dims, leaves: Gridded(leaves[0], leaves[1], dims),
    )
    register_adapter(Gridded, GriddedAdapter())
    try:
        traces = 0

        def step(a: Gridded, b: Gridded) -> Any:
            nonlocal traces
            traces += 1
            return ND(a, b).data

        f = jax.jit(step)
        data = jnp.ones((4, 5))
        for origin in (0.0, 600000.0, 123456.7):
            transform = jnp.asarray([20.0, 0.0, origin, 0.0, -20.0, 0.0])
            f(
                Gridded(data, transform, ("y", "x")),
                Gridded(data * 2, transform, ("y", "x")),
            )
        assert traces == 1
    finally:
        unregister_adapter(Gridded)


# -- PK-K4: summaries show dims for every step -------------------------------------


def test_pk_k4_summary_has_dims_for_every_step() -> None:
    cube = DimArray(
        np.ones((12, 3, 4, 5)), ("time", "band", "y", "x"), {"band": ("a", "b", "c")}
    )
    pipe = Sequential([ISel(dim="time", index=0), BAND_SUM, Reduce("mean", dim="x")])
    summary = pipe.summary(cube)
    assert "?" not in summary, summary
    assert "(y=4)" in summary
    assert pipe(cube).dims == ("y",)


# -- PK-K5: dispatch overhead ------------------------------------------------------

#: budget from the spike (geotoolz#516): dispatch adds at most 25 µs per call
#: over calling the kernel on the raw data
BUDGET_US = 25.0


@pytest.mark.slow
def test_pk_k5_dispatch_overhead() -> None:
    rng = np.random.default_rng(0)
    nir_raw, red_raw = rng.random((16, 16)), rng.random((16, 16))
    nir, red = DimArray(nir_raw, ("y", "x")), DimArray(red_raw, ("y", "x"))
    n = 5_000

    def per_call(fn: Any) -> float:
        """The best of 5 repeats: the least disturbed by a shared machine."""
        fn()
        best = float("inf")
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(n):
                fn()
            best = min(best, (time.perf_counter() - start) / n * 1e6)
        return best

    raw = per_call(lambda: normalized_difference(nir_raw, red_raw, 1e-10))
    dispatched = per_call(lambda: ND(nir, red))
    print(
        f"\nPK-K5: raw {raw:.1f} µs, KernelOp {dispatched:.1f} µs, "
        f"budget raw + {BUDGET_US:.0f} µs"
    )
    assert dispatched <= raw + BUDGET_US


def test_kernel_spec_from_the_kit_imports() -> None:
    assert KernelSpec("(c)->()").inputs == (("c",),)
    assert pa.testing.DIMS == ("time", "band", "y", "x")
