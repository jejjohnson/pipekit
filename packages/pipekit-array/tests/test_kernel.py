"""Tests for `KernelSpec` and `KernelOp` (#103)."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from pipekit import Signature, get_adapter, register_adapter, unregister_adapter
from pipekit_array import BackendError, DimArray, KernelOp, KernelSpec


BANDS = ("B1", "B2", "B3", "B4")
TARGET = np.array([0.1, -0.2, 0.3, 0.05])


# -- L1 kernels, arrays only ----------------------------------------------------


def normalized_difference(a: Any, b: Any, eps: float) -> Any:
    # ND = (a - b) / (a + b + ε)
    return (a - b) / (a + b + eps)


def _xp(a: Any) -> Any:
    if type(a).__module__.startswith(("jax", "jaxlib")) or "Tracer" in type(a).__name__:
        import jax.numpy as jnp

        return jnp
    return np


def background_stats(x: Any) -> tuple[Any, Any]:
    """μ and Σ of one column: (R, C) → (C,), (C, C)."""
    xp = _xp(x)
    # μ = (1/R) ∑ᵣ xᵣ
    mean = xp.mean(x, axis=0)
    # Σ = x̃ᵀx̃ / (R - 1) + λI
    centred = x - mean
    covariance = centred.T @ centred / (x.shape[0] - 1)
    return mean, covariance + 1e-6 * xp.eye(x.shape[1])


def matched_filter_score(x: Any, mu: Any, sigma: Any, *, t: Any) -> Any:
    """alpha per pixel of one column: (R, C), (C,), (C, C) → (R,)."""
    xp = _xp(x)
    # w = Σ⁻¹t, solved
    w = xp.linalg.solve(sigma, t)
    # alpha_r = (xᵣ - μ)ᵀw / tᵀw
    return ((x - mu) @ w) / (t @ w)


def gradient(f: Any, *, spacing: Any) -> tuple[Any, Any]:
    """Centred differences with the carrier's spacing: (Y, X) → (Y, X), (Y, X)."""
    xp = _xp(f)
    gy, gx = xp.gradient(f, axis=(-2, -1))
    return gy / spacing[0], gx / spacing[1]


ND = KernelOp(normalized_difference, "(),()->()", core={}, eps=1e-10)
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


def _scene(xp: Any = np, seed: int = 0) -> DimArray:
    rng = np.random.default_rng(seed)
    data = xp.asarray(rng.random((4, 16, 12)))
    return DimArray(data, ("band", "y", "x"), {"band": BANDS})


@pytest.fixture(params=["numpy", "jax"])
def xp(request: pytest.FixtureRequest) -> Any:
    if request.param == "jax":
        jax = pytest.importorskip("jax")
        previous = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        yield pytest.importorskip("jax.numpy")
        jax.config.update("jax_enable_x64", previous)
        return
    yield np


# -- KernelSpec -------------------------------------------------------------------


def test_spec_parses() -> None:
    spec = KernelSpec("(r,c),(c),(c,d)->(r)")
    assert spec.inputs == (("r", "c"), ("c",), ("c", "d"))
    assert spec.outputs == (("r",),)
    assert spec.letters == ("r", "c", "d")
    assert KernelSpec("(),()->()").inputs == ((), ())


@pytest.mark.parametrize("bad", ["(c)", "(c)->", "c->()", "((c))->()", ""])
def test_spec_rejects_bad_signatures(bad: str) -> None:
    with pytest.raises(ValueError, match="gufunc"):
        KernelSpec(bad)


def test_spec_sizes() -> None:
    with pytest.raises(ValueError, match="need a size"):
        KernelSpec("(r,c)->(c),(c,d)")
    with pytest.raises(ValueError, match="not an output-only"):
        KernelSpec("(r,c)->(c)", sizes=(("c", "r"),))
    with pytest.raises(ValueError, match="not an input letter"):
        KernelSpec("(r,c)->(d)", sizes=(("d", "z"),))
    assert hash(KernelSpec("(r,c)->(c),(c,d)", sizes=(("d", "c"),)))


def test_core_must_map_the_letters() -> None:
    with pytest.raises(ValueError, match="exactly"):
        KernelOp(background_stats, STATS.spec, core={"r": "y"})


# -- one kernel, every carrier -----------------------------------------------------


def test_raw_arrays_and_carriers_agree(xp: Any) -> None:
    scene = _scene(xp)
    nir = DimArray(scene.data[3], ("y", "x"))
    red = DimArray(scene.data[2], ("y", "x"))
    on_carriers = ND(nir, red)
    on_raw = ND(scene.data[3], scene.data[2])
    assert on_carriers.dims == ("y", "x")
    np.testing.assert_allclose(np.asarray(on_carriers.data), np.asarray(on_raw))


def test_matched_filter_on_named_carriers(xp: Any) -> None:
    """Per-column statistics consume y; the score comes back as y x (F2)."""
    scene = _scene(xp)
    mu, sigma = STATS(scene)
    assert mu.dims == ("band", "x")
    assert sigma.dims == ("band", "x", "band2")
    assert mu.labels == {"band": BANDS}
    alpha = SCORE(scene, mu, sigma)
    # the gufunc gives (x, y); KernelOp puts it in the first input's order
    assert alpha.dims == ("y", "x")
    values = np.asarray(scene.data)
    for col in (0, 7):
        x = values[:, :, col].T
        m = x.mean(0)
        s = np.cov(x, rowvar=False) + 1e-6 * np.eye(4)
        w = np.linalg.solve(s, TARGET)
        expected = ((x - m) @ w) / (TARGET @ w)
        np.testing.assert_allclose(np.asarray(alpha.data)[:, col], expected, rtol=1e-9)


def test_jit_traces_once_for_new_values() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    traces = 0

    def step(a: DimArray, b: DimArray) -> DimArray:
        nonlocal traces
        traces += 1
        return ND(a, b)

    f = jax.jit(step)
    for seed in range(5):
        scene = _scene(jnp, seed)
        out = f(
            DimArray(scene.data[3], ("y", "x")), DimArray(scene.data[2], ("y", "x"))
        )
    assert traces == 1
    assert out.dims == ("y", "x")


def test_geometry_is_derived_per_call(xp: Any) -> None:
    field = DimArray(xp.asarray(np.outer(np.arange(6.0), np.ones(5))), ("y", "x"))
    spacing = xp.asarray([2.0, 4.0])
    op = KernelOp(
        gradient,
        "(y,x)->(y,x),(y,x)",
        core={"y": "y", "x": "x"},
        geometry={"spacing": lambda _: spacing},
    )
    gy, gx = op(field)
    assert gy.dims == gx.dims == ("y", "x")
    np.testing.assert_allclose(np.asarray(gy.data), 0.5)
    np.testing.assert_allclose(np.asarray(gx.data), 0.0)


def test_loop_dims_broadcast_by_name() -> None:
    """A (time, x) carrier and an (x,) carrier map over time and x by name."""
    series = DimArray(np.arange(6.0).reshape(2, 3), ("time", "x"))
    offset = DimArray(np.array([10.0, 20.0, 30.0]), ("x",))
    out = KernelOp(lambda a, b: a + b, "(),()->()", core={})(series, offset)
    assert out.dims == ("time", "x")
    np.testing.assert_array_equal(out.data, series.data + offset.data)
    flipped = DimArray(series.data.T.copy(), ("x", "time"))
    out2 = KernelOp(lambda a, b: a - b, "(),()->()", core={})(series, flipped)
    np.testing.assert_array_equal(out2.data, np.zeros((2, 3)))


# -- errors -------------------------------------------------------------------------


def test_mixed_backends_name_to_jax() -> None:
    jnp = pytest.importorskip("jax.numpy")
    a = DimArray(np.ones((2, 3)), ("y", "x"))
    b = DimArray(jnp.ones((2, 3)), ("y", "x"))
    with pytest.raises(BackendError, match="to_jax"):
        ND(a, b)


def test_undeclared_backend() -> None:
    jnp = pytest.importorskip("jax.numpy")
    numpy_only = KernelOp(
        normalized_difference,
        KernelSpec("(),()->()", backends=("numpy",)),
        core={},
        eps=0.0,
    )
    a = DimArray(jnp.ones((2, 3)), ("y", "x"))
    with pytest.raises(BackendError, match="declares"):
        numpy_only(a, a)


def test_raw_array_next_to_carriers() -> None:
    a = DimArray(np.ones((2, 3)), ("y", "x"))
    with pytest.raises(TypeError, match="DimArray"):
        ND(a, np.ones((2, 3)))


def test_missing_core_dim_and_size_clash() -> None:
    with pytest.raises(ValueError, match="core dim"):
        STATS(DimArray(np.ones((4, 5)), ("band", "x")))
    a = DimArray(np.ones((2, 3)), ("y", "x"))
    with pytest.raises(ValueError, match="size"):
        ND(a, DimArray(np.ones((2, 4)), ("y", "x")))
    with pytest.raises(TypeError, match="input"):
        ND(a)


@dataclasses.dataclass(frozen=True)
class Batched:
    """A toy carrier with one geometry row per leading item."""

    data: Any
    dims: tuple[str, ...]


class BatchedAdapter:
    def dims(self, x: Batched) -> tuple[str, ...]:
        return x.dims

    def data(self, x: Batched) -> Any:
        return x.data

    def labels(self, x: Batched, dim: str) -> None:
        return None

    def isel(self, x: Batched, indexers: dict[str, Any]) -> Batched:
        raise NotImplementedError

    def rewrap(self, x: Batched, data: Any, dims: tuple[str, ...]) -> Batched:
        return Batched(data, dims)

    def check_compatible(self, *xs: Batched) -> tuple[Batched, ...]:
        return xs

    def signature(self, x: Batched) -> Signature:
        return Signature(dims=tuple(zip(x.dims, x.data.shape, strict=True)))

    def batch_dims(self, x: Batched) -> tuple[str, ...]:
        return x.dims[:1]


def test_geometry_on_a_batched_carrier_names_map_over() -> None:
    """F4: derived geometry would be passed whole, but varies per item."""
    register_adapter(Batched, BatchedAdapter())
    try:
        op = KernelOp(
            gradient,
            "(y,x)->(y,x),(y,x)",
            core={"y": "y", "x": "x"},
            geometry={"spacing": lambda _: np.array([1.0, 1.0])},
        )
        with pytest.raises(ValueError, match="map_over"):
            op(Batched(np.ones((3, 4, 5)), ("patch", "y", "x")))
    finally:
        unregister_adapter(Batched)


def test_an_adapter_with_its_own_kernel_path_takes_over() -> None:
    calls: list[str] = []

    class Own(BatchedAdapter):
        def apply_kernel(self, kernel, spec, core, kwargs, *xs):
            calls.append(spec.signature)
            return "own path"

    register_adapter(Batched, Own())
    try:
        assert (
            ND(Batched(np.ones(2), ("x",)), Batched(np.ones(2), ("x",))) == "own path"
        )
        assert calls == ["(),()->()"]
        with pytest.raises(BackendError, match="mix"):
            ND(Batched(np.ones(2), ("x",)), DimArray(np.ones(2), ("x",)))
    finally:
        unregister_adapter(Batched)


# -- signatures ---------------------------------------------------------------------


def _sig(d: DimArray) -> Signature:
    adapter = get_adapter(d)
    assert adapter is not None
    return adapter.signature(d)


def test_output_signature_equals_the_actual_one() -> None:
    scene = _scene()
    mu_sig, sigma_sig = STATS.compute_output_signature(_sig(scene))
    mu, sigma = STATS(scene)
    assert mu_sig.dims == _sig(mu).dims
    assert sigma_sig.dims == _sig(sigma).dims
    # sizes come from the spec: band2 has band's size
    assert dict(sigma_sig.dims)["band2"] == 4
    reduce_band = KernelOp(lambda x: x.sum(-1), "(c)->()", core={"c": "band"})
    sig = reduce_band.compute_output_signature(_sig(scene))
    assert sig.dims == _sig(reduce_band(scene)).dims == (("y", 16), ("x", 12))
    # several inputs: unknown here
    assert ND.compute_output_signature(_sig(scene)) is None


def test_config_is_a_debug_repr() -> None:
    assert KernelOp.forbid_in_yaml
    config = SCORE.get_config()
    assert config["kernel"].endswith(":matched_filter_score")
    assert config["spec"] == "(r,c),(c),(c,d)->(r)"
    assert config["params"] == ["t"]


# -- review fixes ---------------------------------------------------------------------


def test_param_named_like_a_constructor_argument() -> None:
    def scale(x: Any, geometry: float) -> Any:
        return x * geometry

    op = KernelOp(scale, "()->()", core={}, params={"geometry": 3.0})
    out = op(DimArray(np.ones((2,)), ("x",)))
    np.testing.assert_array_equal(out.data, [3.0, 3.0])
    with pytest.raises(TypeError, match="params="):
        KernelOp(scale, "()->()", core={}, geometry=3)


def test_param_and_derived_geometry_names_clash() -> None:
    with pytest.raises(ValueError, match="derived-geometry"):
        KernelOp(
            lambda x, dx: x, "()->()", core={}, geometry={"dx": lambda c: 1.0}, dx=2.0
        )


def test_dim_core_for_one_input_loop_for_another() -> None:
    op = KernelOp(lambda a, b: a.sum(-1) * b, "(c),()->()", core={"c": "band"})
    a = DimArray(np.ones((2, 3)), ("y", "band"))
    with pytest.raises(ValueError, match="core dims of one input but loop dims"):
        op(a, a)


def test_raw_inputs_are_checked_like_carriers() -> None:
    jnp = pytest.importorskip("jax.numpy")
    nd = KernelOp(lambda a, b: a - b, "(),()->()", core={})
    with pytest.raises(BackendError, match="to_jax"):
        nd(np.ones(3), jnp.ones(3))
    jax_only = KernelOp(lambda a: a, KernelSpec("()->()", backends=("jax",)), core={})
    with pytest.raises(BackendError, match="not 'numpy'"):
        jax_only(np.ones(3))


def test_vectorize_refuses_dask_instead_of_computing_it() -> None:
    da = pytest.importorskip("dask.array")
    op = KernelOp(
        lambda x: x.sum(), KernelSpec("(c)->()", vectorize=True), core={"c": "x"}
    )
    with pytest.raises(BackendError, match="eagerly"):
        op(DimArray(da.ones((3, 4), chunks=1), ("y", "x")))
