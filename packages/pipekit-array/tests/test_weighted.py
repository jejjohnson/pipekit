"""`pipekit_array.weighted`: masked, weighted sums and means, and groups."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

import pytest


# the module under test needs both, and so do its tests
pytest.importorskip("array_api_compat")
strict = pytest.importorskip("array_api_strict")

import numpy as np
from hypothesis import given, strategies as st
from pipekit_array import DimArray
from pipekit_array.weighted import Groups, weighted_mean, weighted_sum
from pipekit_ops import BackendError


needs_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="JAX is not installed"
)
SECOND = strict.Device("device1")
RNG = np.random.default_rng(0)
N, G = 200, 12
# labels 3 and 11 stay empty, and about a tenth of the samples are in no group
LABELS = RNG.choice([-1, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10], size=N)
GROUPS = Groups.from_labels(LABELS, G)
# positive, so that float32 keeps its relative precision
X = np.abs(RNG.normal(size=(3, N)))
W = RNG.uniform(0.1, 2.0, size=N)
M = RNG.random((3, N)) > 0.2


def bincount(values: np.ndarray, use: np.ndarray) -> np.ndarray:
    """The grouped sum of each row by ``np.bincount``."""
    use = np.broadcast_to(use, values.shape) & (LABELS >= 0)
    return np.stack(
        [
            np.bincount(LABELS[u], weights=row[u], minlength=G)
            for row, u in zip(values, use, strict=True)
        ]
    )


CALLS: dict[str, Callable[..., Any]] = {
    "weighted_sum": lambda x, w, m: weighted_sum(x, weights=w, valid=m, axis=-1),
    "weighted_mean": lambda x, w, m: weighted_mean(x, weights=w, valid=m, axis=-1),
    "Groups.sum": lambda x, w, m: GROUPS.sum(x, valid=m),
    "Groups.mean": lambda x, w, m: GROUPS.mean(x, weights=w, valid=m),
}


class TestClosedForm:
    @pytest.mark.parametrize("axis", [None, 0, -1, (0, 1)])
    def test_sum_and_mean(self, axis: Any) -> None:
        weight = np.where(M, W, 0.0)
        total = np.sum(weight * X, axis=axis)
        out = weighted_sum(X, weights=W, valid=M, axis=axis)
        np.testing.assert_allclose(out, total, rtol=1e-13)
        with np.errstate(invalid="ignore"):
            expected = total / np.sum(weight, axis=axis)
        out = weighted_mean(X, weights=W, valid=M, axis=axis)
        np.testing.assert_allclose(out, expected, rtol=1e-13)

    def test_without_weights_or_a_mask(self) -> None:
        np.testing.assert_allclose(weighted_mean(X, axis=-1), X.mean(-1), rtol=1e-14)

    def test_the_scale_of_the_weights_cancels(self) -> None:
        np.testing.assert_allclose(
            weighted_mean(X, weights=1e6 * W), weighted_mean(X, weights=W), rtol=1e-13
        )

    def test_groups_against_bincount(self) -> None:
        total = bincount(X, M)
        np.testing.assert_allclose(GROUPS.sum(X, valid=M), total, rtol=1e-13)
        count = bincount(np.ones_like(X), M)
        np.testing.assert_array_equal(GROUPS.count(X, valid=M), count)
        with np.errstate(invalid="ignore"):
            expected = bincount(W * X, M) / bincount(np.broadcast_to(W, X.shape), M)
        out = GROUPS.mean(X, weights=W, valid=M)
        np.testing.assert_allclose(out, expected, rtol=1e-13)

    @given(
        st.integers(1, 6).flatmap(
            lambda g: st.tuples(
                st.just(g), st.lists(st.integers(-1, g - 1), min_size=0, max_size=30)
            )
        )
    )
    def test_any_labels(self, case: tuple[int, list[int]]) -> None:
        n_groups, labels = case
        groups = Groups.from_labels(np.asarray(labels, dtype=int), n_groups)
        values = np.arange(1.0, len(labels) + 1.0)
        # whole numbers, so every sum is exact
        expected = np.zeros(n_groups)
        for value, label in zip(values, labels, strict=True):
            if label >= 0:
                expected[label] += value
        np.testing.assert_array_equal(groups.sum(values), expected)

    def test_rings_keep_float32(self) -> None:
        n = 256
        k = np.fft.fftfreq(n, 1.0 / n)
        ring = np.rint(np.hypot(k[:, None], k[None, :])).astype(int).ravel()
        groups = Groups.from_labels(ring, int(ring.max()) + 1)
        power = np.random.default_rng(1).exponential(size=n * n).astype(np.float32)
        expected = np.bincount(ring, weights=power.astype(float))
        np.testing.assert_allclose(groups.sum(power), expected, rtol=1e-6)

    @pytest.mark.parametrize("axis", [0, 1, 2, -3])
    def test_the_groups_take_the_place_of_the_samples(self, axis: int) -> None:
        x = RNG.normal(size=(2, 3, N))
        moved = np.moveaxis(x, -1, axis)
        for method in (GROUPS.sum, GROUPS.mean, GROUPS.count):
            expected = np.moveaxis(method(x), -1, axis)
            np.testing.assert_array_equal(method(moved, axis=axis), expected)
        spread = GROUPS.spread(GROUPS.mean(moved, axis=axis), axis=axis)
        assert spread.shape == moved.shape

    def test_spread_is_the_adjoint_of_the_sum(self) -> None:
        groups = Groups.from_labels(np.where(LABELS < 0, 0, LABELS), G)
        x, r = RNG.normal(size=N), RNG.normal(size=G)
        assert groups.sum(x) @ r == pytest.approx(x @ groups.spread(r), rel=1e-12)

    def test_spread_of_a_sample_in_no_group_is_nan(self) -> None:
        spread = GROUPS.spread(np.arange(float(G)))
        np.testing.assert_array_equal(np.isnan(spread), LABELS < 0)
        np.testing.assert_array_equal(spread[LABELS >= 0], LABELS[LABELS >= 0])


class TestMissing:
    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_an_invalid_sample_is_not_read(self, name: str) -> None:
        call = CALLS[name]
        spoiled = np.where(M, X, np.nan)
        np.testing.assert_array_equal(call(spoiled, W, M), call(X, W, M))
        weights = np.where(M, W, np.nan)
        np.testing.assert_array_equal(call(X, weights, M), call(X, W, M))

    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_a_nan_in_a_valid_sample_gives_nan(self, name: str) -> None:
        spoiled = X.copy()
        spoiled[1, np.argmax(LABELS == 5)] = np.nan
        clean = CALLS[name](X, W, np.ones_like(M))
        out = CALLS[name](spoiled, W, np.ones_like(M))
        assert (np.isnan(out) & ~np.isnan(clean)).sum() == 1

    def test_nothing_to_use(self) -> None:
        none = np.zeros_like(M)
        assert np.isnan(weighted_mean(X, weights=W, valid=none, axis=-1)).all()
        assert np.isnan(weighted_mean(X, weights=0.0 * W, axis=-1)).all()
        assert (weighted_sum(X, valid=none, axis=-1) == 0.0).all()
        for empty in (3, 11):
            assert (GROUPS.sum(X)[:, empty] == 0.0).all()
            assert np.isnan(GROUPS.mean(X)[:, empty]).all()
        for labels in (np.full(4, -1), np.zeros(0, dtype=int)):
            groups = Groups.from_labels(labels, 2)
            np.testing.assert_array_equal(groups.sum(np.ones(labels.size)), [0.0, 0.0])

    @needs_jax
    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_the_gradient_is_finite_next_to_a_nan(self, name: str) -> None:
        import jax
        import jax.numpy as jnp

        valid = M.copy()
        # an empty mean too
        valid[1] = False
        spoiled = np.where(valid, X, np.nan).astype(np.float32)

        def loss(x: Any) -> Any:
            out = CALLS[name](x, jnp.asarray(W, jnp.float32), valid)
            return jnp.where(jnp.isnan(out), 0.0, out).sum()

        assert np.isfinite(jax.grad(loss)(jnp.asarray(spoiled))).all()


class TestBackends:
    @pytest.mark.parametrize("name", sorted(CALLS))
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("device", [None, SECOND], ids=["first", "second"])
    def test_the_standard_and_its_devices(
        self, name: str, dtype: str, device: Any
    ) -> None:
        x, w = (
            strict.asarray(a, dtype=getattr(strict, dtype), device=device)
            for a in (X, W)
        )
        # a mask of the host joins the device of the values
        for mask in (M, strict.asarray(M, device=device)):
            out = CALLS[name](x, w, mask)
            assert out.dtype == getattr(strict, dtype)
            assert out.device == x.device
            host = np.asarray(out.to_device(strict.Device("CPU_DEVICE")))
            rtol = 1e-5 if dtype == "float32" else 1e-12
            np.testing.assert_allclose(host, CALLS[name](X, W, M), rtol=rtol)

    @needs_jax
    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_jax_under_jit_and_vmap(self, name: str) -> None:
        import jax
        import jax.numpy as jnp

        x, w = jnp.asarray(X, jnp.float32), jnp.asarray(W, jnp.float32)
        expected = CALLS[name](X, W, M)
        for out in (
            CALLS[name](x, w, M),
            jax.jit(CALLS[name])(x, w, M),
            jax.vmap(CALLS[name], (0, None, 0))(x, w, jnp.asarray(M)),
        ):
            assert out.dtype == jnp.float32
            np.testing.assert_allclose(out, expected, rtol=1e-5)

    @needs_jax
    def test_groups_are_a_static_argument(self) -> None:
        import jax

        traces = []

        def mean(groups: Groups, x: Any) -> Any:
            traces.append(groups)
            return groups.mean(x)

        jitted = jax.jit(mean, static_argnums=0)
        for _ in range(2):
            jitted(GROUPS, X.astype(np.float32))
        assert len(traces) == 1

    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_labelled_arrays_are_refused(self, name: str) -> None:
        with pytest.raises(BackendError):
            CALLS[name](DimArray(X, ("row", "sample")), W, M)


class TestArguments:
    @pytest.mark.parametrize("valid", [[1.0, 0.0], [1, 0], np.asarray([0, 1])])
    def test_a_mask_that_is_not_boolean(self, valid: Any) -> None:
        with pytest.raises(ValueError, match="'valid' must be boolean"):
            weighted_mean([1.0, 2.0], valid=valid)

    def test_a_mask_may_be_a_list_or_a_python_bool(self) -> None:
        assert weighted_sum([1.0, 2.0], valid=[True, False]) == 1.0
        assert weighted_sum([1.0, 2.0], valid=False) == 0.0

    @pytest.mark.parametrize(
        ("labels", "match"),
        [
            ([0, 2], r"got 2 at position 1"),
            ([0, -2], r"got -2 at position 1"),
            ([[0, 1]], r"1-D array of integers"),
            ([0.0, 1.0], r"1-D array of integers"),
        ],
    )
    def test_labels_that_are_wrong(self, labels: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Groups.from_labels(labels, 2)

    def test_no_groups(self) -> None:
        with pytest.raises(ValueError, match="'n_groups' must be at least 1"):
            Groups.from_labels([], 0)

    def test_an_axis_of_another_size(self) -> None:
        for method in (GROUPS.sum, GROUPS.mean, GROUPS.count):
            with pytest.raises(ValueError, match="must hold the 200 samples"):
                method(X, axis=0)
        with pytest.raises(ValueError, match="must hold the 12 groups"):
            GROUPS.spread(X)

    def test_the_tables_are_read_only_copies(self) -> None:
        labels = LABELS.copy()
        groups = Groups.from_labels(labels, G)
        labels[:] = 0
        np.testing.assert_array_equal(groups.labels, LABELS)
        with pytest.raises(ValueError, match="read-only"):
            groups.index[0, 0] = 1
