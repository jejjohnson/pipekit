"""`batch` / `Padded`, `stack_items` and ``reduce(..., where=)`` (#119)."""

from __future__ import annotations

import numpy as np
import pytest
from pipekit_array import DimArray, Padded, batch, reduce, stack, stack_items


def _items(n: int) -> list[DimArray]:
    """``n`` (y, x) items; item i is filled with i."""
    return [DimArray(np.full((2, 3), float(i)), ("y", "x")) for i in range(n)]


def test_last_pads_with_copies_of_the_last_real_item() -> None:
    batches = list(batch(_items(6), size=4, dim="sample"))

    assert [b.value.dims for b in batches] == [("sample", "y", "x")] * 2
    first, last = batches
    assert first.valid.tolist() == [True] * 4
    assert last.valid.tolist() == [True, True, False, False]
    assert last.value.data.shape == (4, 2, 3)
    np.testing.assert_array_equal(last.value.data[:, 0, 0], [4.0, 5.0, 5.0, 5.0])
    assert np.isfinite(last.value.data).all()


def test_none_yields_a_short_last_batch_and_drop_drops_it() -> None:
    short = list(batch(_items(6), size=4, dim="sample", pad="none"))
    dropped = list(batch(_items(6), size=4, dim="sample", pad="drop"))

    assert [b.value.data.shape[0] for b in short] == [4, 2]
    assert short[-1].valid.tolist() == [True, True]
    assert [b.value.data.shape[0] for b in dropped] == [4]


def test_an_exact_multiple_has_no_padding() -> None:
    batches = list(batch(_items(8), size=4, dim="sample"))
    assert len(batches) == 2 and all(b.valid.all() for b in batches)
    assert list(batch([], size=4, dim="sample")) == []


def test_items_are_consumed_lazily() -> None:
    seen: list[int] = []

    def stream():
        for i, item in enumerate(_items(10)):
            seen.append(i)
            yield item

    first = next(batch(stream(), size=3, dim="sample"))
    assert first.valid.all() and seen == [0, 1, 2]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"size": 0}, "size must be positive"),
        ({"size": 2, "pad": "zeros"}, "pad must be one of"),
    ],
)
def test_bad_arguments_are_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        next(batch(_items(3), dim="sample", **kwargs))


def test_items_with_different_dims_are_rejected() -> None:
    items = [*_items(1), DimArray(np.ones((2, 3)), ("x", "y"))]
    with pytest.raises(ValueError):
        next(batch(items, size=2, dim="sample"))


def test_stack_items_is_stack_without_a_fast_path() -> None:
    items = _items(3)
    a, b = stack_items(items, "sample"), stack(items, "sample")
    assert a.dims == b.dims and np.array_equal(a.data, b.data)
    with pytest.raises(ValueError, match="at least one"):
        stack_items([], "sample")


def test_stack_items_uses_the_adapter_fast_path() -> None:
    from pipekit import get_adapter, register_adapter, unregister_adapter

    class Tagged(DimArray):
        pass

    calls: list[str] = []

    class TaggedAdapter(type(get_adapter(DimArray(np.ones(1), ("x",))))):
        def stack_items(self, xs, dim):  # type: ignore[no-untyped-def]
            calls.append(dim)
            return stack(xs, dim)

    register_adapter(Tagged, TaggedAdapter())
    try:
        out = stack_items([Tagged(np.ones(2), ("x",))] * 2, "patch")
    finally:
        unregister_adapter(Tagged)
    assert calls == ["patch"] and out.dims == ("patch", "x")


def test_masked_mean_leaves_padding_out() -> None:
    """The mean over a padded batch equals the mean over its real items."""
    padded = list(batch(_items(6), size=4, dim="sample"))[-1]

    masked = reduce(padded.value, "mean", "sample", where=padded.valid)
    real = reduce(stack(_items(6)[4:], "sample"), "mean", "sample")

    np.testing.assert_array_equal(masked.data, real.data)
    assert masked.dims == ("y", "x")


def test_a_named_where_aligns_by_dim_name() -> None:
    cube = DimArray(np.arange(24.0).reshape(2, 3, 4), ("t", "y", "x"))
    keep_x = DimArray(np.array([True, False, True, False]), ("x",))

    out = reduce(cube, "sum", ("y", "x"), where=keep_x)

    np.testing.assert_array_equal(out.data, cube.data[:, :, ::2].sum(axis=(1, 2)))


def test_a_named_where_in_another_order_is_transposed() -> None:
    cube = DimArray(np.arange(6.0).reshape(2, 3), ("t", "x"))
    mask = np.array([[True, False], [True, True], [False, True]])  # (x, t)

    out = reduce(cube, "sum", "x", where=DimArray(mask, ("x", "t")))

    np.testing.assert_array_equal(out.data, (cube.data * mask.T).sum(axis=1))


def test_a_plain_where_must_match_the_leading_dims() -> None:
    cube = DimArray(np.ones((4, 3)), ("sample", "x"))
    with pytest.raises(ValueError, match="leading dims"):
        reduce(cube, "mean", "sample", where=np.ones(3, dtype=bool))  # x's size


def test_a_named_where_with_an_unknown_dim_is_rejected() -> None:
    cube = DimArray(np.ones((4, 3)), ("sample", "x"))
    with pytest.raises(ValueError, match="not in"):
        reduce(cube, "mean", "x", where=DimArray(np.ones(5, dtype=bool), ("band",)))


def test_reduce_where_on_xarray_matches_the_generic_path() -> None:
    xr = pytest.importorskip("xarray")
    data = np.arange(12.0).reshape(4, 3)
    valid = np.array([True, True, False, True])

    generic = reduce(DimArray(data, ("sample", "x")), "mean", "sample", where=valid)
    on_xr = reduce(
        xr.DataArray(data, dims=("sample", "x")), "mean", "sample", where=valid
    )

    np.testing.assert_array_equal(np.asarray(on_xr), generic.data)


def test_a_named_where_with_other_lengths_is_rejected() -> None:
    """Review on #121: (3, 2) has (2, 3)'s element count, but not its lengths."""
    cube = DimArray(np.arange(6.0).reshape(2, 3), ("t", "x"))
    mask = DimArray(np.ones((3, 2), dtype=bool), ("t", "x"))
    with pytest.raises(ValueError, match="t: 3 vs 2, x: 2 vs 3"):
        reduce(cube, "sum", "x", where=mask)


def test_a_named_where_of_length_one_broadcasts() -> None:
    cube = DimArray(np.arange(6.0).reshape(2, 3), ("t", "x"))
    keep_all_t = DimArray(np.array([[True, False, True]]), ("t", "x"))  # (1, 3)
    out = reduce(cube, "sum", "x", where=keep_all_t)
    np.testing.assert_array_equal(out.data, cube.data[:, ::2].sum(axis=1))


def test_reduce_where_on_another_backend_says_so() -> None:
    """Review on #121: torch's reductions take no where= (nor dask's)."""
    from pipekit_array.dims import _require_where_backend

    class Tensor:  # stands in for torch.Tensor: only the module matters
        __module__ = "torch"

    class CupyArray:  # review on #121: cupy.mean / sum take no where= either
        __module__ = "cupy"

    with pytest.raises(TypeError, match=r"needs NumPy or JAX.*torch"):
        _require_where_backend(Tensor())
    with pytest.raises(TypeError, match=r"needs NumPy or JAX.*cupy"):
        _require_where_backend(CupyArray())
    _require_where_backend(np.ones(2))  # NumPy passes


def test_reduce_where_on_dask_says_so() -> None:
    da = pytest.importorskip("dask.array")
    cube = DimArray(da.ones((4, 3), chunks=2), ("sample", "x"))
    with pytest.raises(TypeError, match="dask"):
        reduce(cube, "mean", "sample", where=np.ones(4, dtype=bool))


def test_a_jitted_step_on_padded_batches_compiles_once() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    traces: list[int] = []

    @jax.jit
    def step(padded: Padded):
        traces.append(1)
        out = reduce(padded.value, "mean", ("sample", "y", "x"), where=padded.valid)
        return out.data

    items = [DimArray(jnp.full((2, 3), float(i)), ("y", "x")) for i in range(10)]
    means = [float(step(b)) for b in batch(items, size=4, dim="sample")]

    assert len(traces) == 1
    assert means == [1.5, 5.5, 8.5]  # the last batch: items 8, 9 (+ 2 copies of 9)
