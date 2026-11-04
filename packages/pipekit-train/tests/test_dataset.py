"""Tests for `pipekit_train.dataset`."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest
from pipekit import Lambda, Operator, StatefulOperator
from pipekit_train.dataset import (
    CachedDataset,
    IterableDataset,
    SimulationDataset,
    TrainingDataset,
)


# --- Helpers --------------------------------------------------------------


class _MockForward:
    """Minimal ForwardModel — multiplies state by 2 per step."""

    dt: float = 1.0
    state_signature: Any = None

    def step(self, state: Any, dt: float) -> Any:
        return state * 2.0


class _MockPrior(Operator):
    """Returns a deterministic Gaussian sample per seed."""

    def __init__(self, dim: int = 3, scale: float = 1.0) -> None:
        self.dim = dim
        self.scale = scale

    def _apply(self, seed: int | None = None) -> Any:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim) * self.scale

    def __call__(self, *, seed: int | None = None) -> Any:
        return self._apply(seed=seed)


# --- TrainingDataset base -------------------------------------------------


def test_training_dataset_iter_is_abstract():
    class Bad(TrainingDataset):
        pass

    with pytest.raises(NotImplementedError, match="__iter__"):
        list(iter(Bad()))


def test_training_dataset_call_is_typeerror():
    """Datasets are iterated, not called like operators."""

    class _Trivial(TrainingDataset):
        def __iter__(self):
            yield (np.array([1.0]), np.array([2.0]))

    with pytest.raises(TypeError, match="iterated, not called"):
        _Trivial()()


def test_training_dataset_rejects_unknown_split():
    class _Trivial(TrainingDataset):
        def __iter__(self):
            yield (np.array([1.0]), np.array([2.0]))

    with pytest.raises(ValueError, match="split"):
        _Trivial(split="invalid")  # type: ignore[arg-type]


def test_training_dataset_with_split_returns_clone():
    class _Trivial(TrainingDataset):
        def __iter__(self):
            yield (np.array([1.0]), np.array([2.0]))

    a = _Trivial(seed=42, split="train")
    b = a.with_split("val")
    assert a.split == "train" and b.split == "val"
    assert a.seed == b.seed == 42
    assert b is not a


def test_default_content_hash_folds_in_config():
    class _ConfiguredDS(TrainingDataset):
        def __init__(self, scale: float = 1.0, seed: int = 0, split="train"):
            super().__init__(seed=seed, split=split)
            self.scale = scale

        def __iter__(self):
            yield (np.array([1.0]), np.array([2.0]))

    a = _ConfiguredDS(scale=1.0)
    b = _ConfiguredDS(scale=2.0)
    assert a.content_hash() != b.content_hash()


# --- IterableDataset ------------------------------------------------------


def test_iterable_dataset_wraps_callable_source():
    def gen():
        for i in range(3):
            yield (np.array([i]), np.array([i * 2]))

    ds = IterableDataset(source=gen, content_hash="abc")
    pairs = list(ds)
    assert len(pairs) == 3
    assert pairs[0] == (np.array([0]), np.array([0]))


def test_iterable_dataset_wraps_sequence_source():
    src = [(np.array([1.0]), np.array([2.0])), (np.array([3.0]), np.array([4.0]))]
    ds = IterableDataset(source=src, content_hash="seq")
    pairs = list(ds)
    assert len(pairs) == 2


def test_iterable_dataset_requires_content_hash():
    with pytest.raises(ValueError, match="content_hash"):
        IterableDataset(source=iter([]), content_hash="")


def test_iterable_dataset_rejects_non_iterable():
    with pytest.raises(TypeError, match="iterable"):
        IterableDataset(source=42, content_hash="h")  # type: ignore[arg-type]


def test_iterable_dataset_content_hash_depends_on_user_hash_and_split():
    a = IterableDataset(source=iter([]), content_hash="h1")
    b = IterableDataset(source=iter([]), content_hash="h2")
    c = a.with_split("val")
    assert a.content_hash() != b.content_hash()
    assert a.content_hash() != c.content_hash()


# --- SimulationDataset ----------------------------------------------------


def test_simulation_dataset_raises_a_typeerror_from_inside_the_prior():
    calls = []

    class BrokenPrior(Operator):
        def _apply(self, *, seed=None):
            calls.append(seed)
            raise TypeError("a real bug inside the prior")

    sim = SimulationDataset(
        forward_model=_MockForward(), prior=BrokenPrior(), n_samples=10
    )
    with pytest.raises(TypeError, match="real bug"):
        sim[0]
    assert calls == [0]


def test_simulation_dataset_calls_a_seedless_prior_without_seed():
    class Fixed(Operator):
        def _apply(self):
            return np.ones(2)

    sim = SimulationDataset(forward_model=_MockForward(), prior=Fixed(), n_samples=10)
    np.testing.assert_array_equal(sim[0][0], np.ones(2))


def test_simulation_dataset_reads_a_lambda_priors_function_signature():
    seedless = Lambda(lambda: np.ones(2))
    seeded = Lambda(lambda seed: np.full(2, float(seed)))

    sim = SimulationDataset(forward_model=_MockForward(), prior=seedless, n_samples=10)
    np.testing.assert_array_equal(sim[3][0], np.ones(2))

    sim = SimulationDataset(forward_model=_MockForward(), prior=seeded, n_samples=10)
    np.testing.assert_array_equal(sim[3][0], np.full(2, 3.0))


def test_simulation_dataset_seeds_each_sample_by_its_index():
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    np.testing.assert_array_equal(sim[2][0], sim[2][0])
    assert not np.array_equal(sim[2][0], sim[3][0])


def _prior_of_ones() -> Lambda:
    return Lambda(lambda: np.ones(2))


def test_simulation_dataset_rolls_out_a_real_cycle():
    pc = pytest.importorskip("pipekit_cycle")
    forward = _MockForward()
    sim = SimulationDataset(
        forward_model=forward,
        prior=_prior_of_ones(),
        n_samples=10,
        cycle=pc.Cycle(forward, n_steps=24),
    )

    expected = np.ones(2)
    for _ in range(24):
        expected = forward.step(expected, forward.dt)
    params, output = sim[0]

    np.testing.assert_array_equal(params, np.ones(2))
    np.testing.assert_array_equal(output, expected)


def test_simulation_dataset_samples_dont_depend_on_read_order():
    pc = pytest.importorskip("pipekit_cycle")

    class AddCount(StatefulOperator):
        def _apply(self, carrier, state):
            state.cycle_count += 1
            return carrier + state.cycle_count, state

    def build() -> SimulationDataset:
        return SimulationDataset(
            forward_model=_MockForward(),
            prior=_MockPrior(),
            n_samples=10,
            cycle=AddCount(),
            cycle_state=pc.DAState(),
        )

    in_order = build()
    out_of_order = build()
    first = [in_order[i][1] for i in (1, 3)]
    second = out_of_order[3][1], out_of_order[1][1]

    np.testing.assert_array_equal(first[0], second[1])
    np.testing.assert_array_equal(first[1], second[0])
    assert in_order.cycle_state.cycle_count == 0


def test_simulation_dataset_calls_a_stateless_cycle_with_params_only():
    sim = SimulationDataset(
        forward_model=_MockForward(),
        prior=_prior_of_ones(),
        n_samples=10,
        cycle=Lambda(lambda x: 3.0 * x),
    )
    np.testing.assert_array_equal(sim[0][1], np.full(2, 3.0))


def test_simulation_dataset_hashes_cycle_state_only_when_set():
    pc = pytest.importorskip("pipekit_cycle")

    def build(**kwargs: Any) -> SimulationDataset:
        return SimulationDataset(
            forward_model=_MockForward(), prior=_MockPrior(), n_samples=10, **kwargs
        )

    assert build().content_hash() == build().content_hash()
    equal_a = build(cycle_state=pc.DAState(t=1.0))
    equal_b = build(cycle_state=pc.DAState(t=1.0))
    other = build(cycle_state=pc.DAState(t=2.0))
    assert equal_a.content_hash() == equal_b.content_hash()
    assert equal_a.content_hash() != other.content_hash()
    assert equal_a.content_hash() != build().content_hash()


def test_simulation_dataset_keeps_its_seed_choice_through_pickle():
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(sim))[2][0], sim[2][0])


def test_simulation_dataset_yields_pairs():
    sim = SimulationDataset(
        forward_model=_MockForward(),
        prior=_MockPrior(dim=4),
        n_samples=10,
    )
    pairs = list(sim)
    assert len(pairs) == 8  # 80% of 10
    x, y = pairs[0]
    assert x.shape == (4,)
    np.testing.assert_array_almost_equal(y, x * 2.0)


def test_simulation_dataset_split_sizes_are_integer_exact():
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=20
    )
    assert len(sim) == 16  # train
    assert len(sim.with_split("val")) == 2  # 80% .. 90%
    assert len(sim.with_split("test")) == 2  # 90% .. 100%


def test_simulation_dataset_splits_are_disjoint_by_seed():
    """Same global seed; train and val draw from disjoint seed offsets."""
    train = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=20, seed=42
    )
    val = train.with_split("val")
    train_first = train[0][0]
    val_first = val[0][0]
    # The two splits use different sample seeds (train: 42+0, val: 42+16),
    # so the parameters must differ.
    assert not np.allclose(train_first, val_first)


def test_simulation_dataset_indexable():
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    assert len(sim) == 8
    pair = sim[0]
    assert len(pair) == 2
    with pytest.raises(IndexError):
        _ = sim[100]


def test_simulation_dataset_rejects_non_forward_model():
    class _Bad:
        pass  # no .step or .dt

    with pytest.raises(TypeError, match="ForwardModel"):
        SimulationDataset(forward_model=_Bad(), prior=_MockPrior(), n_samples=10)


def test_simulation_dataset_rejects_non_operator_prior():
    with pytest.raises(TypeError, match="Operator"):
        SimulationDataset(forward_model=_MockForward(), prior=lambda: 0, n_samples=10)


def test_simulation_dataset_rejects_nonpositive_n_samples():
    with pytest.raises(ValueError, match="positive"):
        SimulationDataset(forward_model=_MockForward(), prior=_MockPrior(), n_samples=0)


def test_simulation_dataset_content_hash_changes_with_split():
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=20
    )
    assert sim.content_hash() != sim.with_split("val").content_hash()


def test_simulation_dataset_obs_op_applied_to_output():
    """obs_op transforms the output before yielding (SBI training path)."""

    class _Negate(Operator):
        def _apply(self, x):
            return -x

    sim = SimulationDataset(
        forward_model=_MockForward(),
        prior=_MockPrior(dim=2),
        n_samples=10,
        obs_op=_Negate(),
    )
    x, y = sim[0]
    np.testing.assert_array_almost_equal(y, -(x * 2.0))


# --- CachedDataset --------------------------------------------------------

# zarr is only needed for the CachedDataset path. Each test starts with
# `pytest.importorskip("zarr")` so environments without it (e.g. the CI
# job that runs `uv sync --group dev` only) skip the cache tests
# cleanly without crashing the rest of the suite.


def test_cached_dataset_round_trip(tmp_path):
    pytest.importorskip("zarr")
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(dim=3), n_samples=10
    )
    cached = CachedDataset(source=sim, cache_dir=str(tmp_path))
    pairs_first = list(cached)
    # Re-iterate — should now read from cache, not regenerate.
    pairs_second = list(cached)
    assert len(pairs_first) == len(pairs_second) == 8  # 80% of 10
    for (x1, y1), (x2, y2) in zip(pairs_first, pairs_second, strict=True):
        np.testing.assert_array_almost_equal(np.asarray(x1), np.asarray(x2))
        np.testing.assert_array_almost_equal(np.asarray(y1), np.asarray(y2))


def test_cached_dataset_inherits_source_content_hash(tmp_path):
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    cached = CachedDataset(source=sim, cache_dir=str(tmp_path))
    assert cached.content_hash() == sim.content_hash()


def test_cached_dataset_indexable_after_materialise(tmp_path):
    pytest.importorskip("zarr")
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    cached = CachedDataset(source=sim, cache_dir=str(tmp_path))
    pair = cached[3]
    assert len(pair) == 2


def test_cached_dataset_invalidate_re_materialises(tmp_path):
    pytest.importorskip("zarr")
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    cached = CachedDataset(source=sim, cache_dir=str(tmp_path))
    _ = list(cached)  # materialise
    assert cached._materialised()
    cached.invalidate()
    assert not cached._materialised()


def _count_store_opens(monkeypatch) -> list[str]:
    zarr = pytest.importorskip("zarr")
    opens: list[str] = []
    real_open = zarr.open

    def counting_open(*args, **kwargs):
        opens.append(kwargs.get("mode", "?"))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(zarr, "open", counting_open)
    return opens


def test_cached_getitem_does_not_reopen_store(tmp_path, monkeypatch):
    """Regression for #141: the store used to reopen twice on every item."""
    opens = _count_store_opens(monkeypatch)
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    cached = CachedDataset(source=sim, cache_dir=str(tmp_path))
    first = cached[0]
    opens.clear()
    for i in range(100):
        second = cached[i % len(cached)]
    list(cached)
    assert opens == []
    assert len(second) == 2
    np.testing.assert_array_equal(cached[0][0], first[0])


def test_cached_dataset_reopens_after_invalidate(tmp_path):
    pytest.importorskip("zarr")
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    cached = CachedDataset(source=sim, cache_dir=str(tmp_path))
    x_before = np.asarray(cached[3][0])
    cached.invalidate()
    assert not cached._materialised()
    np.testing.assert_array_equal(cached[3][0], x_before)
    assert cached._materialised()


def test_cached_dataset_follows_a_change_of_content_hash(tmp_path):
    pytest.importorskip("zarr")
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    cached = CachedDataset(source=sim, cache_dir=str(tmp_path))
    before = np.asarray(cached[0][0])
    cached.source = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10, seed=7
    )
    assert not np.array_equal(cached[0][0], before)


def test_cached_dataset_in_memory_serves_numpy_with_the_same_values(tmp_path):
    pytest.importorskip("zarr")
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(dim=3), n_samples=10
    )
    on_disk = CachedDataset(source=sim, cache_dir=str(tmp_path))
    in_memory = CachedDataset(source=sim, cache_dir=str(tmp_path), in_memory=True)
    assert in_memory.content_hash() == on_disk.content_hash()
    for (x_mem, y_mem), (x_disk, y_disk) in zip(in_memory, on_disk, strict=True):
        np.testing.assert_array_equal(x_mem, x_disk)
        np.testing.assert_array_equal(y_mem, y_disk)
    assert isinstance(in_memory._arrays[1], np.ndarray)
    assert in_memory.with_split("val").in_memory is True


def test_cached_dataset_pickles_without_its_open_arrays(tmp_path):
    pytest.importorskip("zarr")
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    cached = CachedDataset(source=sim, cache_dir=str(tmp_path), in_memory=True)
    x = np.asarray(cached[1][0])
    assert cached._arrays is not None
    clone = pickle.loads(pickle.dumps(cached))
    assert clone._arrays is None
    assert cached._arrays is not None
    np.testing.assert_array_equal(clone[1][0], x)


def test_cached_dataset_empty_source_raises_clear_error(tmp_path):
    """A 0-length indexable source can't be materialised — we need at
    least one sample to infer shape/dtype for the zarr arrays."""
    pytest.importorskip("zarr")

    class _EmptyDataset(SimulationDataset):
        # n_samples=10 with split="test" + 90/10 slice = 1 sample;
        # we force 0 by overriding __len__.
        def __len__(self) -> int:
            return 0

    sim = _EmptyDataset(forward_model=_MockForward(), prior=_MockPrior(), n_samples=10)
    cached = CachedDataset(source=sim, cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="empty source"):
        cached._materialise()


def test_cached_dataset_with_split_uses_different_cache_path(tmp_path):
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=20
    )
    train = CachedDataset(source=sim, cache_dir=str(tmp_path))
    val = train.with_split("val")
    assert train._cache_path() != val._cache_path()


def test_cached_dataset_rejects_non_indexable_source(tmp_path):
    bad = IterableDataset(source=iter([]), content_hash="h")
    with pytest.raises(TypeError, match="indexable"):
        CachedDataset(source=bad, cache_dir=str(tmp_path))


def test_cached_dataset_rejects_non_training_dataset(tmp_path):
    with pytest.raises(TypeError, match="TrainingDataset"):
        CachedDataset(source=[1, 2, 3], cache_dir=str(tmp_path))  # type: ignore[arg-type]


def test_cached_dataset_unimplemented_formats(tmp_path):
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    with pytest.raises(NotImplementedError, match=r"v0\.3"):
        CachedDataset(source=sim, cache_dir=str(tmp_path), format="parquet")


# --- fsspec-backed CachedDataset (#18 — v0.2) -----------------------------


def test_cached_dataset_is_fsspec_uri_detection():
    """URI detection: scheme-prefixed paths route through fsspec;
    bare local paths and file:// don't."""
    assert CachedDataset._is_fsspec_uri("s3://bucket/path") is True
    assert CachedDataset._is_fsspec_uri("gcs://bucket/path") is True
    assert CachedDataset._is_fsspec_uri("memory://tmp") is True
    assert CachedDataset._is_fsspec_uri("/tmp/cache") is False
    assert CachedDataset._is_fsspec_uri("relative/cache") is False
    # file:// is treated as local, not fsspec, because zarr's
    # LocalStore handles it natively.
    assert CachedDataset._is_fsspec_uri("file:///tmp/cache") is False


def test_cached_dataset_cache_path_for_fsspec_uri_uses_forward_slash():
    """The cache_path for an fsspec URI never goes through os.path.join
    (which would mangle the scheme on Windows or strip slashes)."""
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    cached = CachedDataset(source=sim, cache_dir="s3://bucket/prefix")
    path = cached._cache_path()
    assert path.startswith("s3://bucket/prefix/")
    # Hash suffix present.
    assert path == f"s3://bucket/prefix/{sim.content_hash()}"


def test_cached_dataset_round_trip_via_fsspec_memory_filesystem():
    """End-to-end fsspec path via fsspec's in-memory filesystem.

    No real S3 / GCS — uses `memory://` which doesn't require any
    cloud creds and is in fsspec's default backend set.
    """
    pytest.importorskip("zarr")
    pytest.importorskip("fsspec")
    # Memory filesystem is built into fsspec; no extra import needed
    # to register it (fsspec auto-registers on first url_to_fs call).

    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(dim=3), n_samples=10
    )

    import uuid

    # Unique memory:// URI per test so we don't collide across runs.
    cached = CachedDataset(source=sim, cache_dir=f"memory://test-{uuid.uuid4().hex}")
    assert not cached._materialised()

    pairs_first = list(cached)
    assert cached._materialised()

    pairs_second = list(cached)
    assert len(pairs_first) == len(pairs_second) == 8  # 80% of 10
    for (x1, y1), (x2, y2) in zip(pairs_first, pairs_second, strict=True):
        np.testing.assert_array_almost_equal(np.asarray(x1), np.asarray(x2))
        np.testing.assert_array_almost_equal(np.asarray(y1), np.asarray(y2))


def test_cached_dataset_invalidate_via_fsspec_memory_filesystem():
    """`invalidate()` correctly removes the cache directory on fsspec."""
    pytest.importorskip("zarr")
    pytest.importorskip("fsspec")
    import uuid

    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(dim=3), n_samples=10
    )
    cached = CachedDataset(
        source=sim, cache_dir=f"memory://invalidate-{uuid.uuid4().hex}"
    )
    _ = list(cached)  # materialise
    assert cached._materialised()
    cached.invalidate()
    assert not cached._materialised()
