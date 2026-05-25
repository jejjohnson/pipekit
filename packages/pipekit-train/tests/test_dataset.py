"""Tests for `pipekit_train.dataset`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pipekit import Operator
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


def test_cached_dataset_round_trip(tmp_path):
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
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    cached = CachedDataset(source=sim, cache_dir=str(tmp_path))
    pair = cached[3]
    assert len(pair) == 2


def test_cached_dataset_invalidate_re_materialises(tmp_path):
    sim = SimulationDataset(
        forward_model=_MockForward(), prior=_MockPrior(), n_samples=10
    )
    cached = CachedDataset(source=sim, cache_dir=str(tmp_path))
    _ = list(cached)  # materialise
    assert cached._materialised()
    cached.invalidate()
    assert not cached._materialised()


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
    with pytest.raises(NotImplementedError, match=r"v0\.2"):
        CachedDataset(source=sim, cache_dir=str(tmp_path), format="parquet")
