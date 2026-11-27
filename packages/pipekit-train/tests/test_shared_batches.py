"""The batch source every backend reads (`pipekit_train.backends._batches`)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from pipekit_train import IterableDataset, TrainingDataset


pytest.importorskip("grain")

from pipekit_train.backends._batches import BatchSource, is_indexable


class _Indexed(TrainingDataset):
    forbid_in_yaml = True
    __config_mixin_auto__ = False

    def __init__(self, n: int = 20) -> None:
        super().__init__()
        self._x = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
        self._y = np.arange(n, dtype=np.float32)[:, None]

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, i: int):
        return self._x[i], self._y[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def content_hash(self) -> str:
        return "indexed-20"


def test_indexable_batches_are_numpy_and_cover_a_shuffled_epoch() -> None:
    source = BatchSource(_Indexed(), batch_size=4, seed=0)
    assert source.kind == "grain"
    batches = [source.next_batch() for _ in range(5)]
    for x, y in batches:
        assert isinstance(x, np.ndarray)
        assert x.shape == (4, 2)
        assert y.shape == (4, 1)
    seen = np.sort(np.concatenate([y[:, 0] for _, y in batches]))
    np.testing.assert_array_equal(seen, np.arange(20, dtype=np.float32))


def test_same_seed_same_batches_and_state_resumes() -> None:
    first = BatchSource(_Indexed(), batch_size=4, seed=3)
    second = BatchSource(_Indexed(), batch_size=4, seed=3)
    for _ in range(3):
        np.testing.assert_array_equal(first.next_batch()[0], second.next_batch()[0])
    state = first.get_state()
    resumed = BatchSource(_Indexed(), batch_size=4, seed=3)
    resumed.set_state(state)
    np.testing.assert_array_equal(first.next_batch()[0], resumed.next_batch()[0])


def test_equinox_wrapper_reads_the_shared_batches() -> None:
    pytest.importorskip("equinox")
    from pipekit_train.backends.equinox._batches import _BatchSource

    shared = BatchSource(_Indexed(), batch_size=4, seed=1)
    wrapped = _BatchSource(_Indexed(), batch_size=4, seed=1)
    for _ in range(3):
        x_shared, y_shared = shared.next_batch()
        x_jax, y_jax = wrapped.next_batch()
        np.testing.assert_array_equal(np.asarray(x_jax), x_shared)
        np.testing.assert_array_equal(np.asarray(y_jax), y_shared)


def test_streaming_dataset_has_no_state_and_restarts() -> None:
    pairs = [(np.full(2, i, np.float32), np.full(1, i, np.float32)) for i in range(5)]
    source = BatchSource(IterableDataset(source=pairs, content_hash="s5"), 2, 0)
    assert source.kind == "streaming"
    assert not is_indexable(IterableDataset(source=pairs, content_hash="s5"))
    assert source.get_state() is None
    source.set_state(None)
    ys = [source.next_batch()[1][:, 0].tolist() for _ in range(3)]
    # five samples give two full batches, then the dataset restarts
    assert ys == [[0.0, 1.0], [2.0, 3.0], [0.0, 1.0]]


def test_a_streaming_dataset_smaller_than_a_batch_raises() -> None:
    pairs = [(np.zeros(1), np.zeros(1))] * 3
    source = BatchSource(IterableDataset(source=pairs, content_hash="s3"), 4, 0)
    with pytest.raises(ValueError, match="3 samples per pass"):
        source.next_batch()


def test_errors_keep_their_messages() -> None:
    with pytest.raises(ValueError, match="batch_size=32"):
        BatchSource(_Indexed(), batch_size=32, seed=0)
    with pytest.raises(ValueError, match="divisible by process_count=3"):
        BatchSource(_Indexed(), batch_size=4, seed=0, process_count=3)
    pairs = [(np.zeros(1), np.zeros(1))]
    with pytest.raises(NotImplementedError, match="indexable"):
        BatchSource(
            IterableDataset(source=pairs, content_hash="s1"), 1, 0, process_count=2
        )


def test_process_shards_are_disjoint() -> None:
    shards = [
        BatchSource(_Indexed(), batch_size=4, seed=0, process_index=i, process_count=2)
        for i in range(2)
    ]
    values = [
        set(np.concatenate([s.next_batch()[1][:, 0] for _ in range(5)]).tolist())
        for s in shards
    ]
    assert values[0].isdisjoint(values[1])


def test_importing_the_module_imports_neither_jax_nor_keras() -> None:
    code = (
        "import sys; import pipekit_train.backends._batches; "
        "print('jax' in sys.modules, 'keras' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False False"
