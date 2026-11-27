"""The batches every backend reads, from an indexable or a streaming dataset.

`BatchSource` turns a `TrainingDataset` into ``(x, y)`` batches with a shuffle
fixed by the seed and a position that can be saved and restored. It imports
neither JAX nor Keras, so each backend converts and places the batches itself:
the Equinox backend turns them into JAX arrays on its devices, and the Keras
backend hands them to ``fit``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Literal, cast

import numpy as np


def is_indexable(dataset: Any) -> bool:
    """True when ``dataset`` supports random access (``len`` and ``getitem``)."""
    return hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")


def stack_pairs(samples: list[tuple[Any, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``(x, y)`` samples into one ``(x, y)`` batch along a new leading axis."""
    xs = np.stack([np.asarray(sample[0]) for sample in samples])
    ys = np.stack([np.asarray(sample[1]) for sample in samples])
    return xs, ys


class BatchSource:
    """Batches of ``(x, y)`` from a `TrainingDataset`, with a saveable position.

    An indexable dataset is read through Grain's ``MapDataset`` chain, whose
    iterator state `get_state` returns. A streaming dataset is iterated
    directly, has no state, and restarts when it runs out. A dataset may also
    build its own Grain iterator through a ``build_batch_iter(batch_size, seed)``
    method, which is tried first and may return ``None`` to decline.

    Args:
        dataset: The `TrainingDataset` to read.
        batch_size: The global number of samples per batch.
        seed: The shuffle seed.
        process_index: This process's index, for multi-process reading.
        process_count: The number of processes. Above one, each process reads a
            disjoint shard of an indexable dataset and batches
            ``batch_size // process_count`` samples, so the local batches
            assemble into a global batch of ``batch_size``.
        stack: How a streaming dataset's samples become one batch; NumPy's
            `stack_pairs` by default. The Equinox backend stacks with JAX.

    Raises:
        ValueError: If an indexable dataset is smaller than ``batch_size``, or
            ``batch_size`` is not divisible by ``process_count``.
        NotImplementedError: For a streaming dataset read by several processes.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        seed: int,
        *,
        process_index: int = 0,
        process_count: int = 1,
        stack: Callable[[list[tuple[Any, Any]]], tuple[Any, Any]] = stack_pairs,
    ) -> None:
        self.batch_size = batch_size
        self.seed = seed
        build = getattr(dataset, "build_batch_iter", None)
        custom = build(batch_size, seed) if build is not None else None
        self.kind: Literal["grain", "streaming"]
        if custom is not None:
            self._inner: Iterator[Any] = custom
            self.kind = "grain"
        elif is_indexable(dataset):
            self._inner = _grain_iter(
                dataset, batch_size, seed, process_index, process_count
            )
            self.kind = "grain"
        else:
            if process_count > 1:
                raise NotImplementedError(
                    "Multi-host sharding needs an indexable dataset (the Grain "
                    "process-shard path); streaming IterableDatasets are "
                    "single-process. Materialise the dataset or wrap it as an "
                    "indexable TrainingDataset."
                )
            self._inner = _iter_streaming(dataset, batch_size, stack)
            self.kind = "streaming"

    def next_batch(self) -> tuple[Any, Any]:
        """Return the next ``(x, y)`` batch.

        Grain batches are NumPy arrays, streaming batches are what ``stack``
        returns, and a dataset's own iterator returns whatever it yields.
        """
        batch = next(self._inner)
        return batch[0], batch[1]

    def get_state(self) -> Any | None:
        """Return the Grain iterator's state, or ``None`` for a streaming dataset."""
        if self.kind == "grain":
            return cast(Any, self._inner).get_state()
        return None

    def set_state(self, state: Any) -> None:
        """Restore the Grain iterator's state; does nothing for a streaming dataset."""
        if self.kind == "grain" and state is not None:
            cast(Any, self._inner).set_state(state)


def _grain_iter(
    dataset: Any,
    batch_size: int,
    seed: int,
    process_index: int,
    process_count: int,
) -> Iterator[Any]:
    """Build the Grain iterator over an indexable dataset.

    The chain is ``MapDataset.source(dataset) → .shuffle(seed) →
    .repeat(num_epochs=None) → .batch(local_batch, drop_remainder=True) →
    .to_iter_dataset()``. Dropping the remainder keeps every batch the same
    shape, so a compiled step does not recompile at an epoch boundary.
    """
    try:
        import grain  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "The training data path needs grain (part of the [equinox] and "
            "[keras] extras). Install with `pip install pipekit-train[equinox]` "
            "or `pip install pipekit-train[keras]`."
        ) from exc

    n = len(dataset)
    if n < batch_size:
        raise ValueError(
            f"Dataset has {n} samples but batch_size={batch_size}. "
            "Reduce batch_size, grow the dataset, or wrap as "
            "IterableDataset for streaming (which doesn't have this "
            "constraint)."
        )
    map_ds = grain.MapDataset.source(dataset)
    local_batch = batch_size
    if process_count > 1:
        if batch_size % process_count != 0:
            raise ValueError(
                f"Multi-host batch_size={batch_size} must be divisible by "
                f"process_count={process_count} so the global batch stays "
                f"{batch_size}."
            )
        map_ds = map_ds[process_index::process_count]
        local_batch = batch_size // process_count
    map_ds = (
        map_ds.shuffle(seed=seed)
        .repeat(num_epochs=None)
        .batch(batch_size=local_batch, drop_remainder=True)
    )
    return iter(map_ds.to_iter_dataset())


def _iter_streaming(
    dataset: Any,
    batch_size: int,
    stack: Callable[[list[tuple[Any, Any]]], tuple[Any, Any]],
) -> Iterator[tuple[Any, Any]]:
    """Iterate a dataset without random access, restarting when it runs out.

    Raises:
        ValueError: If one pass over the dataset holds fewer than ``batch_size``
            samples, which would otherwise loop forever.
    """
    while True:
        batch: list[tuple[Any, Any]] = []
        seen = 0
        for x, y in dataset:
            batch.append((x, y))
            seen += 1
            if len(batch) == batch_size:
                yield stack(batch)
                batch = []
        if seen < batch_size:
            raise ValueError(
                f"The streaming dataset has {seen} samples per pass but "
                f"batch_size={batch_size}; no full batch can be read. Reduce "
                "batch_size or grow the dataset."
            )
