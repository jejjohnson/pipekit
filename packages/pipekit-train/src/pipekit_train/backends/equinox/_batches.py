"""The batches a training run reads, from an indexable or a streaming dataset."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import jax
import jax.numpy as jnp

from pipekit_train.backends.equinox._sharding import ShardingSpec, _place_batch


# ---------------------------------------------------------------------
# Data iteration — Grain or direct fallback
# ---------------------------------------------------------------------


def _is_indexable(dataset: Any) -> bool:
    """True when ``dataset`` supports random access (``len`` + ``getitem``)."""
    return hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")


def _stack_pair(samples: list[tuple[Any, Any]]) -> tuple[jax.Array, jax.Array]:
    xs = jnp.stack([jnp.asarray(s[0]) for s in samples])
    ys = jnp.stack([jnp.asarray(s[1]) for s in samples])
    return xs, ys


class _BatchSource:
    """Iterator-with-state wrapper used by the Equinox `run()` loop.

    Exposes `next_batch()` + optional `get_state()` / `set_state()`
    so the loop can checkpoint and resume iterator position.
    Indexable datasets use Grain's `MapDataset` chain (state methods
    work); streaming datasets fall back to direct Python iteration
    (state methods return / accept ``None``).
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        seed: int,
        sharding: ShardingSpec | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._seed = seed
        self._sharding = sharding
        # Process-sharding only kicks in for an *explicit* spec; without one
        # the documented single-device behaviour is preserved even if JAX was
        # distributed-initialised (process_count > 1).
        sharded = sharding is not None
        # Additive hook: a dataset may build its own Grain iterator (e.g.
        # geopatcher's `XarrayWindowDataset` block reader — coalesced reads, window
        # shuffle, multi-host sharding). It returns a checkpointable Grain
        # iterator, or None to decline (then we fall through to the generic
        # branches). Datasets without the method are unaffected.
        build = getattr(dataset, "build_batch_iter", None)
        custom = build(batch_size, seed) if build is not None else None
        if custom is not None:
            self._inner = custom
            self._kind = "grain"
        elif _is_indexable(dataset):
            self._inner = self._build_grain_iter(dataset, batch_size, seed, sharded)
            self._kind = "grain"
        else:
            if sharded and jax.process_count() > 1:
                raise NotImplementedError(
                    "Multi-host sharding needs an indexable dataset (the Grain "
                    "process-shard path); streaming IterableDatasets are "
                    "single-process. Materialise the dataset or wrap it as an "
                    "indexable TrainingDataset."
                )
            self._inner = self._build_streaming_iter(dataset, batch_size)
            self._kind = "streaming"

    @staticmethod
    def _build_grain_iter(
        dataset: Any, batch_size: int, seed: int, sharded: bool
    ) -> Any:
        """Build the Grain iterator.

        Pipeline: ``MapDataset.source(dataset) → .shuffle(seed) →
        .repeat(num_epochs=None) → .batch(local_batch, drop_remainder=True)
        → .to_iter_dataset() → iter(...)``. ``drop_remainder=True``
        matches the v0.1 numpy fallback's behaviour of dropping
        trailing partial batches so `train_step` doesn't recompile
        per epoch boundary.

        ``batch_size`` is the **global** per-step batch. With an explicit
        sharding spec on multiple processes, each process reads a disjoint
        shard (the MapDataset-API equivalent of
        ``grain.python.ShardByJaxProcess``) and batches the *local* size
        ``batch_size // process_count`` so the per-process batches assemble
        back into a global batch of exactly ``batch_size``.

        Raises:
            ValueError: when ``len(dataset) < batch_size``, or when a
                multi-host ``batch_size`` is not divisible by
                ``process_count``.
        """
        try:
            import grain  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise ImportError(
                "The Equinox adapter's data path needs grain (part of the "
                "[equinox] extra). Install with "
                "`pip install pipekit-train[equinox]`."
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
        n_proc = jax.process_count()
        local_batch = batch_size
        if sharded and n_proc > 1:
            if batch_size % n_proc != 0:
                raise ValueError(
                    f"Multi-host batch_size={batch_size} must be divisible by "
                    f"process_count={n_proc} so the global batch stays "
                    f"{batch_size}."
                )
            map_ds = map_ds[jax.process_index() :: n_proc]
            local_batch = batch_size // n_proc
        map_ds = (
            map_ds.shuffle(seed=seed)
            .repeat(num_epochs=None)
            .batch(batch_size=local_batch, drop_remainder=True)
        )
        return iter(map_ds.to_iter_dataset())

    @staticmethod
    def _build_streaming_iter(
        dataset: Any, batch_size: int
    ) -> Iterator[tuple[jax.Array, jax.Array]]:
        return _iter_streaming(dataset, batch_size)

    def next_batch(self) -> tuple[jax.Array, jax.Array]:
        batch = next(self._inner)
        xs, ys = jnp.asarray(batch[0]), jnp.asarray(batch[1])
        if self._sharding is not None:
            data_sharding = self._sharding.data_sharding
            xs = _place_batch(xs, data_sharding)
            ys = _place_batch(ys, data_sharding)
        return xs, ys

    def get_state(self) -> Any | None:
        """Return the inner iterator's state, or ``None`` for streaming."""
        if self._kind == "grain":
            return cast(Any, self._inner).get_state()
        return None

    def set_state(self, state: Any) -> None:
        """Restore the inner iterator's state.

        No-op for streaming iterators (which have no notion of state).
        """
        if self._kind == "grain" and state is not None:
            cast(Any, self._inner).set_state(state)


def _iter_streaming(
    dataset: Any,
    batch_size: int,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    """Direct iteration for non-indexable datasets (e.g. `IterableDataset`)."""
    while True:
        batch: list[tuple[Any, Any]] = []
        for x, y in dataset:
            batch.append((x, y))
            if len(batch) == batch_size:
                yield _stack_pair(batch)
                batch = []


def _pad_batch(batch: Any, batch_size: int) -> tuple[Any, jax.Array]:
    """Pad a batch to ``batch_size`` rows with copies of its last row.

    Copies rather than zeros, so that padded rows hold finite, in-distribution
    values, and any normalisation inside the model stays well behaved.

    Args:
        batch: A pytree of arrays with a shared leading dim of at most
            ``batch_size``.
        batch_size: The number of rows to return.

    Returns:
        The padded pytree, and a ``(batch_size,)`` bool mask, True for a real row.
    """
    size = len(jax.tree.leaves(batch)[0])
    extra = batch_size - size
    padded = jax.tree.map(
        lambda leaf: jnp.concatenate([leaf, jnp.repeat(leaf[-1:], extra, axis=0)]),
        batch,
    )
    return padded, jnp.arange(batch_size) < size
