"""The batches a training run reads, from an indexable or a streaming dataset."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from pipekit_train.backends._batches import BatchSource, is_indexable
from pipekit_train.backends.equinox._sharding import ShardingSpec, _place_batch


# ---------------------------------------------------------------------
# Data iteration — Grain or direct fallback
# ---------------------------------------------------------------------


_is_indexable = is_indexable


def _stack_pair(samples: list[tuple[Any, Any]]) -> tuple[jax.Array, jax.Array]:
    xs = jnp.stack([jnp.asarray(s[0]) for s in samples])
    ys = jnp.stack([jnp.asarray(s[1]) for s in samples])
    return xs, ys


class _BatchSource:
    """The shared `BatchSource`, with its batches as JAX arrays on the devices.

    Exposes `next_batch()` + `get_state()` / `set_state()` so the loop can
    checkpoint and resume the iterator position. The reading itself (Grain for
    an indexable dataset, direct iteration for a streaming one, a dataset's own
    ``build_batch_iter``) is `pipekit_train.backends._batches.BatchSource`,
    which the Keras backend reads too.

    ``batch_size`` is the **global** per-step batch. With an explicit sharding
    spec on multiple processes, each process reads a disjoint shard and batches
    ``batch_size // process_count`` samples; without a spec the documented
    single-device behaviour holds even if JAX was distributed-initialised.
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
        sharded = sharding is not None
        self._source = BatchSource(
            dataset,
            batch_size,
            seed,
            process_index=jax.process_index() if sharded else 0,
            process_count=jax.process_count() if sharded else 1,
            stack=_stack_pair,
        )
        self._kind = self._source.kind

    def next_batch(self) -> tuple[jax.Array, jax.Array]:
        x, y = self._source.next_batch()
        xs, ys = jnp.asarray(x), jnp.asarray(y)
        if self._sharding is not None:
            data_sharding = self._sharding.data_sharding
            xs = _place_batch(xs, data_sharding)
            ys = _place_batch(ys, data_sharding)
        return xs, ys

    def get_state(self) -> Any | None:
        """Return the inner iterator's state, or ``None`` for streaming."""
        return self._source.get_state()

    def set_state(self, state: Any) -> None:
        """Restore the inner iterator's state.

        No-op for streaming iterators (which have no notion of state).
        """
        self._source.set_state(state)


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
