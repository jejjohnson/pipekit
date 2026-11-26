"""Validation passes over a dataset, with padding for the last batch."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import equinox as eqx
import jax
import numpy as np

from pipekit_train.backends.equinox._batches import _pad_batch, _stack_pair
from pipekit_train.backends.equinox._state import TrainTask


# ---------------------------------------------------------------------
# Eval — runs the validation dataset through the model, returns metrics
# ---------------------------------------------------------------------


@eqx.filter_jit
def _eval_step(
    model: eqx.Module,
    batch: Any,
    key: jax.Array,
    task: TrainTask,
) -> dict[str, jax.Array]:
    """Compiled aux metrics for one validation batch.

    ``task`` is a static argument, as in `train_step`. A full batch and the
    shorter last batch each compile once, and later eval rounds reuse both.
    """
    _, aux = task.loss_fn(model, batch, key)
    return aux


@eqx.filter_jit
def _padded_eval_step(
    model: eqx.Module,
    batch: Any,
    key: jax.Array,
    valid: jax.Array,
    task: Any,
) -> dict[str, jax.Array]:
    """Compiled ``task.eval_fn`` for one padded batch; one shape, one compile."""
    return task.eval_fn(model, batch, key, valid)


def _evaluate(
    model: eqx.Module,
    val_dataset: Any,
    batch_size: int,
    task: TrainTask,
    key: jax.Array,
) -> dict[str, float]:
    """Run model on val_dataset, return the per-sample mean of each metric.

    Each batch's metrics are weighted by the batch's size, so a short last
    batch counts for its samples, not as much as a full batch (#122): the
    result is the mean over the validation set for any metric that is a
    per-batch mean.

    A task with an ``eval_fn`` gets padded batches instead; see `TrainTask`.

    ``key`` seeds stochastic tasks (dropout, reparameterised sampling)
    and is split per batch so every batch — and every eval round —
    sees distinct randomness. Deterministic tasks ignore it.
    """
    sums: dict[str, float] = {}
    n = 0  # samples seen
    # We always do a single-pass (one epoch) over the val dataset.
    # Use direct iteration; the loss_fn we already have can score
    # mini-batches as if they were training batches.
    iterator = _eval_pass(val_dataset, batch_size)
    for batch in iterator:
        key, batch_key = jax.random.split(key)
        if hasattr(task, "eval_fn"):
            padded, valid = _pad_batch(batch, batch_size)
            aux = _padded_eval_step(model, padded, batch_key, valid, task)
        else:
            aux = _eval_step(model, batch, batch_key, task)
        size = len(batch[0])
        for k, v in aux.items():
            sums[k] = sums.get(k, 0.0) + float(np.asarray(v)) * size
        n += size
    if n == 0:
        return {}
    return {k: v / n for k, v in sums.items()}


def _eval_pass(
    dataset: Any,
    batch_size: int,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    """One pass over the validation dataset."""
    batch: list[tuple[Any, Any]] = []
    for x, y in dataset:
        batch.append((x, y))
        if len(batch) == batch_size:
            yield _stack_pair(batch)
            batch = []
    if batch:
        yield _stack_pair(batch)
