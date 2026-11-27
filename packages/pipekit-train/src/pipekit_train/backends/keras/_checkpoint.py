"""Checkpoints of a Keras run that resume with the weights of an uninterrupted run.

A checkpoint holds the model's weights, the optimiser's variables (its step
counter among them) and the position of the training batches. It is written at
the end of a Keras epoch, one evaluation period, so a resumed run starts a new
epoch where the interrupted one would have; Keras cannot start ``fit`` inside an
epoch. Keras's own ``BackupAndRestore`` restarts an interrupted epoch from its
first batch with weights already updated, and saves no data position.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from pipekit_train.backends import _common
from pipekit_train.backends._batches import (
    BatchSource,
    _jsonify_grain_state,
    _unjsonify_grain_state,
)


class Feed:
    """The training batches, numbered by the step that trains on them.

    Keras reads ahead of the step it trains: a ``fit`` call of n steps reads
    n + 1 batches from a generator (Keras 3.14). The data position after the
    k-th step is therefore not the source's position when step k ends, and the
    batch read ahead at the end of one ``fit`` call would be lost to the next.
    The feed keeps each batch it read, with the data position after it, until
    training has passed it, and a new generator replays the batches read but
    not yet trained on.

    Args:
        source: The training batches.
        step: The number of steps already trained, 0 for a new run.
    """

    def __init__(self, source: BatchSource, step: int = 0) -> None:
        self.source = source
        self.trained = step
        self._read = step
        self._held: dict[int, tuple[Any, Any]] = {}

    def batches(self) -> Iterator[Any]:
        """Yield the batch of each step after the last one trained."""
        step = self.trained + 1
        while True:
            yield self._batch(step)
            step += 1

    def _batch(self, step: int) -> Any:
        if step not in self._held:
            if step != self._read + 1:
                raise RuntimeError(f"Batch {step} asked for after batch {self._read}.")
            batch = self.source.next_batch()
            self._held[step] = (batch, self.source.get_state())
            self._read = step
        return self._held[step][0]

    def trained_through(self, step: int) -> None:
        """Record that training has passed ``step``, dropping older batches."""
        self.trained = step
        for held in [s for s in self._held if s < step]:
            del self._held[held]

    def state_at(self, step: int) -> Any:
        """The data position after the batch of ``step``."""
        if step in self._held:
            return self._held[step][1]
        if step == self._read:
            return self.source.get_state()
        raise KeyError(f"No data position is kept for step {step}.")


def manager(loop: Any, period: int) -> Any | None:
    """An Orbax checkpoint manager for ``loop``, or ``None`` without a directory.

    The directory and its errors are the Equinox backend's: ``checkpoint_dir``
    or the `Checkpoint` callback's ``save_dir``, local paths only. Without a
    `Checkpoint` callback the cadence is one period, as the Equinox backend's
    is ``eval_every_n_steps``.

    Raises:
        ValueError: If ``Checkpoint.every_n_steps`` is not a multiple of the
            period, because a Keras run can only resume at a period's start.
    """
    callback = next(
        (
            cb
            for cb in loop.callbacks
            if hasattr(cb, "every_n_steps") and hasattr(cb, "keep_last")
        ),
        None,
    )
    directory = _common.resolve_checkpoint_dir(loop.checkpoint_dir, callback)
    if directory is None:
        return None
    every_n = int(callback.every_n_steps) if callback is not None else period
    keep_last = int(callback.keep_last) if callback is not None else 3
    if every_n % period:
        raise ValueError(
            f"Checkpoint.every_n_steps ({every_n}) must be a multiple of "
            f"TrainingLoop.eval_every_n_steps ({period}) for backend='keras': a "
            "Keras run resumes at the start of an evaluation period."
        )
    try:
        import orbax.checkpoint as ocp  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "Keras checkpoints need orbax-checkpoint (part of the [keras] extra)."
        ) from exc
    return ocp.CheckpointManager(
        directory=directory,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=keep_last, save_interval_steps=every_n
        ),
    )


def _arrays(model: Any) -> dict[str, list[np.ndarray]]:
    optimizer = model.optimizer
    if not optimizer.built:
        optimizer.build(model.trainable_variables)
    return {
        "weights": [np.asarray(w) for w in model.get_weights()],
        "optimizer": [np.asarray(v) for v in optimizer.variables],
    }


def save(
    mngr: Any, model: Any, step: int, data_state: Any, *, force: bool = False
) -> None:
    """Write the checkpoint of ``step``: weights, optimiser and data position."""
    import orbax.checkpoint as ocp  # ty: ignore[unresolved-import]

    mngr.save(step, args=ocp.args.StandardSave(_arrays(model)), force=force)
    mngr.wait_until_finished()
    if data_state is not None:
        step_dir = Path(os.fspath(mngr.directory)) / str(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "data_iter.json").write_text(
            json.dumps(_jsonify_grain_state(data_state), sort_keys=True)
        )


def restore(mngr: Any, model: Any) -> tuple[int, Any] | None:
    """Load the latest checkpoint into ``model``.

    Returns:
        ``(step, data_state)``, or ``None`` when there is no checkpoint.
        ``data_state`` is ``None`` for a streaming dataset.
    """
    import orbax.checkpoint as ocp  # ty: ignore[unresolved-import]

    step = mngr.latest_step()
    if step is None:
        return None
    template = _arrays(model)
    abstract = {
        key: [ocp.utils.to_shape_dtype_struct(a) for a in value]
        for key, value in template.items()
    }
    restored = mngr.restore(step, args=ocp.args.StandardRestore(abstract))
    model.set_weights([np.asarray(w) for w in restored["weights"]])
    for variable, value in zip(
        model.optimizer.variables, restored["optimizer"], strict=True
    ):
        variable.assign(np.asarray(value))
    side_car = Path(os.fspath(mngr.directory)) / str(step) / "data_iter.json"
    data_state = None
    if side_car.exists():
        data_state = _unjsonify_grain_state(json.loads(side_car.read_text()))
    return int(step), data_state
