"""The end-to-end Keras training run, `run(loop)` (ADRs D18 and D19)."""

from __future__ import annotations

import copy
import math
import time
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from pipekit_ops import Operator

from pipekit_train.backends import _common
from pipekit_train.backends._batches import BatchSource, is_indexable, stack_pairs
from pipekit_train.backends.keras._backend import import_keras
from pipekit_train.backends.keras._losses import aux_metrics, aux_names, bridge_loss
from pipekit_train.backends.keras._model import KerasModelOp
from pipekit_train.backends.keras._optim import build_keras_optimizer
from pipekit_train.backends.keras._task import KerasTask


if TYPE_CHECKING:
    from pipekit_train.loop._loop import TrainingLoop


#: `TrainingLoop`'s own default, to tell a config the user chose from none
_DEFAULT_OPTIMIZER = {"name": "adamw", "lr": 3e-4}


def run(loop: TrainingLoop) -> tuple[Operator, dict[str, Any]]:
    """Train ``loop`` with Keras's ``compile`` and ``fit``, on JAX.

    One Keras epoch is one evaluation period of ``loop.eval_every_n_steps``
    steps, and a Keras callback stops ``fit`` at exactly ``loop.max_steps``. The
    same callback drives the loop's callbacks and metric writer, and resets
    Keras's metrics after each step, so the logged loss is the step's own and
    not the running mean since the period began.

    Returns:
        ``(trained_model_op, backend_info)``. ``trained_model_op`` is a copy of
        ``loop.model_op`` holding the trained model (Keras trains it in place).
        ``backend_info`` has ``backend``, ``keras_version``, ``keras_backend``,
        ``jax_version``, ``devices``, ``total_seconds``, ``final_step``,
        ``final_metrics`` (the last step) and ``final_eval_metrics`` (the last
        validation, empty without a ``val_dataset``).

    Raises:
        TypeError: If ``loop.model_op`` is not a `KerasModelOp`, or
            ``loop.task`` is not a `KerasTask`.
        ValueError: If there is nothing to compile the model with.
        NotImplementedError: If ``loop.sharding`` is set.
    """
    keras = import_keras()
    import jax

    t0 = time.time()
    if loop.sharding is not None:
        raise NotImplementedError(
            "backend='keras' does not shard yet; leave TrainingLoop.sharding unset."
        )
    if not isinstance(loop.model_op, KerasModelOp):
        raise TypeError(
            "backend='keras' trains a pipekit_train.backends.keras.KerasModelOp; "
            f"got {type(loop.model_op).__name__}."
        )
    model = loop.model_op.model
    task = _resolve_task(loop)

    source = BatchSource(loop.dataset, loop.batch_size, loop.seed)
    first = source.next_batch()
    if not task.keep_compile:
        _compile(model, loop, task, first)
    elif loop.optimizer_config != _DEFAULT_OPTIMIZER:
        warnings.warn(
            "KerasTask(keep_compile=True) keeps the model's own optimiser, so "
            "TrainingLoop.optimizer_config is not used.",
            UserWarning,
            stacklevel=2,
        )
    if getattr(model, "optimizer", None) is None:
        raise ValueError(
            "KerasTask(keep_compile=True) needs a compiled model; call "
            "model.compile(...) first."
        )

    period = loop.eval_every_n_steps
    validation: dict[str, Any] = {}
    if loop.val_dataset is not None:
        steps, batches = _validation_batches(loop.val_dataset, loop.batch_size)
        validation = {"validation_data": batches, "validation_steps": steps}

    bridge = _bridge(keras)(loop, period)
    try:
        model.fit(
            _train_batches(source, first),
            steps_per_epoch=period,
            epochs=math.ceil(loop.max_steps / period),
            callbacks=[bridge, *task.keras_callbacks],
            verbose=0,
            **validation,
        )
    finally:
        if loop.metric_writer is not None:
            loop.metric_writer.close()

    trained_model_op = copy.copy(loop.model_op)
    final_step = int(model.optimizer.iterations.numpy())
    _common.dispatch(
        loop.callbacks,
        "on_train_end",
        loop,
        _carry(trained_model_op, final_step, loop, bridge.last_metrics),
    )
    backend_info = {
        "backend": "keras",
        "keras_version": keras.__version__,
        "keras_backend": keras.config.backend(),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "total_seconds": time.time() - t0,
        "final_step": final_step,
        "final_metrics": dict(bridge.last_metrics),
        "final_eval_metrics": dict(bridge.last_eval_metrics),
    }
    return trained_model_op, backend_info


def _resolve_task(loop: TrainingLoop) -> KerasTask:
    task = loop.task
    if task is None:
        if loop.loss is None:
            raise ValueError(
                "backend='keras' needs loop.loss or loop.task: a pipekit_train.Loss, "
                "or a KerasTask with a Keras loss or keep_compile=True."
            )
        return KerasTask()
    if not isinstance(task, KerasTask):
        raise TypeError(
            "backend='keras' takes a pipekit_train.backends.keras.KerasTask as "
            f"loop.task; got {type(task).__name__}."
        )
    if task.loss is None and loop.loss is None and not task.keep_compile:
        raise ValueError(
            "KerasTask has no loss and loop.loss is unset; give one of them, or "
            "keep_compile=True."
        )
    return task


def _compile(model: Any, loop: TrainingLoop, task: KerasTask, first: Any) -> None:
    """Compile ``model`` with the task's loss and the loop's optimiser."""
    metrics = list(task.metrics)
    if task.loss is not None:
        loss = task.loss
    else:
        x, y = first
        names = aux_names(loop.loss, model(x, training=False), y)
        loss = bridge_loss(loop.loss)
        metrics = aux_metrics(loop.loss, names) + metrics
    model.compile(
        optimizer=build_keras_optimizer(loop.optimizer_config),
        loss=loss,
        metrics=metrics,
        steps_per_execution=1,
    )


def _train_batches(source: BatchSource, first: Any) -> Iterator[Any]:
    """The training batches, starting with the one read to find the metrics.

    ``fit`` reads exactly one batch per step, so the source's position after
    step k is k batches in.
    """
    yield first
    while True:
        yield source.next_batch()


def _validation_batches(dataset: Any, batch_size: int) -> tuple[int, Iterator[Any]]:
    """Batches of the validation set in order, the last one short, repeated.

    Returns:
        The number of batches in one pass, and an endless generator of them;
        ``fit`` reads one pass at each validation.
    """
    size = len(dataset) if is_indexable(dataset) else sum(1 for _ in dataset)
    if size == 0:
        raise ValueError("TrainingLoop.val_dataset is empty.")
    steps = math.ceil(size / batch_size)

    def batches() -> Iterator[Any]:
        while True:
            batch: list[Any] = []
            for sample in dataset:
                batch.append(sample)
                if len(batch) == batch_size:
                    yield stack_pairs(batch)
                    batch = []
            if batch:
                yield stack_pairs(batch)

    return steps, batches()


def _carry(
    model_op: Operator,
    step: int,
    loop: TrainingLoop,
    metrics: dict[str, float],
) -> Any:
    from pipekit_train.loop._loop import TrainerCarryState

    epoch = step // loop.steps_per_epoch if loop.steps_per_epoch else 0
    return TrainerCarryState(
        model=model_op, opt_state=None, step=step, epoch=epoch, metrics=metrics
    )


def _bridge(keras: Any) -> type:
    """The Keras callback that drives the loop (a class, since Keras is lazy)."""

    class _PipekitBridge(keras.callbacks.Callback):
        def __init__(self, loop: TrainingLoop, period: int) -> None:
            super().__init__()
            self.loop = loop
            self.period = period
            self.last_metrics: dict[str, float] = {}
            self.last_eval_metrics: dict[str, float] = {}

        def _step(self) -> int:
            return int(self.model.optimizer.iterations.numpy())

        def _state(self, metrics: dict[str, float]) -> Any:
            return _carry(self.loop.model_op, self._step(), self.loop, metrics)

        def on_train_begin(self, logs: Any = None) -> None:
            _common.dispatch(
                self.loop.callbacks, "on_train_begin", self.loop, self._state({})
            )

        def on_train_batch_end(self, batch: int, logs: Any = None) -> None:
            loop = self.loop
            step = self._step()
            metrics = {name: float(value) for name, value in (logs or {}).items()}
            self.last_metrics = metrics
            stop = step >= loop.max_steps
            # Keras logs the running mean since the epoch began; resetting makes
            # the next step log its own loss. The last step of an epoch keeps its
            # metrics, so Keras's epoch logs hold that step's values, not zeros.
            if not stop and batch + 1 < self.period:
                self.model.reset_metrics()
            if loop.metric_writer is not None and step % loop.log_every_n_steps == 0:
                loop.metric_writer.write(step, metrics)
            state = self._state(metrics)
            _common.dispatch(loop.callbacks, "on_step_end", loop, state, metrics)
            if loop.steps_per_epoch and step % loop.steps_per_epoch == 0:
                _common.dispatch(loop.callbacks, "on_epoch_end", loop, state, metrics)
            if stop or _common.any_should_stop(loop.callbacks):
                self.model.stop_training = True

        def on_test_end(self, logs: Any = None) -> None:
            loop = self.loop
            metrics = {
                name.removeprefix("val_"): float(value)
                for name, value in (logs or {}).items()
            }
            self.last_eval_metrics = metrics
            state = self._state(self.last_metrics)
            _common.dispatch(loop.callbacks, "on_eval_end", loop, state, metrics)
            if _common.any_should_stop(loop.callbacks):
                self.model.stop_training = True

    return _PipekitBridge
