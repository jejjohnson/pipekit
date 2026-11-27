"""`KerasTask`, the Keras backend's training target (ADRs D9 and D18)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class KerasTask:
    """What the Keras backend compiles a model with (``loop.task``).

    With no task, the backend makes one from ``loop.loss``.

    Attributes:
        loss: A Keras loss or its name, passed to ``compile``. ``None`` bridges
            ``loop.loss``, a `pipekit_train.Loss`, into a Keras loss.
        metrics: Keras metrics passed to ``compile``, beside the auxiliary
            terms of a bridged `pipekit_train.Loss`.
        keras_callbacks: Keras callbacks that ``fit`` runs beside the backend's
            own (Keras Tuner, TensorBoard, …).
        keep_compile: Keep the model's own ``compile``, for a model whose loss
            is part of it (a penalty added with ``add_loss``). The loop's
            ``loss`` and ``optimizer_config`` are then not used.
    """

    loss: Any = None
    metrics: tuple[Any, ...] = ()
    keras_callbacks: tuple[Any, ...] = ()
    keep_compile: bool = False
