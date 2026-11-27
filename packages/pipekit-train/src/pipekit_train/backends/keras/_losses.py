"""A `pipekit_train.Loss` as a Keras loss, with its auxiliary terms as metrics."""

from __future__ import annotations

import inspect
from typing import Any

from pipekit_ops import Operator

from pipekit_train.backends.keras._backend import import_keras
from pipekit_train.losses import call_loss


def _scalar(value: Any) -> Any:
    """Average a loss value down to a scalar, as the Equinox backend does."""
    keras = import_keras()
    value = keras.ops.convert_to_tensor(value)
    return keras.ops.mean(value) if len(value.shape) > 0 else value


def requires_key(loss: Any) -> bool:
    """True if ``loss`` cannot be called without a PRNG ``key``.

    A Keras loss receives no key, so such a loss cannot train under this
    backend yet (#330). A ``key`` that defaults to ``None``, as `KL`'s does for
    its analytic divergence, is fine.
    """
    target = loss
    if isinstance(loss, Operator) and type(loss).__call__ is Operator.__call__:
        target = loss._apply
    try:
        parameter = inspect.signature(target).parameters.get("key")
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.default is inspect.Parameter.empty


def bridge_loss(loss: Any) -> Any:
    """Return a Keras loss that calls ``loss(predicted, target)``.

    Keras calls a loss as ``(y_true, y_pred)``, and a `pipekit_train.Loss` as
    ``(predicted, target)``, so the bridge swaps them. It returns the loss's
    scalar; the auxiliary terms become metrics (`aux_metrics`).
    """
    keras = import_keras()

    class _LossBridge(keras.losses.Loss):
        def __init__(self) -> None:
            super().__init__(name=type(loss).__name__.lower())

        def call(self, y_true: Any, y_pred: Any) -> Any:
            result = call_loss(loss, y_pred, y_true)
            return _scalar(result[0] if isinstance(result, tuple) else result)

    return _LossBridge()


def aux_names(loss: Any, predicted: Any, target: Any) -> list[str]:
    """The names of ``loss``'s auxiliary terms, read by calling it once."""
    result = call_loss(loss, predicted, target)
    return sorted(result[1]) if isinstance(result, tuple) else []


def aux_metrics(loss: Any, names: list[str]) -> list[Any]:
    """One Keras metric per auxiliary term of ``loss``, averaged per sample.

    Each batch's value counts for the batch's size, so over a validation set
    with a short last batch the metric is the mean over the samples, not the
    mean of the batch means.
    """
    keras = import_keras()

    class _AuxMetric(keras.metrics.Metric):
        def __init__(self, aux_name: str) -> None:
            super().__init__(name=aux_name)
            self.aux_name = aux_name
            self.total = self.add_variable(shape=(), initializer="zeros", name="total")
            self.count = self.add_variable(shape=(), initializer="zeros", name="count")

        def update_state(
            self, y_true: Any, y_pred: Any, sample_weight: Any = None
        ) -> None:
            del sample_weight
            value = _scalar(call_loss(loss, y_pred, y_true)[1][self.aux_name])
            size = keras.ops.cast(keras.ops.shape(y_true)[0], self.dtype)
            self.total.assign(self.total + keras.ops.cast(value, self.dtype) * size)
            self.count.assign(self.count + size)

        def result(self) -> Any:
            return keras.ops.divide_no_nan(self.total, self.count)

        def reset_state(self) -> None:
            self.total.assign(0.0)
            self.count.assign(0.0)

    return [_AuxMetric(name) for name in names]
