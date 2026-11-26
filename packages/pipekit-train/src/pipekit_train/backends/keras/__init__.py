"""Keras 3 adapter — v0.3 scaffold.

In v0.1 this module ships as a stub; ``run`` raises
``NotImplementedError``. The planned implementation translates
``loop.loss`` into a ``keras.losses.Loss`` subclass, configures the
optimiser via ``keras.optimizers.get`` from ``optimizer_config``,
wraps ``loop.dataset`` as a ``tf.data.Dataset.from_generator``, then
calls ``model.compile`` and ``model.fit``.

See ``docs/design/api/adapters.md`` (Keras section) for the planned
design.
"""

from __future__ import annotations

from typing import Any


def run(loop: Any) -> tuple[Any, dict[str, Any]]:
    """Train ``loop`` with the Keras 3 backend.

    Args:
        loop: The `TrainingLoop` operator.

    Returns:
        ``(trained_model_op, backend_info)`` once implemented.

    Raises:
        NotImplementedError: Always in v0.1. Use ``backend="equinox"``
            when the Equinox adapter ships, or pin
            ``pipekit-train>=0.3`` when the Keras adapter lands.
    """
    raise NotImplementedError(
        "The Keras adapter is scheduled for v0.3. Use "
        "backend='equinox' for v0.1 (when implemented), or pin "
        "pipekit-train>=0.3 when this adapter lands. See "
        "packages/pipekit-train/docs/design/api/adapters.md."
    )
