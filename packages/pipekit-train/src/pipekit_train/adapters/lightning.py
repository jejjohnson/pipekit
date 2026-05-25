"""PyTorch Lightning adapter — v0.2 scaffold.

In v0.1 this module ships as a stub; ``run`` raises
``NotImplementedError``. The planned implementation synthesises a
``LightningModule`` from ``loop.model_op`` + ``loop.loss`` and a
``LightningDataModule`` from ``loop.dataset`` / ``loop.val_dataset``,
then calls ``Trainer.fit``.

See ``docs/design/api/adapters.md`` (Lightning section) for the
planned design.
"""

from __future__ import annotations

from typing import Any


def run(loop: Any) -> tuple[Any, dict[str, Any]]:
    """Train ``loop`` with the Lightning backend.

    Args:
        loop: The `TrainingLoop` operator.

    Returns:
        ``(trained_model_op, backend_info)`` once implemented.

    Raises:
        NotImplementedError: Always in v0.1. Use ``backend="equinox"``
            when the Equinox adapter ships, or pin
            ``pipekit-train>=0.2`` when the Lightning adapter lands.
    """
    raise NotImplementedError(
        "The Lightning adapter is scheduled for v0.2. Use "
        "backend='equinox' for v0.1 (when implemented), or pin "
        "pipekit-train>=0.2 when this adapter lands. See "
        "packages/pipekit-train/docs/design/api/adapters.md."
    )
