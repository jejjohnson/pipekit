"""The Keras 3 backend, on JAX only (``[keras]``; ADRs D18 to D21).

`KerasModelOp` wraps a Keras model as an operator whose weights round-trip
through a registry as a byte-stable blob (D20). ``run`` is still a stub that
raises ``NotImplementedError``.

Importing this area imports no Keras: Keras reads ``KERAS_BACKEND`` once, at its
first import, so the names below are loaded on first use, and that first use
chooses the JAX backend (D19).

See ``docs/design/keras_adapter.md``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


#: each public name and the module that defines it
_LAZY: dict[str, str] = {
    "KerasModelOp": "pipekit_train.backends.keras._model",
    "import_keras": "pipekit_train.backends.keras._backend",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run(loop: Any) -> tuple[Any, dict[str, Any]]:
    """Train ``loop`` with the Keras 3 backend.

    Args:
        loop: The `TrainingLoop` operator.

    Returns:
        ``(trained_model_op, backend_info)`` once implemented.

    Raises:
        NotImplementedError: Always, until the backend's ``run`` lands (#320).
    """
    raise NotImplementedError(
        "The Keras adapter is scheduled for v0.3. Use "
        "backend='equinox' for v0.1 (when implemented), or pin "
        "pipekit-train>=0.3 when this adapter lands. See "
        "packages/pipekit-train/docs/design/api/adapters.md."
    )


__all__ = ["KerasModelOp", "import_keras", "run"]
