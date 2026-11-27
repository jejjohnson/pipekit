"""The Keras 3 backend, on JAX only (``[keras]``; ADRs D18 to D21).

`run(loop)` trains a `KerasModelOp` with Keras's own ``compile`` and ``fit``
(D18): one Keras epoch is one evaluation period, and a Keras callback stops at
exactly ``max_steps`` and drives the loop's callbacks and metric writer. A
`KerasTask` says what to compile the model with; without one, the loop's
`pipekit_train.Loss` is bridged into a Keras loss. `KerasModelOp` wraps a Keras
model as an operator whose weights round-trip through a registry as a
byte-stable blob (D20). An Equinox model given as a ``JaxOperator``,
``JaxModelOp`` or `EquinoxModelOp` trains as an `EquinoxLayer`, a Keras layer
over copies of its arrays, and comes back in the wrapper it was given (D21).

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
    "EquinoxLayer": "pipekit_train.backends.keras._equinox",
    "KerasModelOp": "pipekit_train.backends.keras._model",
    "KerasTask": "pipekit_train.backends.keras._task",
    "build_keras_optimizer": "pipekit_train.backends.keras._optim",
    "import_keras": "pipekit_train.backends.keras._backend",
    "run": "pipekit_train.backends.keras._run",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EquinoxLayer",
    "KerasModelOp",
    "KerasTask",
    "build_keras_optimizer",
    "import_keras",
    "run",
]
