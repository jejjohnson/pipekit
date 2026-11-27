"""``optimizer_config`` as a Keras optimiser that matches optax's (design §8.4)."""

from __future__ import annotations

from typing import Any

from pipekit_train.backends._common import OPTIMIZER_NAMES
from pipekit_train.backends.keras._backend import import_keras


#: the Keras class of each optimiser name
_CLASSES = {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD", "rmsprop": "RMSprop"}

#: optax's argument names, and Keras's for the same thing
_RENAMES = {
    "lr": "learning_rate",
    "b1": "beta_1",
    "b2": "beta_2",
    "eps": "epsilon",
    "decay": "rho",
}

#: optax's defaults where Keras's differ, so one config is one optimiser
_OPTAX_DEFAULTS: dict[str, dict[str, float]] = {
    "adam": {"epsilon": 1e-8},
    "adamw": {"epsilon": 1e-8, "weight_decay": 1e-4},
    "rmsprop": {"epsilon": 1e-8},
    "sgd": {},
}


def build_keras_optimizer(config: dict[str, Any] | None) -> Any:
    """Translate ``optimizer_config`` into a Keras optimiser.

    The names are those of every backend (``adam``, ``adamw``, ``sgd``,
    ``rmsprop``). optax's argument names become Keras's, and where the config
    is silent the optimiser takes optax's default rather than Keras's, so the
    same config trains the same way on the Equinox and Keras backends. An
    empty config is ``{"name": "adamw", "lr": 1e-3}``, as for Equinox.

    Raises:
        ValueError: For a name outside ``OPTIMIZER_NAMES``.
    """
    keras = import_keras()
    config = dict(config) if config else {"name": "adamw", "lr": 1e-3}
    name = config.pop("name", "adamw")
    if name not in OPTIMIZER_NAMES:
        raise ValueError(
            f"Unknown optimizer {name!r}. Supported: {sorted(OPTIMIZER_NAMES)}."
        )
    kwargs = {_RENAMES.get(key, key): value for key, value in config.items()}
    kwargs.setdefault("learning_rate", 1e-3)
    for key, value in _OPTAX_DEFAULTS[name].items():
        kwargs.setdefault(key, value)
    return getattr(keras.optimizers, _CLASSES[name])(**kwargs)
