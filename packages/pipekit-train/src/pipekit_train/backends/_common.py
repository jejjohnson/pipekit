"""Backend-agnostic helpers shared by every training adapter.

Deliberately dependency-free at module scope (no jax / optax / numpyro
imports) so any adapter — behind any optional extra — can import it
without pulling in another backend's dependencies. `bayes` and the
Equinox adapter both build on these.
"""

from __future__ import annotations

import os
from typing import Any


# optax optimiser names supported by ``optimizer_config``. One tuple for
# every backend, so support can never silently diverge between adapters.
OPTIMIZER_NAMES: tuple[str, ...] = ("adam", "adamw", "sgd", "rmsprop")


def build_optimizer(config: dict[str, Any] | None, *, default_name: str) -> Any:
    """Translate ``optimizer_config`` into an ``optax.GradientTransformation``.

    Supported names: ``adam``, ``adamw``, ``sgd``, ``rmsprop``. The rest
    of the config is forwarded as kwargs to the constructor. ``lr`` is
    normalised to ``learning_rate`` (the design docs use the shorter
    form; optax uses the longer one). An empty / ``None`` config falls
    back to ``{"name": default_name, "lr": 1e-3}``.

    Args:
        config: The `TrainingLoop.optimizer_config` dict.
        default_name: Optimiser used when the config names none — the
            per-backend default (``"adamw"`` for Equinox, ``"adam"`` for
            the Bayesian backends).

    Raises:
        ValueError: For an optimiser name outside `OPTIMIZER_NAMES`.
    """
    import optax  # ty: ignore[unresolved-import]

    config = dict(config) if config else {"name": default_name, "lr": 1e-3}
    name = config.pop("name", default_name)
    if "lr" in config and "learning_rate" not in config:
        config["learning_rate"] = config.pop("lr")
    if name not in OPTIMIZER_NAMES:
        raise ValueError(
            f"Unknown optimizer {name!r}. Supported: {sorted(OPTIMIZER_NAMES)}."
        )
    return getattr(optax, name)(**config)


def dispatch(callbacks: tuple[Any, ...], hook: str, *args: Any) -> None:
    """Call ``hook`` on each callback that implements it."""
    for cb in callbacks:
        fn = getattr(cb, hook, None)
        if fn is not None:
            fn(*args)


def any_should_stop(callbacks: tuple[Any, ...]) -> bool:
    """True if any callback signals ``should_stop`` (e.g. EarlyStopping)."""
    return any(getattr(cb, "should_stop", False) for cb in callbacks)


def resolve_checkpoint_dir(checkpoint_dir: str | None, callback: Any) -> str | None:
    """Return the absolute directory for checkpoints, or ``None`` for no saving.

    Args:
        checkpoint_dir: ``TrainingLoop.checkpoint_dir``.
        callback: The loop's `Checkpoint` callback, if any; its ``save_dir`` is
            used when the loop sets no directory.

    Returns:
        The absolute local path, or ``None`` if neither sets a directory.

    Raises:
        ValueError: If the two name different directories, or either is a URI,
            because the data-iterator side-car is written with local paths.
    """
    save_dir = getattr(callback, "save_dir", None)
    for candidate in (checkpoint_dir, save_dir):
        if candidate and "://" in candidate:
            raise ValueError(
                f"Checkpoint directories must be local paths; got {candidate!r}."
            )
    directory = checkpoint_dir or save_dir
    if not directory:
        return None
    resolved = os.path.abspath(directory)
    if checkpoint_dir and save_dir and os.path.abspath(save_dir) != resolved:
        raise ValueError(
            f"TrainingLoop.checkpoint_dir ({checkpoint_dir!r}) and "
            f"Checkpoint.save_dir ({save_dir!r}) name different directories. "
            "Set one of them."
        )
    return resolved
