"""Hydra + hydra-zen adapter.

Bridges pipekit operator graphs and Hydra configs. Two directions:

- ``from_hydra_cfg(cfg)`` — instantiate a pipekit `Operator` from a
  Hydra/OmegaConf config (or any object with a ``_target_`` key).
  Uses ``hydra.utils.instantiate``.
- ``to_hydra_cfg(op)`` — build a hydra-zen ``builds(...)`` spec from
  an operator's `state` record. The resulting object round-trips
  through ``hydra-zen``'s ``to_yaml`` / ``instantiate``.

The adapter doesn't import ``hydra`` or ``hydra_zen`` at module load
— each method imports lazily so a user can ``from … import hydra as
hydra_adapter`` in code that branches on what's installed.

Behind ``pipekit-experiment[hydra]`` extra.

See master plan Report 12, section 3.3.
"""

from __future__ import annotations

from typing import Any

from pipekit import Operator


def _require_hydra() -> Any:
    try:
        import hydra.utils as hu  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "HydraConfigLoader requires hydra-core. Install with "
            "`pip install pipekit-experiment[hydra]`."
        ) from e
    return hu


def _require_hydra_zen() -> Any:
    try:
        import hydra_zen  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "to_hydra_cfg requires hydra-zen. Install with "
            "`pip install pipekit-experiment[hydra]`."
        ) from e
    return hydra_zen


class HydraConfigLoader:
    """Bidirectional bridge between pipekit operators and Hydra configs.

    Stateless — every method is a ``@staticmethod``. Construct once
    and reuse, or call the static methods directly.
    """

    @staticmethod
    def from_hydra_cfg(cfg: Any) -> Operator:
        """Instantiate an `Operator` from a Hydra-flavoured config.

        ``cfg`` may be:
          - An ``omegaconf.DictConfig`` / ``ListConfig``.
          - A plain ``dict`` with a ``_target_`` key.
          - Any object that ``hydra.utils.instantiate`` accepts.

        Raises:
            ImportError: if ``hydra-core`` isn't installed.
            TypeError: if the instantiation result isn't an `Operator`.
        """
        hu = _require_hydra()
        obj = hu.instantiate(cfg)
        if not isinstance(obj, Operator):
            raise TypeError(
                "from_hydra_cfg expected the config to instantiate to an "
                f"Operator, got {type(obj).__name__}."
            )
        return obj

    @staticmethod
    def to_hydra_cfg(op: Operator) -> Any:
        """Produce a hydra-zen ``builds(...)`` spec for ``op``.

        The returned object is the hydra-zen-generated dataclass that
        hydra-zen's ``to_yaml`` and ``instantiate`` understand. For a
        plain operator with primitive-typed config the round-trip is
        lossless; operators with closures / callables (Lambda, Tap, …)
        emit a debug-only spec.

        Raises:
            ImportError: if ``hydra-zen`` isn't installed.
            TypeError: if ``op`` isn't an `Operator`.
        """
        if not isinstance(op, Operator):
            raise TypeError(
                f"to_hydra_cfg expects an Operator, got {type(op).__name__}."
            )
        hz = _require_hydra_zen()
        return hz.builds(type(op), populate_full_signature=False, **op.get_config())

    @staticmethod
    def to_yaml(op: Operator) -> str:
        """Render ``op`` as Hydra-compatible YAML text via hydra-zen."""
        hz = _require_hydra_zen()
        cfg = HydraConfigLoader.to_hydra_cfg(op)
        return hz.to_yaml(cfg)

    @staticmethod
    def from_yaml(text: str) -> Operator:
        """Parse a Hydra YAML string back into an `Operator`.

        Uses OmegaConf to load the YAML, then `from_hydra_cfg` to
        instantiate. Requires both ``hydra-core`` and ``omegaconf``
        (the latter is a transitive dep of ``hydra-core``).
        """
        try:
            from omegaconf import OmegaConf  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "from_yaml requires omegaconf (a transitive dep of "
                "hydra-core). Install with "
                "`pip install pipekit-experiment[hydra]`."
            ) from e
        cfg = OmegaConf.create(text)
        return HydraConfigLoader.from_hydra_cfg(cfg)
