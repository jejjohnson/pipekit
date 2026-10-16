"""Hydra + hydra-zen adapter.

Bridges pipekit operator graphs and Hydra configs. Two directions:

- ``from_hydra_cfg(cfg)`` — instantiate a pipekit `Operator` from a
  Hydra/OmegaConf config (or any object with a ``_target_`` key).
  Uses ``hydra.utils.instantiate``.
- ``to_hydra_cfg(op)`` — build a hydra-zen ``builds(...)`` spec from
  an operator's `state` record. The resulting object round-trips
  through ``hydra-zen``'s ``to_yaml`` / ``instantiate``.
- ``from_yaml_sandboxed(text)`` — build a pipeline from untrusted
  ``_target_`` YAML, allowing only classes registered with
  `pipekit.register`. It never calls ``hydra.utils.instantiate``
  (which will call any dotted path): the YAML is translated into a
  pipekit state record and built by `pipekit.from_state_sandboxed`.

The adapter doesn't import ``hydra`` or ``hydra_zen`` at module load
— each method imports lazily so a user can ``from … import hydra as
hydra_adapter`` in code that branches on what's installed.

Behind ``pipekit-experiment[hydra]`` extra.

See master plan Report 12, section 3.3.
"""

from __future__ import annotations

import inspect
import re
import sys
from typing import Any

from pipekit import Operator, from_state_sandboxed


# Hydra's own keys: instructions to `instantiate`, not constructor arguments.
_HYDRA_KEYS = frozenset({"_target_", "_recursive_", "_convert_"})
# Keys a sandboxed config may not use: positional arguments and partials
# have no place in a pipekit state record.
_UNSUPPORTED_KEYS = frozenset({"_args_", "_partial_"})
# How Hydra configs name a class rather than call it: an exception class
# in `Try.on` / `Retry.on`.
_GET_CLASS = "hydra.utils.get_class"


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

    @staticmethod
    def from_yaml_sandboxed(text: str) -> Operator:
        """Build a pipeline from untrusted ``_target_`` YAML, safely.

        Only classes registered with `pipekit.register` can be built, at
        any depth (pipekit's own containers included); every class is
        checked before anything is constructed. The YAML is parsed with
        ``yaml.safe_load``, interpolations (``${...}``) are not resolved,
        and ``hydra.utils.instantiate`` is never called. Each ``_target_``
        must name an `Operator` class (or ``Class.from_config``) in an
        already-imported module; ``{_target_: hydra.utils.get_class,
        path: module.Name}`` names an exception class.

        Args:
            text: A YAML document whose root has a ``_target_``.

        Raises:
            ImportError: if PyYAML isn't installed.
            PermissionError: if a ``_target_`` isn't an `Operator` in a
                loaded module, or names a class that isn't registered.
            ValueError: if a node uses ``_args_`` or ``_partial_``, or the
                document is not a mapping with a ``_target_``.
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "from_yaml_sandboxed requires PyYAML. Install with "
                "`pip install pipekit-experiment[hydra]`."
            ) from e
        document = yaml.load(text, Loader=_yaml_12_bool_loader(yaml))
        if not isinstance(document, dict) or "_target_" not in document:
            raise ValueError(
                "from_yaml_sandboxed expects a YAML mapping with a `_target_`."
            )
        return from_state_sandboxed(_to_state(document, "<root>"))


def _yaml_12_bool_loader(yaml: Any) -> Any:
    """A `SafeLoader` that reads only true / false as booleans (YAML 1.2).

    YAML 1.1, which PyYAML follows, also reads on / off / yes / no as
    booleans, so `Try`'s and `Retry`'s ``on:`` key would load as `True`.
    """
    bool_tag = "tag:yaml.org,2002:bool"

    class Loader(yaml.SafeLoader):
        pass

    resolvers: dict[Any, list[Any]] = {}
    for first, entries in yaml.SafeLoader.yaml_implicit_resolvers.items():
        resolvers[first] = [entry for entry in entries if entry[0] != bool_tag]
    Loader.yaml_implicit_resolvers = resolvers
    Loader.add_implicit_resolver(
        bool_tag,
        re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
        list("tTfF"),
    )
    return Loader


def _to_state(node: Any, where: str) -> Any:
    """Translate a ``_target_`` tree into a pipekit state tree."""
    if isinstance(node, list):
        items: list[Any] = []
        for i, item in enumerate(node):
            items.append(_to_state(item, f"{where}[{i}]"))
        return items
    if not isinstance(node, dict):
        return node
    if "_target_" not in node:
        plain: dict[str, Any] = {}
        for key, value in node.items():
            plain[key] = _to_state(value, f"{where}.{key}")
        return plain

    unsupported = sorted(_UNSUPPORTED_KEYS & set(node))
    if unsupported:
        raise ValueError(f"{where}: {unsupported} are not supported in a pipeline.")
    target = node["_target_"]
    kwargs = {key: value for key, value in node.items() if key not in _HYDRA_KEYS}

    if target == _GET_CLASS:
        if set(kwargs) != {"path"}:
            raise ValueError(f"{where}: {_GET_CLASS} takes exactly one `path`.")
        return {"exception": kwargs["path"]}

    located = _locate_loaded(target, where)
    if isinstance(located, type) and issubclass(located, Operator):
        op_type = located
        config: dict[str, Any] = {}
        for key, value in kwargs.items():
            config[key] = _to_state(value, f"{where}.{key}")
    elif _is_from_config(located):
        op_type = located.__self__
        if set(kwargs) != {"config"} or not isinstance(kwargs["config"], dict):
            raise ValueError(f"{where}: `from_config` takes one `config` mapping.")
        config = _to_state(kwargs["config"], f"{where}.config")
    else:
        raise PermissionError(f"{where}: {target!r} is not an Operator class.")
    return {"module": op_type.__module__, "class": op_type.__name__, "config": config}


def _locate_loaded(target: Any, where: str) -> Any:
    """``"pkg.mod.Name"`` ↦ the object, from already-imported modules only."""
    if not isinstance(target, str) or not target:
        raise ValueError(f"{where}: `_target_` must be a dotted path.")
    parts = target.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module = sys.modules.get(".".join(parts[:split]))
        if module is None:
            continue
        obj: Any = module
        for part in parts[split:]:
            obj = getattr(obj, part, None)
        if obj is not None:
            return obj
    raise PermissionError(f"{where}: {target!r} is not in a loaded module.")


def _is_from_config(obj: Any) -> bool:
    """True for ``SomeOperator.from_config`` (a classmethod bound to the class)."""
    return (
        inspect.ismethod(obj)
        and obj.__name__ == "from_config"
        and isinstance(obj.__self__, type)
        and issubclass(obj.__self__, Operator)
    )
