"""Run code when a module is imported, without importing it.

`on_import` lets a library hook into an optional dependency (register a
pytree with JAX, a converter with xarray) without paying for that import
itself: the callback runs now when the module is already imported, and
otherwise once something else imports it. Importing pipekit-array or
geoarray therefore never imports JAX, yet their carriers are JAX pytrees
from the moment JAX is loaded.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import threading
import warnings
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any


__all__ = ["on_import"]

_PENDING: dict[str, list[Callable[[ModuleType], object]]] = {}
_LOCK = threading.RLock()


def on_import(name: str, callback: Callable[[ModuleType], object]) -> None:
    """Call ``callback(module)`` once ``name`` is imported, without importing it.

    If ``name`` is already in ``sys.modules`` the callback runs now;
    otherwise it runs right after the module's first import finishes,
    whoever imports it. Each callback runs once. An exception from the
    callback is reported as a `RuntimeWarning` and never breaks the import
    of ``name``.

    Args:
        name: A top-level module name, e.g. ``"jax"``.
        callback: Called with the imported module.

    Example:
        >>> from pipekit import on_import
        >>> seen = []
        >>> on_import("json", lambda module: seen.append(module.__name__))
        >>> import json
        >>> seen
        ['json']
    """
    with _LOCK:
        module = sys.modules.get(name)
        if module is None:
            _PENDING.setdefault(name, []).append(callback)
            if not any(isinstance(f, _HookFinder) for f in sys.meta_path):
                sys.meta_path.insert(0, _HookFinder())
            return
    _run(name, module, [callback])


def _run(
    name: str, module: ModuleType, callbacks: list[Callable[[ModuleType], object]]
) -> None:
    for callback in callbacks:
        try:
            callback(module)
        except Exception as exc:  # never break the import of `name`
            warnings.warn(
                f"on_import({name!r}) callback {callback!r} failed: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )


class _HookFinder(importlib.abc.MetaPathFinder):
    """Finds nothing itself: wraps the loader of a module someone waits for."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in _PENDING:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = _HookLoader(spec.loader, fullname)
        return spec


class _HookLoader(importlib.abc.Loader):
    """The module's own loader, then the waiting callbacks."""

    def __init__(self, loader: Any, name: str) -> None:
        self._loader = loader
        self._name = name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Any:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        with _LOCK:
            callbacks = _PENDING.pop(self._name, [])
        _run(self._name, module, callbacks)

    def __getattr__(self, attr: str) -> Any:
        # get_resource_reader, is_package, get_code, …: the real loader's
        return getattr(self._loader, attr)
