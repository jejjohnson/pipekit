"""Frozen records of results, registered as JAX pytrees once JAX loads.

A library whose import never loads JAX registers a record at once when JAX is
already loaded, and otherwise when something first imports it
(`pipekit_ops.on_import`).
"""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any, cast

from pipekit_ops import on_import


__all__ = ["register_record"]


def register_record[T: type](cls: T, *, meta_fields: tuple[str, ...] = ()) -> T:
    """Register a frozen dataclass as a JAX pytree, now or once JAX loads.

    Without JAX the record is never registered, and works as it is.

    Args:
        cls: A frozen dataclass whose fields are arrays, numbers or records.
        meta_fields: The names of the fields that are static under JAX: they
            are compared and hashed rather than traced.

    Returns:
        ``cls`` itself.

    Example:
        >>> @register_record
        ... @dataclasses.dataclass(frozen=True)
        ... class Trend:
        ...     slope: float
        >>> Trend(2.0)
        Trend(slope=2.0)
    """
    names = [field.name for field in dataclasses.fields(cast("Any", cls))]
    data_fields = [name for name in names if name not in meta_fields]

    def register(jax: ModuleType) -> None:
        jax.tree_util.register_dataclass(
            cls, data_fields=data_fields, meta_fields=list(meta_fields)
        )

    on_import("jax", register)
    return cls
