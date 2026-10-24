"""Named-array adapters — how carrier-agnostic code reaches a carrier.

A carrier is anything with named dims and a raw array inside: a
`pipekit_array.DimArray`, an ``xr.DataArray``, a geoarray ``GeoArray``.
Code that works on any of them (``KernelOp``, the dim-level functions,
``Sequential.summary``) reaches it through an *adapter* registered per
carrier type, not through methods on the carrier, so no carrier library
imports pipekit in its core and ``xr.DataArray`` needs no accessor.

- `NamedArrayAdapter` — the protocol every adapter satisfies.
- `register_adapter` / `unregister_adapter` / `get_adapter` — the one
  registry, looked up along the carrier type's MRO. Carrier libraries
  register on import: pipekit-array registers `DimArray` (and, with its
  ``[xarray]`` extra, ``xr.DataArray``); geoarray registers its carriers.
- `signature_of` — a carrier's `Signature`, through its adapter.
- `MissingAdapterError` — a known carrier family with nothing registered.

The registry lives here, in pipekit core, because pipekit-array already
depends on pipekit: a lookup from core into pipekit-array would be a
cycle (PK-P1). pipekit core ships no adapter itself.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pipekit.signature import Signature


__all__ = [
    "MissingAdapterError",
    "NamedArrayAdapter",
    "get_adapter",
    "register_adapter",
    "signature_of",
    "unregister_adapter",
]


@runtime_checkable
class NamedArrayAdapter(Protocol):
    """What carrier-agnostic code needs from one carrier family.

    Every method takes the carrier as its first argument, so one adapter
    instance serves every carrier of its type.
    """

    def dims(self, x: Any) -> tuple[str, ...]:
        """The dim names, one per axis, in axis order: ``("band", "y", "x")``."""
        ...

    def data(self, x: Any) -> Any:
        """The raw array: NumPy, JAX or dask."""
        ...

    def labels(self, x: Any, dim: str) -> tuple[Any, ...] | None:
        """The static labels of one dim (band names, level values), or None."""
        ...

    def isel(self, x: Any, indexers: dict[str, Any]) -> Any:
        """Select by position on named dims.

        Slices the data and every coordinate, label and geometry leaf on
        those dims together; an integer index drops its dim.
        """
        ...

    def rewrap(self, x: Any, data: Any, dims: tuple[str, ...]) -> Any:
        """A carrier around new data, built on ``x``.

        Dims kept from ``x`` keep their metadata, dropped dims lose it,
        new dims get none. Never used for selection. A geo carrier whose
        spatial dim was consumed may return a different carrier (a
        ``DimArray``).
        """
        ...

    def check_compatible(self, *xs: Any) -> tuple[Any, ...]:
        """Every carrier back, checked to combine element by element.

        Raises when they cannot (a different grid, different coordinates).
        A check that can only run when a compiled program runs (a traced
        transform under ``jit``) is attached to the carriers handed back,
        so callers must use what this returns.
        """
        ...

    def signature(self, x: Any) -> Signature:
        """The carrier's named dims, sizes and dtype."""
        ...


_REGISTRY: dict[type, NamedArrayAdapter] = {}

#: carrier families whose adapter ships behind an extra:
#: module prefix of the carrier type -> what to install
_EXTRA_HINTS: dict[str, str] = {"xarray.": "pipekit-array[xarray]"}


class MissingAdapterError(TypeError):
    """A carrier from a known family arrived with no adapter registered."""


def register_adapter(carrier_type: type, adapter: NamedArrayAdapter) -> None:
    """Register ``adapter`` for ``carrier_type`` and its subclasses.

    Re-registering a type replaces its adapter.

    Raises:
        TypeError: if ``carrier_type`` is not a class, or ``adapter`` does
            not satisfy `NamedArrayAdapter`.
    """
    if not isinstance(carrier_type, type):
        raise TypeError(
            f"register_adapter needs a class, got {type(carrier_type).__name__}."
        )
    if not isinstance(adapter, NamedArrayAdapter):
        raise TypeError(
            f"{type(adapter).__name__} does not satisfy NamedArrayAdapter "
            "(dims, data, labels, isel, rewrap, check_compatible, signature)."
        )
    _REGISTRY[carrier_type] = adapter


def unregister_adapter(carrier_type: type) -> None:
    """Remove the adapter registered for exactly ``carrier_type``, if any."""
    _REGISTRY.pop(carrier_type, None)


def get_adapter(x: Any) -> NamedArrayAdapter | None:
    """The adapter for ``x``'s type, or ``None`` for an unnamed value.

    Looks the type up along its MRO, so a subclass gets its base's
    adapter unless it has its own (the closest registration wins).

    Raises:
        MissingAdapterError: if ``x`` belongs to a carrier family whose
            adapter ships behind an extra (an ``xr.DataArray`` without
            ``pipekit-array[xarray]``), so it is never silently treated
            as a raw array.

    Example:
        >>> get_adapter([1.0, 2.0]) is None
        True
    """
    cls = type(x)
    for base in cls.__mro__:
        adapter = _REGISTRY.get(base)
        if adapter is not None:
            return adapter
    for prefix, extra in _EXTRA_HINTS.items():
        if cls.__module__.startswith(prefix):
            raise MissingAdapterError(
                f"{cls.__module__}.{cls.__qualname__} needs its named-array "
                f"adapter: pip install '{extra}'."
            )
    return None


def signature_of(x: Any) -> Any:
    """``x``'s `Signature` if it is a registered carrier; ``x`` itself otherwise.

    A `Signature` and ``None`` pass through unchanged, and so does any
    other value, so domain libraries that thread their own signature
    types keep working.
    """
    if x is None or isinstance(x, Signature):
        return x
    adapter = get_adapter(x)
    if adapter is None:
        return x
    return adapter.signature(x)
