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
  register on import: pipekit-array registers `DimArray`; geoarray
  registers its carriers.
- `register_lazy_adapter` — a loader for a carrier family whose library
  may not be imported yet: pipekit-array registers one for
  ``xr.DataArray``, so importing it never imports xarray, and the adapter
  loads the first time a DataArray arrives.
- `signature_of` — a carrier's `Signature`, through its adapter.
- `MissingAdapterError` — a known carrier family with nothing registered.
- `backend_name`, `backend_of`, `BackendError` — which array backend a
  raw array or a carrier sits on (``"numpy"``, ``"jax"``, ``"dask"``),
  for ``Operator.backends`` and pipekit-array's kernels.

The registry lives here, in pipekit core, because pipekit-array already
depends on pipekit: a lookup from core into pipekit-array would be a
cycle (PK-P1). pipekit core ships no adapter itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from pipekit.signature import Signature


__all__ = [
    "BackendError",
    "MissingAdapterError",
    "NamedArrayAdapter",
    "backend_name",
    "backend_of",
    "get_adapter",
    "register_adapter",
    "register_lazy_adapter",
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

#: module prefix of a carrier type -> a loader that registers its adapter
_LAZY: dict[str, Callable[[], None]] = {}

#: carriers whose adapter ships behind an extra, by exact type
#: (``module.qualname``) -> how to get it. Only carriers: an ``xr.Dataset``
#: or ``xr.Variable`` is not one, so it passes through as a plain value.
_EXTRA_HINTS: dict[str, str] = {
    "xarray.core.dataarray.DataArray": (
        "pip install 'pipekit-array[xarray]', then import pipekit_array"
    ),
}


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


def register_lazy_adapter(module_prefix: str, loader: Callable[[], None]) -> None:
    """Register ``loader`` for carrier types defined under ``module_prefix``.

    The first time `get_adapter` meets an unregistered type whose module
    starts with ``module_prefix`` (``"xarray."``), it calls ``loader``
    once, which registers the real adapter, and looks again. The carrier's
    library is already imported by then, so nothing is imported early.
    """
    _LAZY[module_prefix] = loader


def unregister_adapter(carrier_type: type) -> None:
    """Remove the adapter registered for exactly ``carrier_type``, if any."""
    _REGISTRY.pop(carrier_type, None)


def get_adapter(x: Any) -> NamedArrayAdapter | None:
    """The adapter for ``x``'s type, or ``None`` for an unnamed value.

    Looks the type up along its MRO, so a subclass gets its base's
    adapter unless it has its own (the closest registration wins).

    Raises:
        MissingAdapterError: if ``x`` is a carrier whose adapter ships
            behind an extra and is not registered (an ``xr.DataArray``
            without ``pipekit-array[xarray]``, or before ``pipekit_array``
            is imported), so it is never silently treated as a raw array.
            Other values of that library (an ``xr.Dataset``) return ``None``.

    Example:
        >>> get_adapter([1.0, 2.0]) is None
        True
    """
    cls = type(x)
    adapter = _lookup(cls)
    if adapter is not None:
        return adapter
    # a carrier's library (or a subclass's base library) may load its adapter
    modules = [base.__module__ for base in cls.__mro__]
    for prefix in list(_LAZY):
        if any(module.startswith(prefix) for module in modules):
            # load once: a loader that registers nothing is not retried
            _LAZY.pop(prefix)()
            adapter = _lookup(cls)
            if adapter is not None:
                return adapter
    # a subclass of a known carrier gets its base's hint
    for base in cls.__mro__:
        name = f"{base.__module__}.{base.__qualname__}"
        hint = _EXTRA_HINTS.get(name)
        if hint is not None:
            raise MissingAdapterError(
                f"{cls.__module__}.{cls.__qualname__} needs its named-array "
                f"adapter: {hint}."
            )
    return None


def _lookup(cls: type) -> NamedArrayAdapter | None:
    for base in cls.__mro__:
        adapter = _REGISTRY.get(base)
        if adapter is not None:
            return adapter
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


class BackendError(TypeError):
    """Inputs on more than one backend, or a backend the code does not run on."""


def backend_name(data: Any) -> str:
    """``"jax"``, ``"dask"`` or ``"numpy"`` for raw data (tracers count as JAX).

    Example:
        >>> backend_name([1.0, 2.0])
        'numpy'
    """
    module = type(data).__module__
    if module.startswith(("jax", "jaxlib")) or "Tracer" in type(data).__name__:
        return "jax"
    if module.startswith("dask."):
        return "dask"
    return "numpy"


_ARRAY_MODULES = ("numpy", "jax", "jaxlib", "dask.")


def backend_of(x: Any) -> str | None:
    """The backend of a carrier or a raw array; ``None`` for any other value.

    A registered carrier answers through its adapter's raw data; a NumPy,
    JAX (tracers included) or dask array answers for itself. Anything else
    (a number, a string, a table) has no backend.

    Example:
        >>> import numpy as np
        >>> backend_of(np.zeros(3)), backend_of("scene.tif")
        ('numpy', None)
    """
    adapter = get_adapter(x)
    if adapter is not None:
        return backend_name(adapter.data(x))
    cls = type(x)
    if cls.__module__.startswith(_ARRAY_MODULES) or "Tracer" in cls.__name__:
        return backend_name(x)
    return None
