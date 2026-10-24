"""Array API namespace dispatch shim.

The single point of backend dispatch for every pipekit-array
operator. `array_namespace(*xs)` returns the Array API
namespace (numpy, jax.numpy, torch via array-api-compat, …)
shared by all input arrays.

Tries the optional `array-api-compat` library first — without it,
PyTorch tensors don't expose `__array_namespace__` natively and torch
backends won't dispatch. Falls back to native
`x.__array_namespace__()` lookup if the compat shim isn't installed.

See `docs/design/architecture.md` §4 and ADR A1 for the rationale.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


_NO_BACKEND_MSG = (
    "No input implements the Array API. Install one of: "
    "pipekit-array[numpy], pipekit-array[jax], "
    "pipekit-array[torch], pipekit-array[cupy], "
    "pipekit-array[dask]. PyTorch additionally needs "
    "pipekit-array[compat] for namespace dispatch."
)


def _multi_backend_msg(names: list[str]) -> str:
    return (
        f"Inputs span multiple Array API backends: {names}. "
        "pipekit-array operators require a single backend per call."
    )


_UNSET: Any = object()
_compat: Any = _UNSET


def _compat_namespace() -> Callable[..., Any] | None:
    """``array_api_compat.array_namespace``, or None; looked up once.

    A failed import is not cached by Python, so trying it on every call
    cost about 0.2 ms per operator call (KernelOp's PK-K5 budget is 25 µs).
    """
    global _compat
    if _compat is _UNSET:
        try:
            import array_api_compat  # ty: ignore[unresolved-import]

            _compat = array_api_compat.array_namespace
        except ImportError:
            _compat = None
    return _compat


def array_namespace(*xs: Any) -> Any:
    """Return the Array API namespace shared by all input arrays.

    Args:
        *xs: One or more array-like objects. All must share the same
            Array API backend.

    Returns:
        The module-like namespace object (e.g. `numpy`, `jax.numpy`,
        `array_api_compat.torch`).

    Raises:
        TypeError: If no input implements the Array API, or if the
            inputs span more than one backend. The error message is
            consistent whether or not `array-api-compat` is installed.
    """
    compat_ns = _compat_namespace()

    if compat_ns is not None:
        try:
            return compat_ns(*xs)
        except TypeError as exc:
            # array-api-compat's error messages differ from our
            # contract; re-raise with stable wording so tests and
            # users see a consistent shape.
            msg = str(exc)
            if "Multiple namespaces" in msg:
                names = sorted({type(x).__module__.split(".")[0] for x in xs})
                raise TypeError(_multi_backend_msg(names)) from exc
            raise TypeError(_NO_BACKEND_MSG) from exc

    namespaces: set[Any] = set()
    for x in xs:
        ns = getattr(x, "__array_namespace__", None)
        if ns is None:
            raise TypeError(
                f"Input of type {type(x).__name__!r} does not implement "
                f"the Array API (`__array_namespace__` missing). " + _NO_BACKEND_MSG
            )
        namespaces.add(ns())
    if len(namespaces) == 0:
        raise TypeError(_NO_BACKEND_MSG)
    if len(namespaces) > 1:
        names = sorted({ns.__name__ for ns in namespaces})
        raise TypeError(_multi_backend_msg(names))
    return next(iter(namespaces))
