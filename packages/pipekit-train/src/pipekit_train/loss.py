"""Loss protocol.

The carrier-agnostic loss surface from the v0.1 design. Common
`Loss` operators (`MSE`, `NLL`, `KL`, `Composite`) are not shipped
in this scaffold; they land alongside the v0.1 implementation of
the Equinox adapter. See ``docs/design/api/primitives.md``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Loss(Protocol):
    """Loss function — per-batch scalar called by the adapter.

    Implementations may return either a scalar loss directly, or a
    ``(scalar, aux_metrics_dict)`` tuple. The adapter distinguishes
    by introspection.

    See ``docs/design/api/primitives.md`` and ADR D4 in
    ``docs/design/decisions.md``.
    """

    def __call__(
        self,
        predicted: Any,
        target: Any,
    ) -> float | tuple[float, dict[str, float]]: ...
