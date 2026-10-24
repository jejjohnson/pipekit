"""Combinators — `StackAlong`, `ConcatenateAlong`.

Thin wrappers over `xp.stack` and `xp.concat` (the Array API
standardised names) so the operations compose into a pipeline. See
`docs/design/api/operators.md` for the spec.

`ApplyToBands` lands in Phase B alongside `BatchedMap`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pipekit import Operator

from pipekit_array import dims
from pipekit_array._namespace import array_namespace


class StackAlong(Operator):
    """Stack a list-of-arrays along a new axis via `xp.stack`.

    All input arrays must share the same shape, dtype, and backend.

    Args:
        axis: Position of the new axis in the output, for raw arrays.
            Default 0.
        dim: For named carriers instead: the new dim's name
            (`pipekit_array.stack`; it becomes the leading dim). ``axis``
            and ``dim`` are mutually exclusive.

    Raises:
        ValueError: From ``_apply`` when the input sequence is empty
            (also via `array_namespace` when the inputs mix backends).

    Example::

        out = StackAlong(axis=0)([arr_a, arr_b, arr_c])  # shape (3, H, W)
    """

    def __init__(self, axis: int | None = None, dim: str | None = None) -> None:
        if axis is not None and dim is not None:
            raise ValueError("StackAlong takes axis= (raw arrays) or dim=, not both.")
        self.axis = 0 if axis is None and dim is None else axis
        self.dim = dim

    def _apply(self, xs: Sequence[Any]) -> Any:
        if len(xs) == 0:
            raise ValueError("StackAlong requires at least one input array.")
        if self.dim is not None:
            return dims.stack(list(xs), self.dim)
        xp = array_namespace(*xs)
        return xp.stack(list(xs), axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        if self.dim is not None:
            return {"dim": self.dim}
        return {"axis": self.axis}


class ConcatenateAlong(Operator):
    """Concatenate a list-of-arrays along an existing axis via `xp.concat`.

    `xp.concat` is the Array API standardised name; numpy 2.0+
    exposes it as `concat` alongside the legacy `concatenate` alias.
    All input arrays must share shape (except along `axis`), dtype,
    and backend.

    Args:
        axis: Existing axis to concatenate along, for raw arrays. Default 0.
        dim: For named carriers instead: the dim to join along
            (`pipekit_array.concat`). ``axis`` and ``dim`` are mutually
            exclusive.

    Raises:
        ValueError: From ``_apply`` when the input sequence is empty
            (also via `array_namespace` when the inputs mix backends).

    Example::

        out = ConcatenateAlong(axis=0)([arr_a, arr_b])  # shape (2*N, H, W)
    """

    def __init__(self, axis: int | None = None, dim: str | None = None) -> None:
        if axis is not None and dim is not None:
            raise ValueError(
                "ConcatenateAlong takes axis= (raw arrays) or dim=, not both."
            )
        self.axis = 0 if axis is None and dim is None else axis
        self.dim = dim

    def _apply(self, xs: Sequence[Any]) -> Any:
        if len(xs) == 0:
            raise ValueError("ConcatenateAlong requires at least one input array.")
        if self.dim is not None:
            return dims.concat(list(xs), self.dim)
        xp = array_namespace(*xs)
        return xp.concat(list(xs), axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        if self.dim is not None:
            return {"dim": self.dim}
        return {"axis": self.axis}
