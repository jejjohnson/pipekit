"""The operators that join carriers.

`StackAlong` and `ConcatenateAlong` wrap ``xp.stack`` and ``xp.concat`` for raw
arrays (``axis=``), and `pipekit_array.named.stack` and ``concat`` for named
carriers (``dim=``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from pipekit_ops import Operator, Signature

from pipekit_array._recorded import recorded
from pipekit_array.named import _dispatch, _join
from pipekit_array.named._dispatch import array_namespace
from pipekit_array.operators._shared import _signature


__all__ = ["Combine", "ConcatenateAlong", "StackAlong"]


@recorded("pipekit_array.dim_operators")
class Combine(Operator):
    """An element-wise function of carriers with the same dims: ``op(a, b, …)``.

    Holds a user function, so its config is a debug repr only.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def _apply(self, *xs: Any) -> Any:
        return _join.combine(self.fn, *xs)

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        # same dims; fn may change the dtype
        return Signature(dims=sig.dims, dtype=None)


@recorded("pipekit_array.combinators")
class StackAlong(Operator):
    """Stack a list-of-arrays along a new axis via `xp.stack`.

    All input arrays must share the same shape, dtype, and backend; NumPy
    arrays join JAX ones.

    Args:
        axis: Position of the new axis in the output, for raw arrays.
            Default 0.
        dim: For named carriers instead: the new dim's name
            (`pipekit_array.stack`; it becomes the leading dim). ``axis``
            and ``dim`` are mutually exclusive.

    Raises:
        ValueError: From ``_apply`` when the input sequence is empty
            (also when the inputs mix backends other than NumPy and JAX).

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
            return _join.stack(list(xs), self.dim)
        arrays = _dispatch.join_backends(xs, "StackAlong")
        xp = array_namespace(*arrays)
        return xp.stack(arrays, axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        if self.dim is not None:
            return {"dim": self.dim}
        return {"axis": self.axis}


@recorded("pipekit_array.combinators")
class ConcatenateAlong(Operator):
    """Concatenate a list-of-arrays along an existing axis via `xp.concat`.

    `xp.concat` is the Array API standardised name; numpy 2.0+
    exposes it as `concat` alongside the legacy `concatenate` alias.
    All input arrays must share shape (except along `axis`), dtype,
    and backend; NumPy arrays join JAX ones.

    Args:
        axis: Existing axis to concatenate along, for raw arrays. Default 0.
        dim: For named carriers instead: the dim to join along
            (`pipekit_array.concat`). ``axis`` and ``dim`` are mutually
            exclusive.

    Raises:
        ValueError: From ``_apply`` when the input sequence is empty
            (also when the inputs mix backends other than NumPy and JAX).

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
            return _join.concat(list(xs), self.dim)
        arrays = _dispatch.join_backends(xs, "ConcatenateAlong")
        xp = array_namespace(*arrays)
        return xp.concat(arrays, axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        if self.dim is not None:
            return {"dim": self.dim}
        return {"axis": self.axis}
