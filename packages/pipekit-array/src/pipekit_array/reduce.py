"""Reductions — `MeanScalar`.

Wraps `xp.mean` as a pipekit `Operator`. See
`docs/design/api/operators.md` for the spec.
"""

from __future__ import annotations

from typing import Any

from pipekit import Operator
from pipekit.signature import Signature

from pipekit_array._namespace import array_namespace


class MeanScalar(Operator):
    """Reduce an array to a scalar (or per-axis means) via `xp.mean`.

    Defaults to a full reduction (`axis=None`) yielding a scalar of
    rank 0. Output dtype matches the input's floating dtype per the
    Array API `mean` contract.

    Args:
        axis: Axis or tuple of axes to reduce over. `None` reduces
            all axes.
        keepdims: If True, retains the reduced axes with size 1.

    Example::

        out = MeanScalar()(image)               # scalar
        out = MeanScalar(axis=(-2, -1))(image)  # per-band mean
        out = MeanScalar(axis=0, keepdims=True)(image)  # shape (1, H, W)
    """

    def __init__(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def _apply(self, x: Any) -> Any:
        xp = array_namespace(x)
        return xp.mean(x, axis=self.axis, keepdims=self.keepdims)

    def get_config(self) -> dict[str, Any]:
        return {"axis": self.axis, "keepdims": self.keepdims}

    def compute_output_signature(self, input_signature: Any) -> Any:
        if input_signature is None:
            return None

        input_sig: Signature = input_signature
        n_dims = len(input_sig.dims)

        def _validate(axis: int) -> int:
            """Return positive axis index; raise on out-of-range."""
            if n_dims == 0 or axis < -n_dims or axis >= n_dims:
                raise ValueError(
                    f"MeanScalar.axis={axis!r} is out of range for input "
                    f"signature with {n_dims} dim(s); xp.mean would raise."
                )
            return axis % n_dims

        if self.axis is None:
            axes_to_reduce: tuple[int, ...] = tuple(range(n_dims))
        elif isinstance(self.axis, int):
            axes_to_reduce = (_validate(self.axis),)
        else:
            axes_to_reduce = tuple(_validate(a) for a in self.axis)

        if self.keepdims:
            new_dims = tuple(
                (name, 1) if i in axes_to_reduce else (name, size)
                for i, (name, size) in enumerate(input_sig.dims)
            )
        else:
            new_dims = tuple(
                (name, size)
                for i, (name, size) in enumerate(input_sig.dims)
                if i not in axes_to_reduce
            )
        return Signature(dims=new_dims, dtype=input_sig.dtype)
