"""The operators of `pipekit_array.named` that cut a dim into windows."""

from __future__ import annotations

from typing import Any

from pipekit_ops import Operator, Signature

from pipekit_array._recorded import recorded
from pipekit_array.named import _window
from pipekit_array.operators._shared import _reduced_dtype, _signature


__all__ = ["Coarsen", "Window"]


@recorded("pipekit_array.dim_operators")
class Window(Operator):
    """Sliding windows along one dim, as a new last dim; full windows only.

    Args:
        window_dim: The new last dim.
        stride: The step between window starts.
        **sizes: ``dim=S`` for the one dim to window.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> Window(stride=2, time=3)(DimArray(np.zeros((12, 4)), ("time", "x"))).shape
        (5, 4, 3)
    """

    def __init__(
        self, window_dim: str = "window", stride: int = 1, **sizes: int
    ) -> None:
        if len(sizes) != 1:
            raise ValueError(f"Window: give one dim=size, got {sizes}.")
        self.window_dim = window_dim
        self.stride = stride
        self.sizes = dict(sizes)

    def _apply(self, x: Any) -> Any:
        return _window.window(
            x, window_dim=self.window_dim, stride=self.stride, **self.sizes
        )

    def get_config(self) -> dict[str, Any]:
        return {"window_dim": self.window_dim, "stride": self.stride, **self.sizes}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        ((dim, size),) = self.sizes.items()
        length = dict(sig.dims)[dim]
        # n = ⌊(N - S)/s⌋ + 1
        count = None if length is None else (length - size) // self.stride + 1
        windowed = sig.replace_dims({dim: count})
        return Signature(
            dims=(*windowed.dims, (self.window_dim, size)), dtype=sig.dtype
        )


@recorded("pipekit_array.dim_operators")
class Coarsen(Operator):
    """Reduce non-overlapping blocks along named dims by a named reduction.

    Args:
        fn: The reduction's name in the array namespace: ``"mean"``,
            ``"nanmean"``, ``"max"``, …
        boundary: ``"exact"`` or ``"trim"``.
        **windows: ``dim=k`` for each dim to coarsen.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> Coarsen("mean", y=2, x=2)(DimArray(np.ones((4, 6)), ("y", "x"))).shape
        (2, 3)
    """

    def __init__(self, fn: str, boundary: str = "exact", **windows: int) -> None:
        if not isinstance(fn, str):
            raise TypeError(
                "Coarsen takes the reduction's name (e.g. 'mean') so it serialises; "
                "for a function, use pipekit_array.coarsen(x, fn, ...)."
            )
        self.fn = fn
        self.boundary = boundary
        self.windows = dict(windows)

    def _apply(self, x: Any) -> Any:
        return _window.coarsen(x, self.fn, boundary=self.boundary, **self.windows)

    def get_config(self) -> dict[str, Any]:
        return {"fn": self.fn, "boundary": self.boundary, **self.windows}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        sizes = dict(sig.dims)
        coarse: dict[str, int | None] = {}
        for d, k in self.windows.items():
            length = sizes[d]
            # ⌊N/k⌋ blocks; "exact" raises at run time unless k divides N
            coarse[d] = None if length is None else length // k
        dims = sig.replace_dims(coarse).dims
        return Signature(dims=dims, dtype=_reduced_dtype(self.fn, sig.dtype))
