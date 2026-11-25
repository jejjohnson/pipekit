"""The operators of `pipekit_array.named` that change the dims."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Self

from pipekit_ops import Operator, Signature

from pipekit_array._recorded import recorded
from pipekit_array.named import _shape
from pipekit_array.operators._shared import _names, _names_config, _signature


__all__ = [
    "BroadcastLike",
    "ExpandDims",
    "MergeDims",
    "Rename",
    "SplitDims",
    "Squeeze",
    "Transpose",
]


@recorded("pipekit_array.dim_operators")
class Transpose(Operator):
    """Reorder dims by name.

    Args:
        order: Every dim, once, in the new order.
    """

    def __init__(self, order: Sequence[str]) -> None:
        self.order = tuple(order)

    def _apply(self, x: Any) -> Any:
        return _shape.transpose(x, self.order)

    def get_config(self) -> dict[str, Any]:
        return {"order": list(self.order)}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        sizes = dict(sig.dims)
        return Signature(dims=tuple((d, sizes[d]) for d in self.order), dtype=sig.dtype)


@recorded("pipekit_array.dim_operators")
class Rename(Operator):
    """New dim names; data, order and labels unchanged.

    Args:
        **names: ``old=new`` for each dim to rename.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> Rename(band="channel")(DimArray(np.zeros((2, 4)), ("band", "x"))).dims
        ('channel', 'x')
    """

    def __init__(self, **names: str) -> None:
        self.names = dict(names)

    def _apply(self, x: Any) -> Any:
        return _shape.rename(x, **self.names)

    def get_config(self) -> dict[str, Any]:
        return dict(self.names)

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        dims = tuple((self.names.get(d, d), n) for d, n in sig.dims)
        return Signature(dims=dims, dtype=sig.dtype)


@recorded("pipekit_array.dim_operators")
class ExpandDims(Operator):
    """Insert length-1 dims at position ``axis`` of the result.

    Args:
        dim: A new dim name, or several.
        axis: Where the first new dim goes; negative counts from the end.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> ExpandDims("member")(DimArray(np.zeros((2, 4)), ("band", "x"))).dims
        ('member', 'band', 'x')
    """

    def __init__(self, dim: str | Sequence[str], axis: int = 0) -> None:
        self.dim = _names(dim)
        self.axis = axis

    def _apply(self, x: Any) -> Any:
        return _shape.expand_dims(x, self.dim, axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        return {"dim": _names_config(self.dim), "axis": self.axis}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        names = (self.dim,) if isinstance(self.dim, str) else self.dim
        start = self.axis if self.axis >= 0 else self.axis + len(sig.dims) + 1
        added = tuple((n, 1) for n in names)
        dims = (*sig.dims[:start], *added, *sig.dims[start:])
        return Signature(dims=dims, dtype=sig.dtype)


@recorded("pipekit_array.dim_operators")
class Squeeze(Operator):
    """Drop named length-1 dims.

    Args:
        dim: A dim name, or several; each must have size 1.
    """

    def __init__(self, dim: str | Sequence[str]) -> None:
        self.dim = _names(dim)

    def _apply(self, x: Any) -> Any:
        return _shape.squeeze(x, self.dim)

    def get_config(self) -> dict[str, Any]:
        return {"dim": _names_config(self.dim)}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        names = (self.dim,) if isinstance(self.dim, str) else self.dim
        return sig.drop_dims(names)


@recorded("pipekit_array.dim_operators")
class BroadcastLike(Operator):
    """Add ``other``'s missing dims to ``x``, in front: ``op(x, other)``.

    `compute_output_signature` takes the pair ``(x's, other's)``: from
    ``x``'s alone the added dims are unknown, so it returns None.

    Args:
        exclude: Dims of ``other`` neither added nor checked.
    """

    def __init__(self, exclude: Sequence[str] = ()) -> None:
        self.exclude = tuple(exclude)

    def _apply(self, x: Any, other: Any) -> Any:
        return _shape.broadcast_like(x, other, exclude=self.exclude)

    def get_config(self) -> dict[str, Any]:
        return {"exclude": list(self.exclude)}

    def compute_output_signature(self, input_signature: Any) -> Any:
        if not isinstance(input_signature, tuple) or len(input_signature) != 2:
            return None
        sig, other = (_signature(s) for s in input_signature)
        if sig is None or other is None:
            return None
        names = {d for d, _ in sig.dims}
        added = tuple(
            (d, n) for d, n in other.dims if d not in names and d not in self.exclude
        )
        return Signature(dims=(*added, *sig.dims), dtype=sig.dtype)


@recorded("pipekit_array.dim_operators")
class MergeDims(Operator):
    """Merge the listed dims into one, placed last, in C order.

    Its config lists the merges in order (JSON objects sort their keys).

    Args:
        **dims: ``new=(source, …)`` for each merged dim.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.zeros((2, 3, 4)), ("band", "y", "x"))
        >>> MergeDims(pixel=("y", "x"))(cube).shape
        (2, 12)
    """

    def __init__(self, **dims: Sequence[str]) -> None:
        self.dims = {name: _merge_group(name, group) for name, group in dims.items()}

    def _apply(self, x: Any) -> Any:
        return _shape.merge_dims(x, **self.dims)

    def get_config(self) -> dict[str, Any]:
        return {"dims": [[name, list(group)] for name, group in self.dims.items()]}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls(**{name: tuple(group) for name, group in config["dims"]})

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        sizes = dict(sig.dims)
        sources = [s for group in self.dims.values() for s in group]
        kept = sig.drop_dims(tuple(sources)).dims
        merged = tuple(
            (name, _product([sizes[s] for s in group]))
            for name, group in self.dims.items()
        )
        return Signature(dims=(*kept, *merged), dtype=sig.dtype)


def _merge_group(name: str, group: Sequence[str]) -> tuple[str, ...]:
    if isinstance(group, str):
        raise TypeError(f"MergeDims: {name}= takes a sequence of dim names.")
    return tuple(group)


def _product(sizes: Sequence[int | None]) -> int | None:
    """The product of sizes; None if any is unknown."""
    if any(n is None for n in sizes):
        return None
    return math.prod(n for n in sizes if n is not None)


@recorded("pipekit_array.dim_operators")
class SplitDims(Operator):
    """Split one dim into several, in place, in C order.

    Its config lists each split's parts in order (JSON objects sort their
    keys).

    Args:
        **dims: ``dim={part: size, …}`` for each dim to split.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> pixels = DimArray(np.zeros((2, 12)), ("band", "pixel"))
        >>> SplitDims(pixel={"y": 3, "x": 4})(pixels).dims
        ('band', 'y', 'x')
    """

    def __init__(self, **dims: Mapping[str, int]) -> None:
        self.dims = {name: dict(part) for name, part in dims.items()}

    def _apply(self, x: Any) -> Any:
        return _shape.split_dims(x, **self.dims)

    def get_config(self) -> dict[str, Any]:
        return {
            "dims": [
                [name, [[p, n] for p, n in part.items()]]
                for name, part in self.dims.items()
            ]
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls(**{name: dict(part) for name, part in config["dims"]})

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        dims: list[tuple[str, int | None]] = []
        for d, n in sig.dims:
            dims.extend(self.dims[d].items() if d in self.dims else [(d, n)])
        return Signature(dims=tuple(dims), dtype=sig.dtype)
