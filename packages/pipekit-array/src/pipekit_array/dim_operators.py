"""The dim-level functions as `Operator`s, for pipelines and config.

Each wraps one function of `pipekit_array.dims` and computes its output
`Signature` without running. Operators that take only names and numbers
(`ISel`, `Sel`, `Transpose`, `Reduce`, `ArgReduce`, `TakeAlong`)
round-trip through ``dumps`` / ``loads``; those that hold a user function
(`MapOver`, `Combine`) set ``forbid_in_yaml``, like `pipekit.Lambda`.

`Reduce` and `ArgReduce` take the reduction by its array-namespace name
(``"mean"``, ``"argmax"``) so they serialise; the function forms in
`pipekit_array.dims` also take a callable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from pipekit import Operator, Signature

from pipekit_array import dims as _dims


__all__ = [
    "ArgReduce",
    "Combine",
    "ISel",
    "MapOver",
    "Reduce",
    "Sel",
    "TakeAlong",
    "Transpose",
]

#: reductions that keep the input's dtype whatever it is
_DTYPE_PRESERVING = frozenset(
    {"max", "min", "sum", "prod", "nanmax", "nanmin", "nansum", "nanprod"}
)


def _slice_config(index: Any) -> Any:
    if isinstance(index, slice):
        return {"start": index.start, "stop": index.stop, "step": index.step}
    return index


def _slice_from_config(index: Any) -> Any:
    if isinstance(index, dict) and set(index) == {"start", "stop", "step"}:
        return slice(index["start"], index["stop"], index["step"])
    return index


def _signature(value: Any) -> Signature | None:
    return value if isinstance(value, Signature) else None


class ISel(Operator):
    """Select by position along one dim; an integer drops the dim.

    Args:
        dim: The dim to index.
        index: An int, or a slice (stored in config as start/stop/step).

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> ISel(dim="time", index=0)(DimArray(np.zeros((12, 4)), ("time", "x"))).dims
        ('x',)
    """

    def __init__(self, dim: str, index: int | slice) -> None:
        self.dim = dim
        self.index = _slice_from_config(index)

    def _apply(self, x: Any) -> Any:
        return _dims.isel(x, **{self.dim: self.index})

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim, "index": _slice_config(self.index)}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        if isinstance(self.index, slice):
            size = dict(sig.dims)[self.dim]
            new = None if size is None else len(range(*self.index.indices(size)))
            return sig.replace_dims({self.dim: new})
        return sig.drop_dims((self.dim,))


class Sel(Operator):
    """Select by exact label along one dim; one label drops the dim.

    Args:
        dim: The dim to select on; it must have labels.
        label: One label, or a list of labels (the dim is kept).
    """

    def __init__(self, dim: str, label: Any) -> None:
        self.dim = dim
        self.label = list(label) if isinstance(label, tuple) else label

    def _apply(self, x: Any) -> Any:
        return _dims.sel(x, **{self.dim: self.label})

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim, "label": self.label}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        if isinstance(self.label, list):
            return sig.replace_dims({self.dim: len(self.label)})
        return sig.drop_dims((self.dim,))


class Transpose(Operator):
    """Reorder dims by name.

    Args:
        order: Every dim, once, in the new order.
    """

    def __init__(self, order: Sequence[str]) -> None:
        self.order = tuple(order)

    def _apply(self, x: Any) -> Any:
        return _dims.transpose(x, self.order)

    def get_config(self) -> dict[str, Any]:
        return {"order": list(self.order)}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        sizes = dict(sig.dims)
        return Signature(dims=tuple((d, sizes[d]) for d in self.order), dtype=sig.dtype)


class Reduce(Operator):
    """Reduce over named dims with an array-namespace reduction.

    Args:
        fn: The reduction's name in the array namespace: ``"mean"``, ``"max"``,
            ``"nanmedian"``, …
        dim: A dim name, or several.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.ones((12, 2, 4)), ("time", "band", "x"))
        >>> Reduce("median", dim="time")(cube).dims
        ('band', 'x')
    """

    def __init__(self, fn: str, dim: str | Sequence[str]) -> None:
        if not isinstance(fn, str):
            raise TypeError(
                "Reduce takes the reduction's name (e.g. 'mean') so it serialises; "
                "for a function, use pipekit_array.reduce(x, fn, dim)."
            )
        self.fn = fn
        self.dim = dim if isinstance(dim, str) else tuple(dim)

    def _apply(self, x: Any) -> Any:
        return _dims.reduce(x, self.fn, self.dim)

    def get_config(self) -> dict[str, Any]:
        dim = self.dim if isinstance(self.dim, str) else list(self.dim)
        return {"fn": self.fn, "dim": dim}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        names = (self.dim,) if isinstance(self.dim, str) else self.dim
        dropped = sig.drop_dims(tuple(names))
        keeps_dtype = self.fn in _DTYPE_PRESERVING or "float" in (sig.dtype or "")
        return Signature(dims=dropped.dims, dtype=sig.dtype if keeps_dtype else None)


class ArgReduce(Operator):
    """The index of a reduction along one dim (``"argmax"``, ``"argmin"``).

    The output signature's dtype is ``None``: the integer width depends on
    the backend.
    """

    def __init__(self, fn: str, dim: str) -> None:
        if not isinstance(fn, str):
            raise TypeError("ArgReduce takes the reduction's name, e.g. 'argmax'.")
        self.fn = fn
        self.dim = dim

    def _apply(self, x: Any) -> Any:
        return _dims.argreduce(x, self.fn, self.dim)

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        return Signature(dims=sig.drop_dims((self.dim,)).dims, dtype=None)


class TakeAlong(Operator):
    """Gather one entry of ``dim`` per position: ``op(x, index)``.

    ``index`` is a named carrier with ``x``'s dims without ``dim``, such as
    an `ArgReduce` result.
    """

    def __init__(self, dim: str) -> None:
        self.dim = dim

    def _apply(self, x: Any, index: Any) -> Any:
        return _dims.take_along(x, index, self.dim)

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        return sig.drop_dims((self.dim,))


class MapOver(Operator):
    """``fn`` on each item along ``dim``; the results stacked along ``dim``, first.

    Holds a user function, so its config is a debug repr only.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, fn: Callable[[Any], Any], dim: str) -> None:
        self.fn = fn
        self.dim = dim

    def _apply(self, x: Any) -> Any:
        return _dims.map_over(x, self.fn, self.dim)

    def compute_output_signature(self, input_signature: Any) -> Any:
        # fn is opaque: its output dims are unknown without running it
        return None


class Combine(Operator):
    """An element-wise function of carriers with the same dims: ``op(a, b, …)``.

    Holds a user function, so its config is a debug repr only.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def _apply(self, *xs: Any) -> Any:
        return _dims.combine(self.fn, *xs)

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        # same dims; fn may change the dtype
        return Signature(dims=sig.dims, dtype=None)
