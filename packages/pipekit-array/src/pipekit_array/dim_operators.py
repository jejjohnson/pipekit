"""The dim-level functions as `Operator`s, for pipelines and config.

Each wraps one function of `pipekit_array.dims`,
`pipekit_array.dim_shapes` or `pipekit_array.dim_shifts` and computes its
output `Signature` without running. Operators that take only names and
numbers (`ISel`, `Sel`, `Transpose`, `Reduce`, `ArgReduce`, `TakeAlong`,
`Rename`, `ExpandDims`, `Squeeze`, `BroadcastLike`, `MergeDims`,
`SplitDims`, `Pad`, `Roll`, `Shift`, `Diff`, `CumReduce`, `Where`)
round-trip through ``dumps`` / ``loads``; those that hold a user function
(`MapOver`, `Combine`) set ``forbid_in_yaml``, like `pipekit_ops.Lambda`.

`Reduce` and `ArgReduce` take the reduction by its array-namespace name
(``"mean"``, ``"argmax"``) so they serialise; the function forms in
`pipekit_array.dims` also take a callable.

Phase 3 (`pipekit_array.dim_windows`): `Window` and `Coarsen` (by the
reduction's name) round-trip; `ApplyAlong` holds an `Operator`, not a
function, so it round-trips whenever that operator does.

Phase 4 (`pipekit_array.dim_algebra`): `Quantile` and `Dot` round-trip;
``q`` is stored as a float or an ordered list.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Self

from pipekit_ops import Operator, Signature
from pipekit_ops._base.operator import nested_config, require_operator

from pipekit_array import (
    dim_algebra as _dim_algebra,
    dim_shapes as _dim_shapes,
    dim_shifts as _dim_shifts,
    dim_windows as _dim_windows,
    dims as _dims,
)
from pipekit_array.dimarray import is_mask


__all__ = [
    "ApplyAlong",
    "ArgReduce",
    "BroadcastLike",
    "Coarsen",
    "Combine",
    "CumReduce",
    "Diff",
    "Dot",
    "ExpandDims",
    "ISel",
    "MapOver",
    "MergeDims",
    "Pad",
    "Quantile",
    "Reduce",
    "Rename",
    "Roll",
    "Sel",
    "Shift",
    "SplitDims",
    "Squeeze",
    "TakeAlong",
    "Transpose",
    "Where",
    "Window",
]

#: reductions that keep the input's dtype whatever it is
#: reductions whose output dtype is always the input's
_DTYPE_PRESERVING = frozenset({"max", "min", "nanmax", "nanmin"})
#: reductions that keep a float input's dtype (integers and bools promote)
_FLOAT_PRESERVING = frozenset(
    {
        "sum",
        "prod",
        "nansum",
        "nanprod",
        "mean",
        "nanmean",
        "median",
        "nanmedian",
        "std",
        "nanstd",
        "var",
        "nanvar",
    }
)
#: reductions that return booleans
_BOOLEAN = frozenset({"all", "any"})


def _reduced_dtype(fn: str, dtype: str | None) -> str | None:
    """The dtype a reduction returns, when it is known without running it."""
    if dtype is None:
        return None
    if fn in _DTYPE_PRESERVING:
        return dtype
    if fn in _BOOLEAN:
        return "bool"
    if fn in _FLOAT_PRESERVING and "float" in dtype:
        return dtype
    return None


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
        index = _slice_from_config(index)
        if type(index).__module__ == "numpy":
            # NumPy positions or a mask (array or scalar) as int / list, so
            # the config is JSON; numpy is optional, so no isinstance
            index = index.tolist()
        self.index = index

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
        if getattr(self.index, "ndim", None) == 0 or isinstance(self.index, int):
            return sig.drop_dims((self.dim,))
        # a list or 1-D array of positions keeps the dim, one entry each; a
        # boolean mask keeps one entry per True
        return sig.replace_dims({self.dim: _selected_count(self.index)})


def _selected_count(index: Any) -> int | None:
    """Entries a 1-D position array or boolean mask selects; None if traced."""
    if not is_mask(index):
        return len(index)
    if "Tracer" in type(index).__name__:
        return None
    import numpy as np

    return int(np.count_nonzero(np.asarray(index)))


_TUPLE = "__tuple__"


def _encode_tuples(value: Any) -> Any:
    """Tuples (at any depth in lists) as ``{"__tuple__": [...]}`` for JSON."""
    if isinstance(value, tuple):
        return {_TUPLE: [_encode_tuples(v) for v in value]}
    if isinstance(value, list):
        return [_encode_tuples(v) for v in value]
    return value


def _decode_tuples(value: Any) -> Any:
    """The inverse of `_encode_tuples`; other values pass through."""
    if isinstance(value, dict) and set(value) == {_TUPLE}:
        return tuple(_decode_tuples(v) for v in value[_TUPLE])
    if isinstance(value, list):
        return [_decode_tuples(v) for v in value]
    return value


class Sel(Operator):
    """Select by exact label along one dim; one label drops the dim.

    Args:
        dim: The dim to select on; it must have labels.
        label: One label, or a list of labels (the dim is kept). A tuple is
            one label when the dim has tuple-valued labels, else several.
    """

    def __init__(self, dim: str, label: Any) -> None:
        self.dim = dim
        # a tuple that is itself a label (tuple-valued labels) stays one label;
        # a config from JSON spells it {"__tuple__": [...]}
        label = _decode_tuples(label)
        if getattr(label, "ndim", 0) == 1:
            # a 1-D array of labels is a list of labels: the dim is kept, and
            # the config stays JSON
            label = list(label.tolist())
        self.label = label

    def _apply(self, x: Any) -> Any:
        return _dims.sel(x, **{self.dim: self.label})

    def get_config(self) -> dict[str, Any]:
        # JSON has no tuples: keep "one tuple label" apart from "a list of labels"
        return {"dim": self.dim, "label": _encode_tuples(self.label)}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        if isinstance(self.label, tuple):
            return None  # one tuple-valued label or several: unknown without labels
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
        return Signature(dims=dropped.dims, dtype=_reduced_dtype(self.fn, sig.dtype))


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


# Phase 1: dims


def _names(dim: str | Sequence[str]) -> str | tuple[str, ...]:
    """One dim name as is, several as a tuple."""
    return dim if isinstance(dim, str) else tuple(dim)


def _names_config(dim: str | tuple[str, ...]) -> str | list[str]:
    return dim if isinstance(dim, str) else list(dim)


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
        return _dim_shapes.rename(x, **self.names)

    def get_config(self) -> dict[str, Any]:
        return dict(self.names)

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        dims = tuple((self.names.get(d, d), n) for d, n in sig.dims)
        return Signature(dims=dims, dtype=sig.dtype)


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
        return _dim_shapes.expand_dims(x, self.dim, axis=self.axis)

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


class Squeeze(Operator):
    """Drop named length-1 dims.

    Args:
        dim: A dim name, or several; each must have size 1.
    """

    def __init__(self, dim: str | Sequence[str]) -> None:
        self.dim = _names(dim)

    def _apply(self, x: Any) -> Any:
        return _dim_shapes.squeeze(x, self.dim)

    def get_config(self) -> dict[str, Any]:
        return {"dim": _names_config(self.dim)}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        names = (self.dim,) if isinstance(self.dim, str) else self.dim
        return sig.drop_dims(names)


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
        return _dim_shapes.broadcast_like(x, other, exclude=self.exclude)

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
        return _dim_shapes.merge_dims(x, **self.dims)

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
        return _dim_shapes.split_dims(x, **self.dims)

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


# Phase 2: index-only transforms

#: cumulative functions that keep a float input's dtype
_CUMULATIVE_FLOAT_PRESERVING = frozenset(
    {"cumulative_sum", "cumulative_prod", "nancumsum"}
)


class Pad(Operator):
    """Widen named dims at either end; a padded dim loses its labels.

    Args:
        mode: ``"constant"``, ``"edge"``, ``"reflect"`` or ``"wrap"``.
        constant_values: The fill for ``mode="constant"``; None is NaN.
        **pad_width: ``dim=`` a width for both ends, or ``(before, after)``.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> Pad(time=(2, 0))(DimArray(np.zeros((12, 4)), ("time", "x"))).shape
        (14, 4)
    """

    def __init__(
        self,
        mode: str = "constant",
        constant_values: float | None = None,
        **pad_width: int | Sequence[int],
    ) -> None:
        self.mode = mode
        self.constant_values = constant_values
        # an int or a pair, as (before, after); JSON gives the pair as a list
        self.pad_width = {
            d: (w, w) if isinstance(w, int) else (w[0], w[1])
            for d, w in pad_width.items()
        }

    def _apply(self, x: Any) -> Any:
        return _dim_shifts.pad(
            x, mode=self.mode, constant_values=self.constant_values, **self.pad_width
        )

    def get_config(self) -> dict[str, Any]:
        widths = {d: list(w) for d, w in self.pad_width.items()}
        return {"mode": self.mode, "constant_values": self.constant_values, **widths}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        sizes = dict(sig.dims)
        grown: dict[str, int | None] = {}
        for d, (before, after) in self.pad_width.items():
            size = sizes[d]
            grown[d] = None if size is None else size + before + after
        return sig.replace_dims(grown)


class Roll(Operator):
    """Shift data cyclically along named dims; labels stay.

    Args:
        **shifts: ``dim=k``; a positive k moves data to higher positions.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> Roll(time=1)(DimArray(np.arange(3.0), ("time",))).data
        array([2., 0., 1.])
    """

    def __init__(self, **shifts: int) -> None:
        self.shifts = dict(shifts)

    def _apply(self, x: Any) -> Any:
        return _dim_shifts.roll(x, **self.shifts)

    def get_config(self) -> dict[str, Any]:
        return dict(self.shifts)

    def compute_output_signature(self, input_signature: Any) -> Any:
        return _signature(input_signature)


class Shift(Operator):
    """Shift data along named dims, filling the vacated entries; labels stay.

    Args:
        fill_value: The value of the vacated entries; None is NaN.
        **shifts: ``dim=k``; a positive k moves data to higher positions.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> Shift(time=1)(DimArray(np.arange(3.0), ("time",))).data
        array([nan,  0.,  1.])
    """

    def __init__(self, fill_value: float | None = None, **shifts: int) -> None:
        self.fill_value = fill_value
        self.shifts = dict(shifts)

    def _apply(self, x: Any) -> Any:
        return _dim_shifts.shift(x, fill_value=self.fill_value, **self.shifts)

    def get_config(self) -> dict[str, Any]:
        return {"fill_value": self.fill_value, **self.shifts}

    def compute_output_signature(self, input_signature: Any) -> Any:
        return _signature(input_signature)


class Diff(Operator):
    """The n-th forward difference along ``dim``: N entries become N - n.

    Args:
        dim: The dim to difference along.
        n: The order, n ≥ 0.
        label: The labels kept on ``dim``: ``"upper"`` (the minuend's) or
            ``"lower"`` (the subtrahend's).

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> Diff("time")(DimArray(np.zeros((12, 4)), ("time", "x"))).shape
        (11, 4)
    """

    def __init__(self, dim: str, n: int = 1, label: str = "upper") -> None:
        self.dim = dim
        self.n = n
        self.label = label

    def _apply(self, x: Any) -> Any:
        return _dim_shifts.diff(x, self.dim, self.n, label=self.label)

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim, "n": self.n, "label": self.label}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        size = dict(sig.dims)[self.dim]
        return sig.replace_dims(
            {self.dim: None if size is None else max(size - self.n, 0)}
        )


class CumReduce(Operator):
    """A cumulative function along ``dim``, by its array-namespace name.

    Args:
        fn: ``"cumulative_sum"``, ``"cumulative_prod"``, ``"nancumsum"``, …
        dim: The dim to accumulate along.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> CumReduce("cumulative_sum", "time")(DimArray(np.ones(3), ("time",))).data
        array([1., 2., 3.])
    """

    def __init__(self, fn: str, dim: str) -> None:
        if not isinstance(fn, str):
            raise TypeError(
                "CumReduce takes the function's name (e.g. 'cumulative_sum') so it "
                "serialises; for a function, use pipekit_array.cumreduce(x, fn, dim)."
            )
        self.fn = fn
        self.dim = dim

    def _apply(self, x: Any) -> Any:
        return _dim_shifts.cumreduce(x, self.fn, self.dim)

    def get_config(self) -> dict[str, Any]:
        return {"fn": self.fn, "dim": self.dim}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        keeps = (
            self.fn in _CUMULATIVE_FLOAT_PRESERVING
            and sig.dtype is not None
            and "float" in sig.dtype
        )
        return Signature(dims=sig.dims, dtype=sig.dtype if keeps else None)


class Where(Operator):
    """Keep ``x`` where ``cond`` is True, ``other`` elsewhere: ``op(x, cond)``.

    `compute_output_signature` takes ``x``'s signature, or the pair
    ``(x's, cond's)``: the output has ``x``'s dims and dtype either way.

    Args:
        other: The fill where ``cond`` is False; None is NaN.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> valid = DimArray(np.array([True, False]), ("x",))
        >>> Where()(DimArray(np.ones(2), ("x",)), valid).data
        array([ 1., nan])
    """

    def __init__(self, other: float | None = None) -> None:
        self.other = other

    def _apply(self, x: Any, cond: Any) -> Any:
        return _dim_shifts.where(x, cond, self.other)

    def get_config(self) -> dict[str, Any]:
        return {"other": self.other}

    def compute_output_signature(self, input_signature: Any) -> Any:
        if isinstance(input_signature, tuple) and len(input_signature) == 2:
            input_signature = input_signature[0]
        return _signature(input_signature)


# Phase 3: windows


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
        return _dim_windows.window(
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
        return _dim_windows.coarsen(x, self.fn, boundary=self.boundary, **self.windows)

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


class ApplyAlong(Operator):
    """``inner`` on each item along ``dim``; the results stacked along ``dim``, first.

    `map_over` with an `Operator` instead of a function, so it serialises
    whenever ``inner`` does: ``inner`` is stored as a nested config, and
    `ApplyAlong` itself does not set ``forbid_in_yaml`` (a flagged
    ``inner`` is found by the serialiser's walk). Use `MapOver` for a
    plain function. Replaces the planned ``ApplyToBands`` (A15).

    Args:
        inner: The operator applied to each item, ``isel(x, dim=i)``; it
            returns named carriers with the same dims.
        dim: The dim to map over.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.ones((3, 2, 4)), ("time", "band", "x"))
        >>> ApplyAlong(Reduce("sum", dim="band"), dim="time")(cube).dims
        ('time', 'x')
    """

    def __init__(self, inner: Operator, dim: str) -> None:
        require_operator(inner, "ApplyAlong", "inner")
        self.inner = inner
        self.dim = dim

    def _apply(self, x: Any) -> Any:
        return _dims.map_over(x, self.inner, self.dim)

    def get_config(self) -> dict[str, Any]:
        return {"inner": nested_config(self.inner), "dim": self.dim}

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        item = self.inner.compute_output_signature(sig.drop_dims((self.dim,)))
        if not isinstance(item, Signature):
            return None
        return Signature(
            dims=((self.dim, dict(sig.dims)[self.dim]), *item.dims), dtype=item.dtype
        )


# Phase 4: statistics and algebra


class Quantile(Operator):
    """Quantiles over named dims; a sequence ``q`` adds a leading ``quantile`` dim.

    ``q`` is stored in config as a float or an ordered list.

    Args:
        q: A probability in [0, 1], or a sequence of them.
        dim: A dim name, or several, to reduce over.
        method: One of `pipekit_array.dim_algebra.METHODS`.
        skipna: Ignore NaN.

    Raises:
        ValueError: ``method`` is not supported, or a probability is
            outside [0, 1].

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.ones((12, 2, 4)), ("time", "band", "x"))
        >>> Quantile([0.1, 0.9], dim="time")(cube).dims
        ('quantile', 'band', 'x')
    """

    def __init__(
        self,
        q: float | Sequence[float],
        dim: str | Sequence[str],
        method: str = "linear",
        skipna: bool = False,
    ) -> None:
        _dim_algebra.check_method(method, "Quantile")
        self.q = _dim_algebra.probabilities(q, "Quantile")
        self.dim = _names(dim)
        self.method = method
        self.skipna = skipna

    def _apply(self, x: Any) -> Any:
        return _dim_algebra.quantile(
            x, self.q, self.dim, method=self.method, skipna=self.skipna
        )

    def get_config(self) -> dict[str, Any]:
        q = list(self.q) if isinstance(self.q, tuple) else self.q
        return {
            "q": q,
            "dim": _names_config(self.dim),
            "method": self.method,
            "skipna": self.skipna,
        }

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        names = (self.dim,) if isinstance(self.dim, str) else self.dim
        kept = sig.drop_dims(names).dims
        # a float input keeps its dtype; an integer one promotes by backend
        dtype = sig.dtype if sig.dtype is not None and "float" in sig.dtype else None
        if isinstance(self.q, tuple):
            dims = (("quantile", len(self.q)), *kept)
            return Signature(dims=dims, dtype=dtype)
        return Signature(dims=kept, dtype=dtype)


class Dot(Operator):
    """The sum of the product of carriers over named dims: ``op(*xs)``.

    `compute_output_signature` takes the tuple of the inputs' signatures:
    from one signature the other inputs' dims are unknown, so it returns
    None.

    Args:
        dim: A dim name, or several, to sum over.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> x = DimArray(np.ones((12, 13)), ("time", "band"))
        >>> w = DimArray(np.ones((13, 3)), ("band", "component"))
        >>> Dot(dim="band")(x, w).dims
        ('time', 'component')
    """

    def __init__(self, dim: str | Sequence[str]) -> None:
        self.dim = _names(dim)

    def _apply(self, *xs: Any) -> Any:
        return _dim_algebra.dot(*xs, dim=self.dim)

    def get_config(self) -> dict[str, Any]:
        return {"dim": _names_config(self.dim)}

    def compute_output_signature(self, input_signature: Any) -> Any:
        if not isinstance(input_signature, tuple) or not input_signature:
            return None
        sigs = [s for s in map(_signature, input_signature) if s is not None]
        if len(sigs) != len(input_signature):
            return None
        names = (self.dim,) if isinstance(self.dim, str) else self.dim
        sizes: dict[str, int | None] = {}
        for sig in sigs:
            for d, n in sig.dims:
                # the first known size; check_compatible makes them agree
                if sizes.get(d) is None:
                    sizes[d] = n
        dims = tuple((d, n) for d, n in sizes.items() if d not in names)
        dtypes = {sig.dtype for sig in sigs}
        return Signature(dims=dims, dtype=dtypes.pop() if len(dtypes) == 1 else None)
