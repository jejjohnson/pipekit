"""The operators of `pipekit_array.named` that select."""

from __future__ import annotations

from typing import Any

from pipekit_ops import Operator

from pipekit_array._recorded import recorded
from pipekit_array.named import _select
from pipekit_array.named._carrier import is_mask
from pipekit_array.operators._shared import _signature


__all__ = ["ISel", "Sel", "TakeAlong"]


def _slice_config(index: Any) -> Any:
    if isinstance(index, slice):
        return {"start": index.start, "stop": index.stop, "step": index.step}
    return index


def _slice_from_config(index: Any) -> Any:
    if isinstance(index, dict) and set(index) == {"start", "stop", "step"}:
        return slice(index["start"], index["stop"], index["step"])
    return index


@recorded("pipekit_array.dim_operators")
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
        return _select.isel(x, **{self.dim: self.index})

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


@recorded("pipekit_array.dim_operators")
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
        return _select.sel(x, **{self.dim: self.label})

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


@recorded("pipekit_array.dim_operators")
class TakeAlong(Operator):
    """Gather one entry of ``dim`` per position: ``op(x, index)``.

    ``index`` is a named carrier with ``x``'s dims without ``dim``, such as
    an `ArgReduce` result.
    """

    def __init__(self, dim: str) -> None:
        self.dim = dim

    def _apply(self, x: Any, index: Any) -> Any:
        return _select.take_along(x, index, self.dim)

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        return sig.drop_dims((self.dim,))
