"""The operators of `pipekit_array.named` that move data along a dim."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pipekit_ops import Operator

from pipekit_array._recorded import recorded
from pipekit_array.named import _shift
from pipekit_array.operators._shared import _signature


__all__ = ["Diff", "Pad", "Roll", "Shift", "Where"]


@recorded("pipekit_array.dim_operators")
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
        return _shift.pad(
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


@recorded("pipekit_array.dim_operators")
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
        return _shift.roll(x, **self.shifts)

    def get_config(self) -> dict[str, Any]:
        return dict(self.shifts)

    def compute_output_signature(self, input_signature: Any) -> Any:
        return _signature(input_signature)


@recorded("pipekit_array.dim_operators")
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
        return _shift.shift(x, fill_value=self.fill_value, **self.shifts)

    def get_config(self) -> dict[str, Any]:
        return {"fill_value": self.fill_value, **self.shifts}

    def compute_output_signature(self, input_signature: Any) -> Any:
        return _signature(input_signature)


@recorded("pipekit_array.dim_operators")
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
        return _shift.diff(x, self.dim, self.n, label=self.label)

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


@recorded("pipekit_array.dim_operators")
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
        return _shift.where(x, cond, self.other)

    def get_config(self) -> dict[str, Any]:
        return {"other": self.other}

    def compute_output_signature(self, input_signature: Any) -> Any:
        if isinstance(input_signature, tuple) and len(input_signature) == 2:
            input_signature = input_signature[0]
        return _signature(input_signature)
