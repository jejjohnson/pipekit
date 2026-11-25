"""The operators of `pipekit_array.named` that reduce, and `MeanScalar`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pipekit_ops import Operator, Signature

from pipekit_array._recorded import recorded
from pipekit_array.named import _reduce
from pipekit_array.named._dispatch import array_namespace
from pipekit_array.operators._shared import (
    _names,
    _names_config,
    _reduced_dtype,
    _signature,
)


__all__ = ["ArgReduce", "CumReduce", "Dot", "MeanScalar", "Quantile", "Reduce"]


@recorded("pipekit_array.reduce", "pipekit_array.reductions")
class MeanScalar(Operator):
    """Reduce an array to a scalar (or per-axis means) via `xp.mean`.

    Defaults to a full reduction (`axis=None`) yielding a scalar of
    rank 0. Output dtype matches the input's floating dtype per the
    Array API `mean` contract.

    Args:
        axis: Axis or tuple of axes to reduce over. `None` reduces
            all axes.
        keepdims: If True, retains the reduced axes with size 1.

    Raises:
        ValueError: From `compute_output_signature` when ``axis`` is out
            of range for the input signature's rank (``xp.mean`` would
            raise the backend-native equivalent at apply time).

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
        # a tuple of axes reloads from JSON as a list, which `xp.mean` rejects
        self.axis = tuple(axis) if isinstance(axis, list) else axis
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


@recorded("pipekit_array.dim_operators")
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
        return _reduce.reduce(x, self.fn, self.dim)

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


@recorded("pipekit_array.dim_operators")
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
        return _reduce.argreduce(x, self.fn, self.dim)

    def compute_output_signature(self, input_signature: Any) -> Any:
        sig = _signature(input_signature)
        if sig is None:
            return None
        return Signature(dims=sig.drop_dims((self.dim,)).dims, dtype=None)


#: cumulative functions that keep a float input's dtype
_CUMULATIVE_FLOAT_PRESERVING = frozenset(
    {"cumulative_sum", "cumulative_prod", "nancumsum"}
)


@recorded("pipekit_array.dim_operators")
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
        return _reduce.cumreduce(x, self.fn, self.dim)

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


@recorded("pipekit_array.dim_operators")
class Quantile(Operator):
    """Quantiles over named dims; a sequence ``q`` adds a leading ``quantile`` dim.

    ``q`` is stored in config as a float or an ordered list.

    Args:
        q: A probability in [0, 1], or a sequence of them.
        dim: A dim name, or several, to reduce over.
        method: One of `pipekit_array.named.METHODS`.
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
        _reduce.check_method(method, "Quantile")
        self.q = _reduce.probabilities(q, "Quantile")
        self.dim = _names(dim)
        self.method = method
        self.skipna = skipna

    def _apply(self, x: Any) -> Any:
        return _reduce.quantile(
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


@recorded("pipekit_array.dim_operators")
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
        return _reduce.dot(*xs, dim=self.dim)

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
