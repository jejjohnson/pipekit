"""The operators that run a function or an operator along a dim.

`MapOver` holds a function, and `ApplyAlong` (A15) and `BatchAlong` (A19) an
operator, so they round-trip whenever that operator does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pipekit_ops import Operator, Signature
from pipekit_ops._base.operator import nested_config, require_operator

from pipekit_array._recorded import recorded
from pipekit_array.named import _map
from pipekit_array.named._batches import PAD_POLICIES, PadPolicy
from pipekit_array.named._dispatch import adapter_of, axes_of
from pipekit_array.named._join import concat
from pipekit_array.named._select import isel
from pipekit_array.named._shape import set_labels
from pipekit_array.operators._shared import _signature


__all__ = ["ApplyAlong", "BatchAlong", "MapOver"]


@recorded("pipekit_array.dim_operators")
class MapOver(Operator):
    """``fn`` on each item along ``dim``; the results stacked along ``dim``, first.

    Holds a user function, so its config is a debug repr only.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, fn: Callable[[Any], Any], dim: str) -> None:
        self.fn = fn
        self.dim = dim

    def _apply(self, x: Any) -> Any:
        return _map.map_over(x, self.fn, self.dim)

    def compute_output_signature(self, input_signature: Any) -> Any:
        # fn is opaque: its output dims are unknown without running it
        return None


@recorded("pipekit_array.dim_operators")
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
        return _map.map_over(x, self.inner, self.dim)

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


@recorded("pipekit_array.batching")
class BatchAlong(Operator):
    """``inner`` on chunks of ``size`` along ``dim`` of one carrier, joined back.

    y = concat([inner(x_{0:m}), inner(x_{m:2m}), …], dim),  m = ``size``.

    Not `pipekit_ops.BatchedMap`, which takes an iterable and calls its
    operator on lists of items: `BatchAlong` splits *one* named carrier
    along a dim (A19, which supersedes A5). The use case is a GPU-bound
    model on a cube that does not fit in device memory at once.

    ``inner`` must keep ``dim``, one output entry per input entry (a
    per-item operator), for the ``"last"`` padding to be trimmed and for
    ``dim``'s labels to come back; the result then equals ``inner(x)``.

    Algorithm:
        1. Cut ``dim`` into chunks of ``size``; the last may be short.
        2. Apply ``pad`` (ADR A10) to a short last chunk: ``"none"`` keeps
           it short; ``"last"`` pads it to ``size`` with copies of its
           last item, so a jitted ``inner`` compiles once, and trims the
           padding from its output; ``"drop"`` drops it.
        3. Join the outputs along ``dim`` and set ``dim``'s labels to the
           input's (an adapter's generic ``concat`` drops them).

    Args:
        inner: The operator applied to each chunk.
        size: Entries per chunk; must be positive.
        dim: The dim to cut.
        pad: ``"none"`` (default), ``"last"`` or ``"drop"``.

    Raises:
        TypeError: ``inner`` is not an `Operator`.
        ValueError: ``size`` is not positive or ``pad`` is not a policy.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray, Reduce
        >>> cube = DimArray(np.ones((10, 2, 4)), ("sample", "band", "x"))
        >>> per_chunk = BatchAlong(Reduce("sum", "band"), size=4, dim="sample")
        >>> per_chunk(cube).dims, per_chunk(cube).shape
        (('sample', 'x'), (10, 4))
        >>> BatchAlong(Reduce("sum", "band"), 4, "sample", pad="drop")(cube).shape
        (8, 4)
    """

    def __init__(
        self, inner: Operator, size: int, dim: str, pad: PadPolicy = "none"
    ) -> None:
        require_operator(inner, "BatchAlong", "inner")
        if size <= 0:
            raise ValueError(f"BatchAlong: size must be positive, got {size}.")
        if pad not in PAD_POLICIES:
            raise ValueError(
                f"BatchAlong: pad must be one of {PAD_POLICIES}, got {pad!r}."
            )
        self.inner = inner
        self.size = size
        self.dim = dim
        self.pad = pad

    def _apply(self, x: Any) -> Any:
        adapter = adapter_of(x, "BatchAlong")
        (axis,) = axes_of(adapter.dims(x), (self.dim,), "BatchAlong")
        length = adapter.data(x).shape[axis]
        outputs = []
        covered = 0
        for start in range(0, length, self.size):
            stop = min(start + self.size, length)
            missing = self.size - (stop - start)
            if missing and self.pad == "drop":
                break
            covered = stop
            if missing and self.pad == "last":
                # copies of the chunk's last item fill it to size
                positions = [*range(start, stop), *([stop - 1] * missing)]
                padded = self.inner(adapter.isel(x, {self.dim: positions}))
                outputs.append(isel(padded, **{self.dim: slice(0, stop - start)}))
                continue
            outputs.append(self.inner(adapter.isel(x, {self.dim: slice(start, stop)})))
        if not outputs:
            raise ValueError(
                f"BatchAlong: pad='drop' drops every entry of {self.dim!r} "
                f"(size {length} < {self.size})."
            )
        out = concat(outputs, self.dim)
        labels = adapter.labels(x, self.dim)
        if labels is None:
            return out
        out_adapter = adapter_of(out, "BatchAlong")
        (out_axis,) = axes_of(out_adapter.dims(out), (self.dim,), "BatchAlong")
        if out_adapter.data(out).shape[out_axis] != covered:
            return out
        # a per-item inner: entry i of the output is entry i of the input
        return set_labels(out_adapter, out, {self.dim: labels[:covered]}, "BatchAlong")

    def get_config(self) -> dict[str, Any]:
        return {
            "inner": nested_config(self.inner),
            "size": self.size,
            "dim": self.dim,
            "pad": self.pad,
        }

    def compute_output_signature(self, input_signature: Any) -> Any:
        if not isinstance(input_signature, Signature):
            return None
        length = dict(input_signature.dims)[self.dim]
        if length is None:
            return None
        chunk = input_signature.replace_dims({self.dim: min(self.size, length)})
        out = self.inner.compute_output_signature(chunk)
        if not isinstance(out, Signature) or self.dim not in dict(out.dims):
            return None
        # a per-item inner: as many entries as the input keeps
        kept = length - length % self.size if self.pad == "drop" else length
        return out.replace_dims({self.dim: kept})
