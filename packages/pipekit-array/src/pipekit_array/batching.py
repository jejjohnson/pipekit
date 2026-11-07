"""Fixed-size batches of named carriers, with a mask of the real entries.

`batch` groups a stream of carriers into `Padded` batches of one size, so a
jitted step compiles once for every batch, the short last one included:

```text
items ──batch(size=4, dim="sample", pad="last")──▶ Padded
  i₀ i₁ i₂ i₃ i₄ i₅                                 value  (sample, *item_dims)
                                                    valid  (sample,) bool
  batch 1: i₀ i₁ i₂ i₃    valid = T T T T
  batch 2: i₄ i₅ i₅ i₅    valid = T T F F   ← copies of the last real item
```

Padding copies the last real item: its values are finite and in
distribution, so a masked loss and its gradient stay finite and a
normalisation over the batch sees real statistics. Reduce with
``reduce(out, "mean", "sample", where=padded.valid)`` to leave the padding
out.

`BatchAlong` is the operator form for one carrier: it cuts a dim into
chunks, runs an operator on each and joins the results, with the same
``pad`` policies (A19).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Literal, NamedTuple

from pipekit_ops import Operator, Signature
from pipekit_ops._base.operator import nested_config, require_operator

from pipekit_array.dim_shapes import set_labels
from pipekit_array.dims import (
    _axes,
    adapter_of,
    concat,
    isel,
    namespace_of,
    stack_items,
)


__all__ = ["PAD_POLICIES", "BatchAlong", "PadPolicy", "Padded", "batch"]

#: what `batch` does with a short last batch
PadPolicy = Literal["last", "none", "drop"]
PAD_POLICIES: tuple[str, ...] = ("last", "none", "drop")


class Padded(NamedTuple):
    """A batch: the items as one carrier and a mask of the real ones.

    A `NamedTuple`, so JAX treats it as a pytree: a jitted step taking a
    `Padded` does not retrace when the mask changes. A full batch is a
    `Padded` whose ``valid`` is all True.

    Attributes:
        value: The items stacked along the batch dim, first
            (`stack_items`): ``(n, *item_dims)``.
        valid: ``(n,)`` bool, True for real items, in the value's array
            namespace.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray, reduce
        >>> items = [DimArray(np.full(3, float(i)), ("x",)) for i in range(3)]
        >>> padded = next(batch(items, size=4, dim="sample"))
        >>> padded.valid.tolist()
        [True, True, True, False]
        >>> reduce(padded.value, "mean", "sample", where=padded.valid).data
        array([1., 1., 1.])
    """

    value: Any
    valid: Any


def batch(
    items: Iterable[Any], *, size: int, dim: str, pad: PadPolicy = "last"
) -> Iterator[Padded]:
    """Group named carriers into `Padded` batches of ``size`` along ``dim``.

    Args:
        items: Named carriers with the same dims (each registered with
            pipekit's adapter registry), consumed lazily.
        size: Items per batch; must be positive.
        dim: The new leading batch dim (``"sample"``, ``"patch"``,
            ``"member"``); must not be a dim of the items.
        pad: What to do with a short last batch:

            - ``"last"`` (default): pad it to ``size`` with copies of its
              last real item, ``valid`` False for the copies, so every
              batch has one shape;
            - ``"none"``: yield it short, all valid;
            - ``"drop"``: drop it (``drop_remainder`` in tf.data / Grain).

    Yields:
        One `Padded` per ``size`` items, plus the last short batch per
        ``pad``. Each ``value`` is `stack_items` of its items.

    Raises:
        ValueError: ``size`` is not positive, ``pad`` is not a policy, or
            the items' dims differ (from `stack_items`).

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> items = [DimArray(np.ones((2, 2)), ("y", "x")) for _ in range(5)]
        >>> def sizes(pad):
        ...     return [len(b.valid) for b in batch(items, size=2, dim="s", pad=pad)]
        >>> sizes("last"), sizes("none"), sizes("drop")
        ([2, 2, 2], [2, 2, 1], [2, 2])
    """
    if size <= 0:
        raise ValueError(f"batch: size must be positive, got {size}.")
    if pad not in PAD_POLICIES:
        raise ValueError(f"batch: pad must be one of {PAD_POLICIES}, got {pad!r}.")
    group: list[Any] = []
    for item in items:
        group.append(item)
        if len(group) == size:
            yield _padded(group, dim, 0)
            group = []
    if group and pad != "drop":
        yield _padded(group, dim, size - len(group) if pad == "last" else 0)


def _padded(group: list[Any], dim: str, n_pad: int) -> Padded:
    value = stack_items([*group, *([group[-1]] * n_pad)], dim)
    xp = namespace_of(adapter_of(value, "batch").data(value))
    valid = xp.asarray([True] * len(group) + [False] * n_pad, dtype=xp.bool)
    return Padded(value, valid)


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
        (axis,) = _axes(adapter.dims(x), (self.dim,), "BatchAlong")
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
        (out_axis,) = _axes(out_adapter.dims(out), (self.dim,), "BatchAlong")
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
