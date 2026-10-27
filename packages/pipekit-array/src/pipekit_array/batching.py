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
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Literal, NamedTuple

from pipekit_array.dims import adapter_of, namespace_of, stack_items


__all__ = ["PAD_POLICIES", "PadPolicy", "Padded", "batch"]

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
