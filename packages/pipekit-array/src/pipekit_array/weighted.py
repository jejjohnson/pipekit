"""Masked, weighted sums and means, of an array and of static groups of samples.

Plain functions on the array backend of `pipekit_array.functions`: one
implementation in functions of the array API standard, which has no ``nansum``,
no ``bincount`` and no ``segment_sum``. Which samples count is said by ``valid``,
boolean, True = use, and never by a NaN.

This module needs ``pipekit-array[numpy,compat]``, so importing
``pipekit_array`` does not import it.
"""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any

import numpy as np

from pipekit_array.functions import (
    boolean_like,
    divide_or_nan,
    floating_inputs,
    table_like,
)


__all__ = [
    "Groups",
    "reduction_inputs",
    "used_weights",
    "weighted_mean",
    "weighted_sum",
]

type Axis = int | tuple[int, ...] | None


def weighted_sum(
    values: Any,
    *,
    weights: Any = None,
    valid: Any = None,
    axis: Axis = None,
) -> Any:
    """The sum of w x over the valid samples.

    S = ∑ₙ mₙ wₙ xₙ, with m the validity, 1 or 0.

    Args:
        values: x, samples of any shape.
        weights: w, broadcast against ``values``; 1 if not given.
        valid: Which samples to use (True = use), broadcast against ``values``.
            A NaN in a valid sample gives NaN, and one in an invalid sample
            reaches neither the sum nor its gradient.
        axis: The axis or axes to sum over; every axis if not given.

    Returns:
        S, the array without the reduced axes. An empty sum is 0.

    Raises:
        BackendError: If an argument is a labelled array, or of another library.
        ValueError: If ``valid`` is not boolean.

    Example:
        >>> import numpy as np
        >>> x = np.asarray([1.0, np.nan, 3.0])
        >>> float(weighted_sum(x, valid=[True, False, True]))
        4.0
        >>> float(weighted_sum(x[[0, 2]], weights=[0.5, 2.0]))
        6.5
    """
    xp, x, w, m = reduction_inputs(values, weights, valid)
    # S = ∑ₙ mₙ wₙ xₙ
    return xp.sum(_kept(xp, x, w, m), axis=axis, dtype=x.dtype)


def weighted_mean(
    values: Any,
    *,
    weights: Any = None,
    valid: Any = None,
    axis: Axis = None,
) -> Any:
    """The mean of x over the valid samples, weighted by w.

    x̄ = ∑ₙ mₙ wₙ xₙ / ∑ₙ mₙ wₙ, with m the validity, 1 or 0.

    Args:
        values: x, samples of any shape.
        weights: w ≥ 0, broadcast against ``values``; they need not sum to 1.
        valid: Which samples to use (True = use), broadcast against ``values``.
            A NaN in a valid sample gives NaN, and one in an invalid sample
            reaches neither the mean nor its gradient.
        axis: The axis or axes to average over; every axis if not given.

    Returns:
        x̄, the array without the reduced axes, NaN where the weights used sum
        to 0.

    Raises:
        BackendError: If an argument is a labelled array, or of another library.
        ValueError: If ``valid`` is not boolean.

    Example:
        >>> import numpy as np
        >>> x = np.asarray([[1.0, 4.0], [2.0, np.nan]])
        >>> weighted_mean(x, weights=[3.0, 1.0], valid=~np.isnan(x), axis=-1)
        array([1.75, 2.  ])
        >>> float(weighted_mean(x[0], weights=[0.0, 0.0]))
        nan
    """
    xp, x, w, m = reduction_inputs(values, weights, valid)
    # S = ∑ₙ mₙ wₙ xₙ
    total = xp.sum(_kept(xp, x, w, m), axis=axis, dtype=x.dtype)
    # W = ∑ₙ mₙ wₙ
    weight = xp.sum(used_weights(xp, x, w, m), axis=axis, dtype=x.dtype)
    return divide_or_nan(total, weight, xp)


def reduction_inputs(
    values: Any, weights: Any, valid: Any, *, params: tuple[Any, ...] = ()
) -> tuple[ModuleType, Any, Any, Any]:
    """The namespace of a reduction, and its inputs broadcast against each other.

    For a reduction that `weighted_sum`, `weighted_mean` and `Groups` do not
    cover, with their rules for masks and weights.

    Args:
        values: x.
        weights: w, or ``None``.
        valid: The validity, boolean (True = use), or ``None``.
        params: Other arguments, as in `floating_inputs`: they help to choose the
            namespace and the device, and never decide the dtype.

    Returns:
        The namespace, x and w in one floating dtype, and the validity as a
        boolean array; w and the validity stay ``None`` where they are.

    Raises:
        BackendError: If an argument is a labelled array, or of another library.
        ValueError: If ``valid`` is not boolean.

    Example:
        >>> import numpy as np
        >>> xp, x, w, m = reduction_inputs(np.ones((2, 3)), [1.0, 2.0, 3.0], None)
        >>> x.shape, w.shape, m
        ((2, 3), (2, 3), None)
    """
    if weights is None:
        xp, (x,) = floating_inputs(values, params=(valid, *params))
        w = None
    else:
        xp, (x, w) = floating_inputs(values, weights, params=(valid, *params))
    m = None if valid is None else boolean_like(valid, x, xp)
    given = [array for array in (x, w, m) if array is not None]
    broadcast = iter(xp.broadcast_arrays(*given))
    return (
        xp,
        next(broadcast),
        None if w is None else next(broadcast),
        None if m is None else next(broadcast),
    )


def _kept(xp: ModuleType, x: Any, w: Any, m: Any) -> Any:
    """m w x, where an invalid sample is replaced before the product."""
    # 0 in place of xₙ where mₙ = 0: 0 times NaN is NaN
    kept = x if m is None else xp.where(m, x, xp.zeros_like(x))
    return kept if w is None else kept * used_weights(xp, x, w, m)


def used_weights(xp: ModuleType, x: Any, w: Any, m: Any) -> Any:
    """The weight m w that each sample enters a reduction with.

    Args:
        xp: The namespace of the call.
        x: The samples, as `reduction_inputs` returns them.
        w: The weights in ``x``'s shape and dtype, or ``None`` for 1.
        m: The validity in ``x``'s shape, or ``None`` if every sample is valid.

    Returns:
        m w, in ``x``'s shape and dtype.

    Example:
        >>> import numpy as np
        >>> x = np.asarray([1.0, np.nan])
        >>> used_weights(np, x, None, np.asarray([True, False]))
        array([1., 0.])
    """
    weight = xp.ones_like(x) if w is None else w
    return weight if m is None else xp.where(m, weight, xp.zeros_like(weight))


#: The label of a sample that is in no group.
NO_GROUP = -1


@dataclasses.dataclass(frozen=True, eq=False)
class Groups:
    """Which samples belong to which group; built once, on the host.

    The tables are NumPy arrays and static: under JAX a `Groups` is a closure or
    a static argument, and never a traced one. It is compared by identity, so
    build it once; a new one compiles again.

    G groups of at most M samples cost G M reads, so one very large group
    among many small ones costs memory.

    Attributes:
        index: Positions of each group's samples, padded with 0, (G, M).
        member: Which entries of ``index`` are real, (G, M).
        labels: The group of each of the N samples, -1 for none.

    Example:
        >>> import numpy as np
        >>> groups = Groups.from_labels([0, 1, 0, -1], n_groups=3)
        >>> x = np.asarray([1.0, 2.0, 3.0, 100.0])
        >>> groups.sum(x)
        array([4., 2., 0.])
        >>> groups.mean(x)
        array([ 2.,  2., nan])
        >>> groups.count(x)
        array([2, 1, 0], dtype=int32)
        >>> x - groups.spread(groups.mean(x))
        array([-1.,  0.,  1., nan])
    """

    index: np.ndarray
    member: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_labels(cls, labels: Any, n_groups: int) -> Groups:
        """The groups of samples with the same label.

        Args:
            labels: The group of each sample, concrete integers in [0, G), or
                -1 for a sample in no group.
            n_groups: G, at least 1; a group may be empty.

        Returns:
            The groups.

        Raises:
            ValueError: If ``labels`` is not a 1-D array of integers, if
                one is neither -1 nor in [0, G), or if ``n_groups`` is below 1.
        """
        labels = np.asarray(labels)
        if n_groups < 1:
            msg = f"'n_groups' must be at least 1, got {n_groups}"
            raise ValueError(msg)
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            msg = (
                "'labels' must be a 1-D array of integers, got shape "
                f"{labels.shape} and dtype {labels.dtype}"
            )
            raise ValueError(msg)
        # a copy: the caller may write to its array later
        labels = labels.astype(np.int64)
        outside = (labels < NO_GROUP) | (labels >= n_groups)
        if outside.any():
            # segment_sum would drop it silently
            msg = (
                f"'labels' must be -1 or in [0, {n_groups}), got "
                f"{labels[outside][0]} at position {int(np.argmax(outside))}"
            )
            raise ValueError(msg)

        # --- the samples in a group, in group order ---
        position = np.flatnonzero(labels != NO_GROUP)
        order = position[np.argsort(labels[position], kind="stable")]
        sizes = np.bincount(labels[position], minlength=n_groups)

        # --- one padded row per group ---
        member = np.arange(int(sizes.max())) < sizes[:, None]
        index = np.zeros(member.shape, dtype=np.int64)
        # row by row, as `order` is
        index[member] = order
        for table in (index, member, labels):
            table.setflags(write=False)
        return cls(index=index, member=member, labels=labels)

    @property
    def n_groups(self) -> int:
        """G, the number of groups, empty ones included."""
        return self.index.shape[0]

    @property
    def n_samples(self) -> int:
        """N, the number of samples, those in no group included."""
        return self.labels.shape[0]

    def sum(self, values: Any, *, axis: int = -1, valid: Any = None) -> Any:
        """The sum of each group's valid samples.

        Sⱼ = ∑_{n : g(n) = j} mₙ xₙ, with g(n) the group of sample n and m the
        validity, 1 or 0.

        Args:
            values: x, with the N samples on ``axis``.
            axis: The axis of the samples.
            valid: Which samples to use (True = use), broadcast against
                ``values``. A NaN in a valid sample gives NaN.

        Returns:
            S, with the G groups on ``axis``. An empty group has sum 0.

        Raises:
            BackendError: If an argument is a labelled array, or of another library.
            ValueError: If ``axis`` does not hold N samples, or ``valid``
                is not boolean.
        """
        xp, x, _, m = self._inputs(values, None, valid, axis)
        x_rows, used = self._rows(xp, x, m)
        kept = xp.where(used, x_rows, xp.zeros_like(x_rows))
        # Sⱼ = ∑ mₙ xₙ over the row of group j
        return xp.moveaxis(xp.sum(kept, axis=-1, dtype=x.dtype), -1, axis)

    def count(self, values: Any, *, axis: int = -1, valid: Any = None) -> Any:
        """The number of valid samples in each group.

        Args:
            values: x, with the N samples on ``axis``; only its shape, its
                library and its device matter, so a NaN counts.
            axis: The axis of the samples.
            valid: Which samples to count (True = count), broadcast against
                ``values``.

        Returns:
            The counts, int32, with the G groups on ``axis``.

        Raises:
            BackendError: If an argument is a labelled array, or of another library.
            ValueError: If ``axis`` does not hold N samples, or ``valid``
                is not boolean.
        """
        xp, x, _, m = self._inputs(values, None, valid, axis)
        x_rows, used = self._rows(xp, x, m)
        ones = xp.astype(xp.broadcast_to(used, x_rows.shape), xp.int32)
        return xp.moveaxis(xp.sum(ones, axis=-1, dtype=xp.int32), -1, axis)

    def mean(
        self,
        values: Any,
        *,
        axis: int = -1,
        valid: Any = None,
        weights: Any = None,
    ) -> Any:
        """The mean of each group's valid samples, weighted by w.

        x̄ⱼ = Sⱼ / ∑_{n : g(n) = j} mₙ wₙ, with Sⱼ = ∑_{n : g(n) = j} mₙ wₙ xₙ.

        Args:
            values: x, with the N samples on ``axis``.
            axis: The axis of the samples.
            valid: Which samples to use (True = use), broadcast against
                ``values``. A NaN in a valid sample gives NaN.
            weights: w ≥ 0, broadcast against ``values``; they need not sum to 1.

        Returns:
            x̄, with the G groups on ``axis``, NaN where a group is empty or its
            weights sum to 0.

        Raises:
            BackendError: If an argument is a labelled array, or of another library.
            ValueError: If ``axis`` does not hold N samples, or ``valid``
                is not boolean.
        """
        xp, x, w, m = self._inputs(values, weights, valid, axis)
        x_rows, used = self._rows(xp, x, m)
        used = xp.broadcast_to(used, x_rows.shape)
        w_rows = None if w is None else self._gather(xp, w)
        # mₙ wₙ, 0 on the padding too
        weight_rows = used_weights(xp, x_rows, w_rows, used)
        kept = xp.where(used, x_rows, xp.zeros_like(x_rows))
        # Sⱼ = ∑ mₙ wₙ xₙ over the row of group j
        total = xp.sum(kept * weight_rows, axis=-1, dtype=x.dtype)
        weight = xp.sum(weight_rows, axis=-1, dtype=x.dtype)
        return xp.moveaxis(divide_or_nan(total, weight, xp), -1, axis)

    def spread(self, grouped: Any, *, axis: int = -1) -> Any:
        """Each sample's group value: the inverse lookup.

        yₙ = x_{g(n)}, so that ``x - groups.spread(groups.mean(x))`` is the
        anomaly from the group's mean.

        Args:
            grouped: One value per group, on ``axis``.
            axis: The axis of the groups.

        Returns:
            y, with the N samples on ``axis``, NaN for a sample in no group.

        Raises:
            BackendError: If ``grouped`` is a labelled array.
            ValueError: If ``axis`` does not hold G groups.
        """
        xp, (x,) = floating_inputs(grouped)
        self._check_axis(x, axis, self.n_groups, "groups")
        grouped_last = xp.moveaxis(x, axis, -1)
        labelled = self.labels != NO_GROUP
        # yₙ = x_{g(n)}; a sample in no group reads group 0, and is NaN below
        lookup = table_like(np.where(labelled, self.labels, 0), x, xp)
        values = xp.take(grouped_last, lookup, axis=-1)
        in_a_group = table_like(labelled, x, xp)
        spread = xp.where(in_a_group, values, xp.full_like(values, xp.nan))
        return xp.moveaxis(spread, -1, axis)

    # --- the gather -----------------------------------------------------------

    def _inputs(
        self,
        values: Any,
        weights: Any,
        valid: Any,
        axis: int,
    ) -> tuple[ModuleType, Any, Any, Any]:
        """The namespace, and x, w and m with the samples on the last axis."""
        xp, *arrays = reduction_inputs(values, weights, valid)
        self._check_axis(arrays[0], axis, self.n_samples, "samples")
        x, w, m = (
            None if array is None else xp.moveaxis(array, axis, -1) for array in arrays
        )
        return xp, x, w, m

    @staticmethod
    def _check_axis(x: Any, axis: int, size: int, what: str) -> None:
        """Refuse an axis of another size: a gather would not notice it."""
        if not -x.ndim <= axis < x.ndim or x.shape[axis] != size:
            msg = f"axis {axis} of shape {x.shape} must hold the {size} {what}"
            raise ValueError(msg)

    def _gather(self, xp: ModuleType, x: Any) -> Any:
        """(…, N) samples as (…, G, M) rows, padding included."""
        flat = table_like(self.index.reshape(-1), x, xp)
        rows = xp.take(x, flat, axis=-1)
        return xp.reshape(rows, (*x.shape[:-1], *self.index.shape))

    def _rows(self, xp: ModuleType, x: Any, m: Any) -> tuple[Any, Any]:
        """The rows of x, and which of their entries are real and valid."""
        x_rows = self._gather(xp, x)
        used = table_like(self.member, x, xp)
        if m is not None:
            used = xp.logical_and(used, self._gather(xp, m))
        return x_rows, used
