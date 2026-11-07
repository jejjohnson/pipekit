"""Dim-level functions that move data by position: pad, roll, shift, diff, …

Phase 2 of the dim-functions plan (``docs/design/api/dim_functions.md``,
ADRs A11, A12 and A16). Like the functions of `pipekit_array.dims`, each
runs on any carrier with a registered `pipekit_ops.NamedArrayAdapter` and
calls an adapter method of the same name, when there is one, instead of
the generic path.

- `pad` — widen named dims, with a constant or from the data's edges.
- `roll`, `shift` — move data along named dims, cyclically or with a fill.
- `diff` — the n-th forward difference along one dim.
- `cumreduce` — a cumulative function along one dim (``"cumulative_sum"``).
- `where` — keep entries where a mask is True, aligned by dim name.

**Index-only.** None reads a label. Labels stay on their positions
(xarray's ``roll_coords=False``), and a dim whose size changes loses its
labels, which ``rewrap`` does by itself.

**Dtype preserved.** A fill of NaN needs a floating dtype: `pad`,
`shift` and `where` raise `TypeError` on integer or bool data unless a
fill value is given, where xarray would promote to float (A12).

The `Operator` forms live in `pipekit_array.dim_operators`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pipekit_array.dims import (
    Reduction,
    _aligned_mask,
    _axes,
    _fast_path,
    _resolve,
    adapter_of,
    namespace_of,
)


__all__ = ["cumreduce", "diff", "pad", "roll", "shift", "where"]

#: the `pad` modes NumPy, JAX and dask all share
PAD_MODES = ("constant", "edge", "reflect", "wrap")


def _holds_nan(xp: Any, dtype: Any) -> bool:
    """Whether ``dtype`` has a NaN: floating or complex, not integer or bool."""
    isdtype = getattr(xp, "isdtype", None)
    if isdtype is None:
        # dask has no isdtype; its dtypes are NumPy's
        return dtype.kind in "fc"
    return bool(isdtype(dtype, ("real floating", "complex floating")))


def _no_nan_fill(what: str, argument: str, dtype: Any) -> TypeError:
    return TypeError(
        f"{what}: {dtype} data has no NaN to fill with; pass {argument}= "
        "(the dtype is kept, never promoted to float)."
    )


def _widths(dim: str, width: int | Sequence[int]) -> tuple[int, int]:
    """``(before, after)`` from an int or a pair; negative widths raise."""
    pair = (width, width) if isinstance(width, int) else tuple(width)
    if len(pair) != 2 or any(w < 0 for w in pair):
        raise ValueError(
            f"pad: {dim}= takes a width ≥ 0 or a pair (before, after), got {width!r}."
        )
    return (pair[0], pair[1])


def pad(
    x: Any,
    /,
    *,
    mode: str = "constant",
    constant_values: float | None = None,
    **pad_width: int | tuple[int, int],
) -> Any:
    """Widen named dims at either end (xarray's ``pad``, four modes).

    y_{…, t, …} = x_{…, t - b, …} for b ≤ t < b + N; the new entries follow
    ``mode``.

    A padded dim loses its labels (xarray pads its coordinate with NaN).

    Args:
        x: A named carrier.
        mode: ``"constant"`` (``constant_values``), ``"edge"`` (repeat the
            end entry), ``"reflect"`` (mirror, end excluded) or ``"wrap"``
            (the other end's entries).
        constant_values: The fill for ``mode="constant"``; None is NaN.
        **pad_width: ``dim=`` a width b for both ends, or a pair
            ``(before, after)``.

    Raises:
        KeyError: A padded dim is not a dim of ``x``.
        ValueError: ``mode`` is not one of the four, a width is negative,
            or ``constant_values`` is given with another mode.
        TypeError: ``mode="constant"`` on integer or bool data without
            ``constant_values``.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> series = DimArray(np.arange(3.0), ("time",))
        >>> pad(series, time=(2, 0)).data
        array([nan, nan,  0.,  1.,  2.])
        >>> pad(series, mode="wrap", time=1).data
        array([2., 0., 1., 2., 0.])
        >>> counts = DimArray(np.arange(3), ("time",))
        >>> pad(counts, time=1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TypeError: pad: int64 data has no NaN to fill with; pass constant_values=
    """
    adapter = adapter_of(x, "pad")
    dims = adapter.dims(x)
    _axes(dims, list(pad_width), "pad")
    if mode not in PAD_MODES:
        raise ValueError(f"pad: mode {mode!r} is not one of {PAD_MODES}.")
    if mode != "constant" and constant_values is not None:
        raise ValueError(f"pad: constant_values is for mode='constant', not {mode!r}.")
    widths = {d: _widths(d, w) for d, w in pad_width.items()}
    data = adapter.data(x)
    xp = namespace_of(data)
    if (
        mode == "constant"
        and constant_values is None
        and not _holds_nan(xp, data.dtype)
    ):
        raise _no_nan_fill("pad", "constant_values", data.dtype)
    fast = _fast_path(adapter, "pad")
    if fast is not None:
        return fast(x, widths, mode=mode, constant_values=constant_values)
    full_widths = tuple(widths.get(d, (0, 0)) for d in dims)
    if mode == "constant":
        fill = float("nan") if constant_values is None else constant_values
        padded = xp.pad(data, full_widths, mode=mode, constant_values=fill)
    else:
        padded = xp.pad(data, full_widths, mode=mode)
    return adapter.rewrap(x, padded, dims)


def roll(x: Any, /, **shifts: int) -> Any:
    """Shift data cyclically along named dims; labels stay (xarray's ``roll``).

    y_{…, t, …} = x_{…, (t - k) mod N, …}.

    Args:
        x: A named carrier.
        **shifts: ``dim=k``; a positive k moves data to higher positions.
            k may be traced under ``jit``.

    Raises:
        KeyError: A rolled dim is not a dim of ``x``.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> series = DimArray(np.arange(4.0), ("time",), {"time": tuple("abcd")})
        >>> rolled = roll(series, time=1)
        >>> rolled.data, rolled.labels["time"]
        (array([3., 0., 1., 2.]), ('a', 'b', 'c', 'd'))
        >>> bool(np.array_equal(roll(rolled, time=-1).data, series.data))
        True
    """
    adapter = adapter_of(x, "roll")
    dims = adapter.dims(x)
    axes = _axes(dims, list(shifts), "roll")
    fast = _fast_path(adapter, "roll")
    if fast is not None:
        return fast(x, dict(shifts))
    data = adapter.data(x)
    xp = namespace_of(data)
    # y_t = x_{(t - k) mod N}
    rolled = xp.roll(data, shift=tuple(shifts.values()), axis=axes)
    return adapter.rewrap(x, rolled, dims)


def shift(x: Any, /, *, fill_value: float | None = None, **shifts: int) -> Any:
    """Shift data along named dims, filling the vacated entries (xarray's ``shift``).

    y_{…, t, …} = x_{…, t - k, …} if 0 ≤ t - k < N, else the fill.

    Labels stay on their positions. A shift of at least the dim's length
    fills it entirely.

    Args:
        x: A named carrier.
        fill_value: The value of the vacated entries; None is NaN.
        **shifts: ``dim=k``, a Python int (static under ``jit``); a
            positive k moves data to higher positions.

    Raises:
        KeyError: A shifted dim is not a dim of ``x``.
        TypeError: Integer or bool data without ``fill_value``.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> series = DimArray(np.arange(4.0), ("time",))
        >>> shift(series, time=1).data
        array([nan,  0.,  1.,  2.])
        >>> shift(DimArray(np.arange(4), ("time",)), time=-2, fill_value=-1).data
        array([ 2,  3, -1, -1])
    """
    adapter = adapter_of(x, "shift")
    dims = adapter.dims(x)
    axes = _axes(dims, list(shifts), "shift")
    data = adapter.data(x)
    xp = namespace_of(data)
    if fill_value is None and not _holds_nan(xp, data.dtype):
        raise _no_nan_fill("shift", "fill_value", data.dtype)
    fast = _fast_path(adapter, "shift")
    if fast is not None:
        return fast(x, dict(shifts), fill_value=fill_value)
    fill = float("nan") if fill_value is None else fill_value
    shifted = data
    for axis, offset in zip(axes, shifts.values(), strict=True):
        size = shifted.shape[axis]
        vacated = min(abs(offset), size)
        if vacated == 0:
            continue
        # the entries that stay: the first N - |k| for k > 0, the last for k < 0
        kept = slice(0, size - vacated) if offset > 0 else slice(vacated, size)
        index = tuple(kept if a == axis else slice(None) for a in range(data.ndim))
        filled_shape = tuple(
            vacated if a == axis else n for a, n in enumerate(shifted.shape)
        )
        filled = xp.full(filled_shape, fill, dtype=data.dtype)
        parts = [filled, shifted[index]] if offset > 0 else [shifted[index], filled]
        shifted = xp.concat(parts, axis=axis)
    return adapter.rewrap(x, shifted, dims)


def diff(x: Any, dim: str, /, n: int = 1, *, label: str = "upper") -> Any:
    """The n-th forward difference along ``dim`` (xarray's ``diff``).

    Δy_t = y_{t+1} - y_t, applied n times: N entries become N - n.

    The result's labels on ``dim`` are the minuend's (``"upper"``, positions
    n … N - 1) or the subtrahend's (``"lower"``, positions 0 … N - n - 1).

    Args:
        x: A named carrier.
        dim: The dim to difference along.
        n: The order, n ≥ 0; static under ``jit``.
        label: ``"upper"`` or ``"lower"``.

    Raises:
        KeyError: ``dim`` is not a dim of ``x``.
        ValueError: ``n`` is negative, or ``label`` is not ``"upper"`` or
            ``"lower"``.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> series = DimArray(np.array([1.0, 4.0, 9.0, 16.0]), ("time",))
        >>> diff(series, "time").data
        array([3., 5., 7.])
        >>> labelled = DimArray(series.data, ("time",), {"time": tuple("abcd")})
        >>> second = diff(labelled, "time", 2, label="lower")
        >>> second.data, second.labels["time"]
        (array([2., 2.]), ('a', 'b'))
    """
    adapter = adapter_of(x, "diff")
    dims = adapter.dims(x)
    _axes(dims, (dim,), "diff")
    if n < 0:
        raise ValueError(f"diff: the order n must be ≥ 0, got {n}.")
    if label not in ("upper", "lower"):
        raise ValueError(f"diff: label must be 'upper' or 'lower', got {label!r}.")
    fast = _fast_path(adapter, "diff")
    if fast is not None:
        return fast(x, dim, n=n, label=label)
    out = x
    for _ in range(n):
        upper = adapter.isel(out, {dim: slice(1, None)})
        lower = adapter.isel(out, {dim: slice(None, -1)})
        xp = namespace_of(adapter.data(upper))
        # Δy_t = y_{t+1} - y_t
        difference = xp.subtract(adapter.data(upper), adapter.data(lower))
        # the labelled slice carries its labels onto the difference
        labelled = upper if label == "upper" else lower
        out = adapter.rewrap(labelled, difference, dims)
    return out


def cumreduce(x: Any, fn: Reduction, dim: str) -> Any:
    """A cumulative function along ``dim``; labels and length unchanged.

    c_t = ∑_{τ ≤ t} x_τ for ``"cumulative_sum"``.

    No implicit ``skipna`` (xarray's ``cumsum`` skips NaN by default): ask
    for ``"nancumsum"``.

    Args:
        x: A named carrier.
        fn: An array-namespace name (``"cumulative_sum"``,
            ``"cumulative_prod"``, ``"nancumsum"``) or a function taking
            ``axis=`` that keeps the length.
        dim: The dim to accumulate along.

    Raises:
        KeyError: ``dim`` is not a dim of ``x``.
        ValueError: The namespace has no ``fn``, or ``fn`` changed the
            length of ``dim``.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> change = DimArray(np.array([1.0, 2.0, 3.0]), ("time",))
        >>> cumreduce(change, "cumulative_sum", "time").data
        array([1., 3., 6.])
        >>> gaps = DimArray(np.array([1.0, np.nan, 3.0]), ("time",))
        >>> cumreduce(gaps, "nancumsum", "time").data
        array([1., 1., 4.])
    """
    adapter = adapter_of(x, "cumreduce")
    dims = adapter.dims(x)
    (axis,) = _axes(dims, (dim,), "cumreduce")
    fast = _fast_path(adapter, "cumreduce")
    if fast is not None:
        return fast(x, fn, dim)
    data = adapter.data(x)
    func = _resolve(fn, namespace_of(data), "cumreduce")
    # c_t = ∑_{τ ≤ t} x_τ (or fn's own accumulation)
    out = func(data, axis=axis)
    if tuple(out.shape) != tuple(data.shape):
        raise ValueError(
            f"cumreduce: fn changed the shape {tuple(data.shape)} to "
            f"{tuple(out.shape)}; a cumulative function keeps the length of {dim!r}."
        )
    return adapter.rewrap(x, out, dims)


def where(x: Any, cond: Any, /, other: float | None = None) -> Any:
    """Keep ``x`` where ``cond`` is True, ``other`` elsewhere (xarray's ``where``).

    y = x where v, else the fill; True keeps (A16).

    ``cond`` aligns by dim name, as ``reduce(where=)`` does: a named mask
    whose dims are among ``x``'s, so a ``time y x`` cloud mask applies to a
    ``time band y x`` cube; or a plain array covering ``x``'s leading dims.
    It is never right-broadcast.

    Args:
        x: A named carrier.
        cond: A bool mask v, named or plain.
        other: The fill where ``cond`` is False; None is NaN.

    Raises:
        ValueError: A named ``cond`` has a dim ``x`` lacks or a size that
            differs, or a plain one does not match ``x``'s leading dims.
        TypeError: Integer or bool data without ``other``.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.ones((2, 3, 2)), ("time", "band", "x"))
        >>> valid = DimArray(np.array([[True, False], [True, True]]), ("time", "x"))
        >>> clear = where(cube, valid)
        >>> clear.dims, clear.data[0, :, 1]
        (('time', 'band', 'x'), array([nan, nan, nan]))
        >>> where(cube, valid, 0.0).data[0, 0]
        array([1., 0.])
    """
    adapter = adapter_of(x, "where")
    dims = adapter.dims(x)
    data = adapter.data(x)
    mask = _aligned_mask(cond, dims, data, "where")
    xp = namespace_of(data)
    if other is None and not _holds_nan(xp, data.dtype):
        raise _no_nan_fill("where", "other", data.dtype)
    fast = _fast_path(adapter, "where")
    if fast is not None:
        return fast(x, mask, other)
    fill = xp.asarray(float("nan") if other is None else other, dtype=data.dtype)
    # y = x if v else the fill
    kept = xp.where(mask, data, fill)
    return adapter.rewrap(x, kept, dims)
