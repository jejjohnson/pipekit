"""Dim-level statistics and algebra: quantiles over dims, and contraction.

Phase 4 of the dim-functions plan (``docs/design/api/dim_functions.md``,
ADRs A17, A18 and A22). Like the functions of `pipekit_array.dims`, each
runs on any carrier with a registered `pipekit_ops.NamedArrayAdapter`,
works on positions only, and calls an adapter method of the same name,
when there is one, instead of the generic path.

- `quantile` — quantiles over named dims; a sequence ``q`` adds a leading
  ``quantile`` dim labelled with ``q``.
- `dot` — the sum of a product over named dims, as ``xr.dot``; a shared
  dim not summed is a batch dim and appears once.

Both set the labels of their new dims through the adapter's optional
``relabel`` (A22), as `pipekit_array.dim_shapes` does.

The `Operator` forms live in `pipekit_array.dim_operators`.
"""

from __future__ import annotations

import string
from collections import Counter
from collections.abc import Sequence
from typing import Any

from pipekit_ops.adapters import backend_name

from pipekit_array.dim_shapes import Labels, set_labels
from pipekit_array.dims import (
    _axes,
    _fast_path,
    _resolve,
    adapter_of,
    namespace_of,
    require_one_backend,
)


__all__ = ["METHODS", "dot", "quantile"]

#: the quantile methods every backend computes: JAX's six (A18)
METHODS: tuple[str, ...] = (
    "linear",
    "lower",
    "higher",
    "midpoint",
    "nearest",
    "inverted_cdf",
)


def check_method(method: str, what: str) -> None:
    """Raise `ValueError` unless ``method`` is one of `METHODS`."""
    if method not in METHODS:
        raise ValueError(
            f"{what}: method {method!r} is not supported; use one of {METHODS} "
            "(the six JAX computes)."
        )


def probabilities(q: float | Sequence[float], what: str) -> float | tuple[float, ...]:
    """``q`` as one float, or a tuple of floats; each must lie in [0, 1].

    Raises:
        ValueError: A probability is outside [0, 1].
    """
    values = float(q) if isinstance(q, (int, float)) else tuple(float(v) for v in q)
    outside = [v for v in _as_tuple(values) if not 0.0 <= v <= 1.0]
    if outside:
        raise ValueError(f"{what}: q must lie in [0, 1], got {outside}.")
    return values


def _as_tuple(values: float | tuple[float, ...]) -> tuple[float, ...]:
    return values if isinstance(values, tuple) else (values,)


def quantile(
    x: Any,
    q: float | Sequence[float],
    dim: str | Sequence[str],
    /,
    *,
    method: str = "linear",
    skipna: bool = False,
) -> Any:
    """Quantiles over named dims (xarray's ``quantile``).

    Q_{k…} = quantile q_k of {x_{…, d…} : d over ``dim``}.

    A sequence ``q`` adds a leading dim ``quantile``, labelled with ``q``;
    a float drops it. A float input keeps its dtype. NaN propagates unless
    ``skipna`` (no implicit ``skipna``, A12).

    Args:
        x: A named carrier.
        q: A probability in [0, 1], or a sequence of them; static under
            ``jit``.
        dim: A dim name, or several, to reduce over.
        method: One of `METHODS`, the six JAX supports.
        skipna: Ignore NaN (``nanquantile``).

    Raises:
        KeyError: A name is not a dim of ``x``.
        ValueError: ``method`` is not one of `METHODS`, a probability is
            outside [0, 1], or a sequence ``q`` meets a kept dim named
            ``quantile``.
        TypeError: The adapter has no ``relabel`` and ``q`` is a sequence.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.arange(24.0).reshape(4, 2, 3), ("time", "band", "x"))
        >>> spread = quantile(cube, [0.1, 0.9], "time")
        >>> spread.dims, spread.labels["quantile"]
        (('quantile', 'band', 'x'), (0.1, 0.9))
        >>> quantile(cube, 0.5, ("time", "x"), method="lower").dims
        ('band',)
    """
    adapter = adapter_of(x, "quantile")
    names = (dim,) if isinstance(dim, str) else tuple(dim)
    dims = adapter.dims(x)
    axes = _axes(dims, names, "quantile")
    check_method(method, "quantile")
    values = probabilities(q, "quantile")
    many = isinstance(values, tuple)
    kept = tuple(d for d in dims if d not in names)
    if many and "quantile" in kept:
        raise ValueError(
            f"quantile: 'quantile' is already a dim of {dims}; rename it first."
        )
    fast = _fast_path(adapter, "quantile")
    if fast is not None:
        return fast(x, values, names, method=method, skipna=skipna)
    import numpy as np

    data = adapter.data(x)
    if backend_name(data) == "dask":
        # dask's quantile needs every reduced axis in one chunk
        data = data.rechunk(dict.fromkeys(axes, -1))
    xp = namespace_of(data)
    func = _resolve("nanquantile" if skipna else "quantile", xp, "quantile")
    floating = np.issubdtype(data.dtype, np.floating)
    # q in the data's float dtype: NumPy promotes float32 to float64 for a
    # float64 sequence q; dask needs a NumPy q
    q_array = np.asarray(values, dtype=data.dtype if floating else None)
    # one q as a NumPy scalar: dask takes a 0-d array for a sequence
    q_value = q_array if many else q_array[()]
    out = func(data, q_value, axis=axes, method=method)
    result = adapter.rewrap(x, out, ("quantile", *kept) if many else kept)
    if not isinstance(values, tuple):
        return result
    labels: dict[str, Labels] = {"quantile": values}
    return set_labels(adapter, result, labels, "quantile")


def dot(*xs: Any, dim: str | Sequence[str]) -> Any:
    """The sum of the product of carriers over named dims (``xr.dot``).

    s_{o…} = ∑_{d ∈ dim} ∏ₖ x⁽ᵏ⁾_{…}, where o… are the dims not in ``dim``.

    The output dims are every input dim not in ``dim``, in order of first
    appearance. A dim shared by several inputs and not in ``dim`` is a
    batch dim: it appears once, so ``dot`` is one named-index ``einsum``,
    never ``tensordot`` (A17). Shared dims must agree exactly in size and
    labels; dims from inputs after the first carry those inputs' labels.

    Algorithm:
        1. Give each distinct dim a letter, in order of first appearance.
        2. Write one subscript per input from its dims, and the output's
           from the dims not in ``dim``.
        3. ``xp.einsum``; for a namespace without it, broadcast every
           input to the union of the dims, multiply, and ``xp.sum`` over
           ``dim`` (exact, but holds the full product).
        4. ``rewrap`` on the first input, then ``relabel`` the dims taken
           from later inputs.

    Args:
        *xs: Named carriers of one family and one backend.
        dim: A dim name, or several, to sum over; required (A17): xarray
            would sum every shared dim, so a dim shared by accident would
            vanish.

    Raises:
        ValueError: No carrier is given, a shared dim differs in size or
            labels, or there are more dims than letters.
        KeyError: A name in ``dim`` is a dim of no input.
        BackendError: The inputs sit on more than one backend.
        TypeError: The adapter has no ``relabel`` and a dim from a later
            input has labels.

    Examples:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> x = DimArray(np.ones((12, 13)), ("time", "band"))
        >>> w = DimArray(np.ones((12, 13, 3)), ("time", "band", "component"))
        >>> s = dot(x, w, dim="band")
        >>> s.dims, bool(np.allclose(s.data, 13.0))
        (('time', 'component'), True)
        >>> pixels = DimArray(np.ones((2, 13, 6)), ("time", "band", "pixel"))
        >>> labelled = {"component": ("pc1", "pc2", "pc3")}
        >>> weights = DimArray(np.ones((13, 3)), ("band", "component"), labelled)
        >>> scores = dot(pixels, weights, dim="band")
        >>> scores.dims, scores.labels["component"]
        (('time', 'pixel', 'component'), ('pc1', 'pc2', 'pc3'))
    """
    if not xs:
        raise ValueError("dot needs at least one carrier.")
    adapters = [adapter_of(x, "dot") for x in xs]
    adapter = adapters[0]
    names = (dim,) if isinstance(dim, str) else tuple(dim)
    each_dims = [a.dims(x) for a, x in zip(adapters, xs, strict=True)]
    union = tuple(dict.fromkeys(d for dims in each_dims for d in dims))
    _axes(union, names, "dot")
    if len(union) > len(string.ascii_letters):
        raise ValueError(
            f"dot: {len(union)} distinct dims, but einsum has only "
            f"{len(string.ascii_letters)} letters."
        )
    datas = [a.data(x) for a, x in zip(adapters, xs, strict=True)]
    require_one_backend(datas, "dot")
    fast = _fast_path(adapter, "dot")
    if fast is not None:
        return fast(list(xs), names)
    # only the shared dims must combine: an empty slice of every other dim
    # checks them without touching the data
    counts = Counter(d for dims in each_dims for d in dims)
    parts = [
        a.isel(x, {d: slice(0, 0) for d in dims if counts[d] == 1})
        for a, x, dims in zip(adapters, xs, each_dims, strict=True)
    ]
    adapter.check_compatible(*parts)
    out_dims = tuple(d for d in union if d not in names)
    xp = namespace_of(datas[0])
    einsum = getattr(xp, "einsum", None)
    if einsum is None:
        out = _product_sum(xp, datas, each_dims, union, names)
    else:
        letters = dict(zip(union, string.ascii_letters, strict=False))
        inputs = ",".join("".join(letters[d] for d in dims) for dims in each_dims)
        output = "".join(letters[d] for d in out_dims)
        # s_{o…} = ∑_{d ∈ dim} ∏ₖ x⁽ᵏ⁾, e.g. "ab,abc->ac" for time band, time
        # band component over band
        out = einsum(f"{inputs}->{output}", *datas)
    result = adapter.rewrap(xs[0], out, out_dims)
    # dims the first input lacks take the labels of the first input with them
    labels: dict[str, Labels] = {}
    for a, x, dims in zip(adapters[1:], xs[1:], each_dims[1:], strict=True):
        for d in dims:
            if d in out_dims and d not in each_dims[0] and d not in labels:
                labels[d] = a.labels(x, d)
    return set_labels(adapter, result, labels, "dot")


def _product_sum(
    xp: Any,
    datas: Sequence[Any],
    each_dims: Sequence[tuple[str, ...]],
    union: tuple[str, ...],
    names: tuple[str, ...],
) -> Any:
    """`dot` without ``einsum``: broadcast to ``union``, multiply, sum ``names``."""
    aligned = []
    for data, dims in zip(datas, each_dims, strict=True):
        order = tuple(dims.index(d) for d in union if d in dims)
        moved = xp.permute_dims(data, order)
        # a length-1 axis for each dim the input lacks, in union order
        for position, d in enumerate(union):
            if d not in dims:
                moved = xp.expand_dims(moved, axis=position)
        aligned.append(moved)
    # p_{u…} = ∏ₖ x⁽ᵏ⁾_{u…}, broadcast over the union u…
    product = aligned[0]
    for factor in aligned[1:]:
        product = product * factor
    # s_{o…} = ∑_{d ∈ dim} p_{u…}
    return xp.sum(product, axis=_axes(union, names, "dot"))
