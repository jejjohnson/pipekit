"""Reduce over named dims: statistics, cumulative functions, quantiles, contraction.

Each is written once, over the carrier's registered
`pipekit_ops.NamedArrayAdapter`; see `pipekit_array.named` for the rules they share.

- `reduce`, `argreduce`: ``reduce(..., where=valid)`` leaves masked entries out.
- `cumreduce`: a cumulative function along one dim (``"cumulative_sum"``).
- `quantile`: a sequence ``q`` adds a leading ``quantile`` dim labelled with ``q``
  (A17).
- `dot`: the sum of a product over named dims, as ``xr.dot``; a shared dim not
  summed is a batch dim and appears once (A18).
"""

from __future__ import annotations

import string
from collections import Counter
from collections.abc import Sequence
from typing import Any

from pipekit_ops.adapters import backend_name

from pipekit_array.named._dispatch import (
    Reduction,
    adapter_of,
    aligned_mask,
    axes_of,
    fast_path,
    join_backends,
    join_mask,
    namespace_of,
    resolve_function,
)
from pipekit_array.named._shape import Labels, set_labels


__all__ = ["METHODS", "argreduce", "cumreduce", "dot", "quantile", "reduce"]


def reduce(
    x: Any, fn: Reduction, dim: str | Sequence[str], *, where: Any | None = None
) -> Any:
    """Reduce over one or more named dims.

    Args:
        x: A named carrier.
        fn: An array-namespace name (``"mean"``, ``"max"``, ``"nanmean"``) or a
            function taking ``axis=`` (and ``where=``, when given).
        dim: A dim name, or several.
        where: Optional bool mask of the entries to keep, passed to ``fn``
            as ``where=`` (``"mean"`` then divides by the kept count). Either
            a named carrier, aligned to ``x`` by dim name (its dims must be
            among ``x``'s), or a plain array covering ``x``'s *leading* dims
            (``where.shape == shape[:where.ndim]``): a ``(sample,)`` mask on
            ``(sample, y, x)`` data, as `Padded.valid` is. Never
            right-broadcast. ``"max"`` / ``"min"`` need an ``initial`` for
            a mask: pass a function that supplies one.

    Raises:
        ValueError: A plain ``where`` does not match ``x``'s leading dims,
            or a named one has a dim ``x`` lacks.
        TypeError: ``where`` on data other than NumPy or JAX (the
            reductions of dask, PyTorch and CuPy take no ``where=``).

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.ones((12, 2, 4)), ("time", "band", "x"))
        >>> reduce(cube, "mean", "time").dims
        ('band', 'x')
        >>> batch = DimArray(np.array([[1.0], [3.0], [9.0]]), ("sample", "x"))
        >>> reduce(batch, "mean", "sample", where=np.array([True, True, False])).data
        array([2.])
    """
    adapter = adapter_of(x, "reduce")
    names = (dim,) if isinstance(dim, str) else tuple(dim)
    fast = fast_path(adapter, "reduce")
    dims = adapter.dims(x)
    data = adapter.data(x)
    if where is not None:
        _require_where_backend(data)
    mask = None if where is None else aligned_mask(where, dims, data, "reduce")
    if mask is not None:
        x, data, mask = join_mask(adapter, x, data, mask, "reduce")
    if fast is not None:
        return fast(x, fn, names) if mask is None else fast(x, fn, names, where=mask)
    axes = axes_of(dims, names, "reduce")
    func = resolve_function(fn, namespace_of(data), "reduce")
    out = func(data, axis=axes) if mask is None else func(data, axis=axes, where=mask)
    kept = tuple(d for d in dims if d not in names)
    return adapter.rewrap(x, out, kept)


#: array modules whose reductions take NumPy's ``where=`` (jax: tracers too)
_WHERE_MODULES = ("numpy", "jax", "jaxlib")


def _require_where_backend(data: Any) -> None:
    """Raise unless ``data``'s reductions honour ``where=`` (NumPy, JAX).

    The reductions of dask, CuPy and PyTorch take no ``where=``
    (array-api-compat forwards it to ``torch.mean``, which rejects it;
    ``cupy.mean`` / ``cupy.sum`` have no such parameter), so a mask there would
    fail deep in the backend, or be ignored by a reduction that swallows
    keywords.
    """
    module = type(data).__module__.split(".")[0]
    if module in _WHERE_MODULES or backend_name(data) == "jax":
        return
    raise TypeError(
        f"reduce: where= needs NumPy or JAX data, got {type(data).__name__} "
        f"({module}), whose reductions take no where=. Compute or convert the "
        "carrier first, or mask with NaN and a nan-reduction ('nanmean')."
    )


def argreduce(x: Any, fn: Reduction, dim: str) -> Any:
    """The index of a reduction along one dim (``"argmax"``, ``"argmin"``).

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> ndvi = DimArray(np.random.default_rng(0).random((12, 4)), ("time", "x"))
        >>> argreduce(ndvi, "argmax", "time").dims
        ('x',)
    """
    adapter = adapter_of(x, "argreduce")
    fast = fast_path(adapter, "argreduce")
    if fast is not None:
        return fast(x, fn, dim)
    dims = adapter.dims(x)
    (axis,) = axes_of(dims, (dim,), "argreduce")
    data = adapter.data(x)
    func = resolve_function(fn, namespace_of(data), "argreduce")
    out = func(data, axis=axis)
    kept = tuple(d for d in dims if d != dim)
    return adapter.rewrap(x, out, kept)


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
    (axis,) = axes_of(dims, (dim,), "cumreduce")
    fast = fast_path(adapter, "cumreduce")
    if fast is not None:
        return fast(x, fn, dim)
    data = adapter.data(x)
    func = resolve_function(fn, namespace_of(data), "cumreduce")
    # c_t = ∑_{τ ≤ t} x_τ (or fn's own accumulation)
    out = func(data, axis=axis)
    if tuple(out.shape) != tuple(data.shape):
        raise ValueError(
            f"cumreduce: fn changed the shape {tuple(data.shape)} to "
            f"{tuple(out.shape)}; a cumulative function keeps the length of {dim!r}."
        )
    return adapter.rewrap(x, out, dims)


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
    axes = axes_of(dims, names, "quantile")
    check_method(method, "quantile")
    values = probabilities(q, "quantile")
    many = isinstance(values, tuple)
    kept = tuple(d for d in dims if d not in names)
    if many and "quantile" in kept:
        raise ValueError(
            f"quantile: 'quantile' is already a dim of {dims}; rename it first."
        )
    fast = fast_path(adapter, "quantile")
    if fast is not None:
        return fast(x, values, names, method=method, skipna=skipna)
    import numpy as np

    data = adapter.data(x)
    if backend_name(data) == "dask":
        # dask's quantile needs every reduced axis in one chunk
        data = data.rechunk(dict.fromkeys(axes, -1))
    xp = namespace_of(data)
    func = resolve_function("nanquantile" if skipna else "quantile", xp, "quantile")
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
    axes_of(union, names, "dot")
    if len(union) > len(string.ascii_letters):
        raise ValueError(
            f"dot: {len(union)} distinct dims, but einsum has only "
            f"{len(string.ascii_letters)} letters."
        )
    datas = [a.data(x) for a, x in zip(adapters, xs, strict=True)]
    datas = join_backends(datas, "dot")
    fast = fast_path(adapter, "dot")
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
    return xp.sum(product, axis=axes_of(union, names, "dot"))
