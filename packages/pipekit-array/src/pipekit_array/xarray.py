"""The ``xr.DataArray`` adapter (``pipekit-array[xarray]``).

Registered lazily: importing pipekit-array never imports xarray; the
adapter loads the first time a DataArray reaches `pipekit.get_adapter`
(`pipekit.register_lazy_adapter`). This module is the only place
pipekit-array imports xarray.

- **Exact coordinates.** `XarrayAdapter.check_compatible` aligns with
  ``join="exact"``: two cubes whose x differ by 1e-7 raise instead of
  inner-joining to an empty result.
- **Lazy on dask.** `KernelOp` runs as one ``xr.apply_ufunc`` call,
  ``dask="parallelized"``, one task per chunk; the dim-level fast paths
  (``reduce``, ``argreduce``, ``take_along``, ``stack``, ``concat``,
  ``combine``, ``transpose``) are native xarray calls, so nothing computes
  until ``.compute()``.
- **``DataArray`` only** (PK-P5): ``Dataset`` and ``DataTree`` dispatch
  stays in xrops.

`from_xarray` turns a DataArray into a `DimArray`, keeping the labels of
dims that have a 1-D index coordinate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import xarray as xr
from pipekit import Signature, get_adapter, register_adapter

from pipekit_array.dimarray import DimArray
from pipekit_array.dims import check_index_labels
from pipekit_array.kernel import BackendError, KernelSpec, kernel_output_dims


__all__ = ["XarrayAdapter", "from_xarray", "register"]


class XarrayAdapter:
    """The `pipekit.NamedArrayAdapter` for ``xr.DataArray``, with native fast paths."""

    # -- the protocol ----------------------------------------------------

    def dims(self, x: xr.DataArray) -> tuple[str, ...]:
        """``x.dims``."""
        return tuple(str(d) for d in x.dims)

    def data(self, x: xr.DataArray) -> Any:
        """``x.data``: NumPy, dask or JAX, never computed."""
        return x.data

    def labels(self, x: xr.DataArray, dim: str) -> tuple[Any, ...] | None:
        """The values of ``dim``'s 1-D index coordinate, or None."""
        if dim in x.indexes and x.coords[dim].ndim == 1:
            values = x.coords[dim].values
            # tolist() turns datetime64[ns] into integers; keep the scalars
            if values.dtype.kind in "mM":
                return tuple(values)
            return tuple(values.tolist())
        return None

    def isel(self, x: xr.DataArray, indexers: dict[str, Any]) -> xr.DataArray:
        """``x.isel``; coordinates follow the data."""
        return x.isel(indexers)

    def rewrap(self, x: xr.DataArray, data: Any, dims: tuple[str, ...]) -> xr.DataArray:
        """A DataArray around new data, keeping coords and attrs of surviving dims.

        A coordinate is kept when all its dims survive with their sizes.
        """
        sizes = dict(zip(dims, data.shape, strict=True))
        coords = {
            name: coord
            for name, coord in x.coords.items()
            if all(sizes.get(d) == x.sizes[d] for d in coord.dims)
        }
        return xr.DataArray(data, dims=dims, coords=coords, attrs=x.attrs, name=x.name)

    def check_compatible(self, *xs: xr.DataArray) -> tuple[xr.DataArray, ...]:
        """The inputs back if their shared coordinates are exactly equal.

        Raises:
            ValueError: Coordinates differ; xarray would otherwise inner-join.
        """
        try:
            xr.align(*xs, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError(
                f"DataArray inputs are not aligned (join='exact'): {exc}"
            ) from exc
        return xs

    def signature(self, x: xr.DataArray) -> Signature:
        """Dims with their sizes, and the dtype's name."""
        dims = tuple((str(d), int(n)) for d, n in x.sizes.items())
        return Signature(dims=dims, dtype=str(x.dtype))

    # -- dim-level fast paths --------------------------------------------

    def transpose(self, x: xr.DataArray, order: tuple[str, ...]) -> xr.DataArray:
        """``x.transpose``."""
        return x.transpose(*order)

    def reduce(
        self, x: xr.DataArray, fn: str | Callable[..., Any], dims: tuple[str, ...]
    ) -> xr.DataArray:
        """``x.reduce`` with the array-namespace function, lazy on dask.

        A name resolves to NumPy's function (``"nanmean"`` included), so NaN
        handling matches the generic path: DataArray's own ``mean`` would
        skip NaN by default.
        """
        return x.reduce(_numpy_function(fn, "reduce"), dim=list(dims))

    def argreduce(
        self, x: xr.DataArray, fn: str | Callable[..., Any], dim: str
    ) -> xr.DataArray:
        """``x.reduce`` with NumPy's ``argmax`` / ``argmin`` (or ``fn``)."""
        return x.reduce(_numpy_function(fn, "argreduce"), dim=dim)

    def take_along(self, x: xr.DataArray, index: Any, dim: str) -> xr.DataArray:
        """One entry of ``dim`` per position, lazily on dask, in ``x``'s dim order.

        An ``apply_ufunc`` over ``take_along_axis``, not a vectorized
        ``isel``: dask cannot index with a lazy indexer. Attributes are kept,
        auxiliary coordinates on ``dim`` (``quality(time, x)``) are gathered
        with the same index, and ``dim``'s own coordinate is dropped.
        """
        rest = [str(d) for d in x.dims if d != dim]
        idx = _as_dataarray(index)  # raises for an unnamed index
        # the labels are checked on the index itself: idx drops its coordinates
        index_adapter = get_adapter(index)
        if index_adapter is not None:
            check_index_labels(self, x, index_adapter, index, index_adapter.dims(index))
        out = _gather(x, idx, dim)
        aux = {
            str(name): _gather(xr.DataArray(c.variable), idx, dim).variable
            for name, c in x.coords.items()
            if dim in c.dims and name != dim
        }
        if aux:
            out = out.assign_coords(aux)
        return out.transpose(*rest, ...)

    def stack(self, xs: Sequence[xr.DataArray], dim: str) -> xr.DataArray:
        """A new leading dim; coordinates must be identical (never aligned)."""
        if dim in xs[0].dims:
            raise ValueError(
                f"stack: {dim!r} is already a dim of {tuple(xs[0].dims)}; use concat."
            )
        first = tuple(xs[0].dims)
        for i, x in enumerate(xs[1:], start=1):
            # xr.concat would broadcast over the union of dims
            if tuple(x.dims) != first:
                raise ValueError(
                    f"stack: input {i} has dims {tuple(x.dims)}, not {first}."
                )
        checked = self.check_compatible(*xs)
        return xr.concat(
            checked, dim=dim, join="exact", coords="minimal", compat="identical"
        )

    def concat(self, xs: Sequence[xr.DataArray], dim: str) -> xr.DataArray:
        """Along an existing dim, joining its coordinate; others must be identical."""
        return xr.concat(
            list(xs), dim=dim, join="exact", coords="minimal", compat="identical"
        )

    def combine(self, fn: Callable[..., Any], *xs: xr.DataArray) -> xr.DataArray:
        """``apply_ufunc(fn, …, join="exact")``, lazy on dask."""
        checked = self.check_compatible(*xs)
        dims = tuple(checked[0].dims)
        ordered = [x.transpose(*dims) for x in checked]
        lazy = any(_is_dask(x) for x in ordered)
        return xr.apply_ufunc(
            fn,
            *ordered,
            dask="parallelized",
            join="exact",
            # dask needs the result dtype up front: read it from a one-element call
            output_dtypes=[_probe_dtype(fn, ordered)] if lazy else None,
        )

    # -- KernelOp ---------------------------------------------------------

    def apply_kernel(
        self,
        kernel: Callable[..., Any],
        spec: KernelSpec,
        core: dict[str, str],
        kwargs: dict[str, Any],
        *xs: xr.DataArray,
    ) -> Any:
        """A `KernelOp` call as one ``xr.apply_ufunc``: lazy, exact, in input order.

        Core dims move last; ``kwargs`` (params and derived geometry) are
        passed whole; output-only core dims take their sizes from
        ``spec.sizes``; outputs come back in the first input's dim order (F2).

        Raises:
            BackendError: The spec does not list ``"xarray"``.
        """
        if "xarray" not in spec.backends:
            raise BackendError(
                f"the kernel declares backends {spec.backends}, not 'xarray'."
            )
        checked = self.check_compatible(*xs)
        in_dims = [[core[c] for c in group] for group in spec.inputs]
        out_dims = [[core[c] for c in group] for group in spec.outputs]
        letter_size: dict[str, int] = {}
        for x, group in zip(checked, spec.inputs, strict=True):
            for letter in group:
                letter_size[letter] = int(x.sizes[core[letter]])
        output_sizes = {core[out]: letter_size[source] for out, source in spec.sizes}
        result = xr.apply_ufunc(
            kernel,
            *checked,
            input_core_dims=in_dims,
            output_core_dims=out_dims,
            kwargs=kwargs,
            vectorize=spec.vectorize,
            dask="parallelized",
            join="exact",
            output_dtypes=_output_dtypes(kernel, spec, kwargs, checked, letter_size),
            dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": output_sizes},
        )
        targets = kernel_output_dims(
            spec, core, [tuple(map(str, x.dims)) for x in checked]
        )
        if len(out_dims) > 1:
            return tuple(r.transpose(*t) for r, t in zip(result, targets, strict=True))
        return result.transpose(*targets[0])


def _gather(values: xr.DataArray, idx: xr.DataArray, dim: str) -> xr.DataArray:
    """``values`` at ``idx`` along ``dim``; coordinates on ``dim`` are dropped."""
    return xr.apply_ufunc(
        _take_along_last,
        values,
        idx,
        input_core_dims=[[dim], []],
        dask="parallelized",
        join="exact",
        keep_attrs=True,
        output_dtypes=[values.dtype],
        # the gathered dim must be whole in each block
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def _take_along_last(values: Any, index: Any) -> Any:
    """values (..., n) at index (...) along the last axis."""
    import numpy as np

    # apply_ufunc gives each input its own loop dims in one order, size 1 or
    # absent where it lacks one: broadcast both to the common loop shape
    loop = np.broadcast_shapes(values.shape[:-1], np.shape(index))
    values = np.broadcast_to(values, (*loop, values.shape[-1]))
    index = np.broadcast_to(index, loop)
    picked = np.take_along_axis(values, index[..., None], axis=-1)
    return picked[..., 0]


def _output_dtypes(
    kernel: Callable[..., Any],
    spec: KernelSpec,
    kwargs: dict[str, Any],
    xs: Sequence[xr.DataArray],
    letter_size: dict[str, int],
) -> list[Any] | None:
    """Each output's dtype, which dask must know before it runs anything.

    ``spec.dtypes`` when declared; else found by calling the kernel once on
    ones with only the core dims (the loop dims dropped), so a mask or an
    index kernel gives bool or int; the first input's dtype when that call
    fails.
    """
    if not any(_is_dask(x) for x in xs):
        # eager: let the kernel (or np.vectorize) decide; a declared dtype
        # would cast the results
        return None
    if spec.dtypes:
        return [xs[0].dtype if d == "same" else np.dtype(d) for d in spec.dtypes]
    fallback = [xs[0].dtype] * len(spec.outputs)
    probes = [
        np.ones(tuple(letter_size[c] for c in group), dtype=x.dtype)
        for x, group in zip(xs, spec.inputs, strict=True)
    ]
    try:
        out = kernel(*probes, **kwargs)
    except Exception:
        return fallback
    outs = out if len(spec.outputs) > 1 else (out,)
    return [np.asarray(o).dtype for o in outs]


def _is_dask(x: xr.DataArray) -> bool:
    return type(x.data).__module__.startswith("dask.")


def _probe_dtype(fn: Callable[..., Any], xs: Sequence[xr.DataArray]) -> Any:
    """``fn``'s result dtype on one-element arrays of the inputs' dtypes."""
    probes = [np.ones((1,) * x.ndim, dtype=x.dtype) for x in xs]
    try:
        return np.asarray(fn(*probes)).dtype
    except Exception:
        return xs[0].dtype


def _numpy_function(fn: Any, what: str) -> Callable[..., Any]:
    if callable(fn):
        return fn
    resolved = getattr(np, fn, None)
    if resolved is None:
        raise ValueError(f"{what}: the array namespace has no {fn!r}.")
    return resolved


def _as_dataarray(index: Any) -> xr.DataArray:
    # bare: an index's own coordinates (a scalar band left by sel) would
    # conflict with x's
    if isinstance(index, xr.DataArray):
        return xr.DataArray(index.data, dims=index.dims)
    if isinstance(index, DimArray):
        return xr.DataArray(index.data, dims=index.dims)
    raise TypeError(
        f"take_along on a DataArray needs a named index, got {type(index).__name__}."
    )


def from_xarray(da: xr.DataArray) -> DimArray:
    """A `DimArray` with the same data and dims; 1-D index coordinates become labels."""
    adapter = XarrayAdapter()
    dims = adapter.dims(da)
    labels = {d: adapter.labels(da, d) for d in dims}
    return DimArray(da.data, dims, {d: v for d, v in labels.items() if v is not None})


def register() -> None:
    """Register `XarrayAdapter` for ``xr.DataArray`` (the lazy loader calls this)."""
    register_adapter(xr.DataArray, XarrayAdapter())
