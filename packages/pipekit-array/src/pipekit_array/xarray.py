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

import xarray as xr
from pipekit import Signature, register_adapter

from pipekit_array.dimarray import DimArray
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
            return tuple(x.coords[dim].values.tolist())
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
        """A named reduction method (``x.mean(dim=…)``) or ``x.reduce(fn, dim=…)``."""
        if isinstance(fn, str):
            method = getattr(x, fn, None)
            if method is None:
                raise ValueError(f"reduce: DataArray has no reduction {fn!r}.")
            return method(dim=list(dims))
        return x.reduce(fn, dim=list(dims))

    def argreduce(
        self, x: xr.DataArray, fn: str | Callable[..., Any], dim: str
    ) -> xr.DataArray:
        """``x.argmax(dim=…)`` / ``x.argmin(dim=…)``, or ``x.reduce(fn, dim=…)``."""
        if isinstance(fn, str):
            method = getattr(x, fn, None)
            if method is None:
                raise ValueError(f"argreduce: DataArray has no {fn!r}.")
            return method(dim=dim)
        return x.reduce(fn, dim=dim)

    def take_along(self, x: xr.DataArray, index: Any, dim: str) -> xr.DataArray:
        """Vectorized ``x.isel({dim: index})``, in ``x``'s dim order."""
        rest = [d for d in x.dims if d != dim]
        out = x.isel({dim: _as_dataarray(index)})
        return out.transpose(*rest).drop_vars(dim, errors="ignore")

    def stack(self, xs: Sequence[xr.DataArray], dim: str) -> xr.DataArray:
        """A new leading dim; coordinates must be identical (never aligned)."""
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
        return xr.apply_ufunc(
            fn,
            *ordered,
            dask="parallelized",
            join="exact",
            output_dtypes=[ordered[0].dtype],
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
            output_dtypes=[checked[0].dtype] * len(out_dims),
            dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": output_sizes},
        )
        targets = kernel_output_dims(
            spec, core, [tuple(map(str, x.dims)) for x in checked]
        )
        if len(out_dims) > 1:
            return tuple(r.transpose(*t) for r, t in zip(result, targets, strict=True))
        return result.transpose(*targets[0])


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
