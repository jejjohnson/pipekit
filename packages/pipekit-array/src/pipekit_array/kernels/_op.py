"""`KernelOp`, a kernel as an `pipekit_ops.Operator`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from pipekit_ops import Operator, backend_of

from pipekit_array._recorded import recorded
from pipekit_array.kernels._run import derive_geometry, run_kernel
from pipekit_array.kernels._spec import KernelSpec, kernel_output_signature


__all__ = ["KernelOp"]


@recorded("pipekit_array.kernel")
class KernelOp(Operator):
    """An L1 kernel wrapped once, run on raw arrays and on every named carrier.

    Args:
        kernel: The L1 function: arrays (core dims last) and keyword params in,
            arrays out.
        spec: A `KernelSpec`, or its signature string.
        core: Each core letter of the spec → the carrier dim it stands for,
            e.g. ``{"c": "band"}``; the same kernel serves ``"band"`` and
            ``"variable"``.
        geometry: Kernel keyword → a function of the first carrier returning
            arrays (``{"spacing": spacing_of}``), computed per call and passed
            whole.
        params: Keyword parameters passed to every call, as a mapping: the
            way to pass one named ``core``, ``geometry`` or ``params``.
        **kwargs: More keyword parameters passed to every call.

    Raises:
        ValueError: ``core`` misses or invents a letter of the spec, or a
            parameter has the name of a derived-geometry keyword.
        TypeError: ``geometry`` is not a mapping of callables (pass a kernel
            parameter named ``geometry`` through ``params=``).

    Every output has the loop dims of all the inputs, in the first input's
    order. On DataArrays an output keeps the coordinates and has no name or
    attributes: a density is not in the pressure's units.

    ``KernelOp`` holds a user function, so its config is a debug repr only
    (``forbid_in_yaml``, like `pipekit_ops.Lambda`); a named operator built on
    one by composition (``geoops.NDVI``) is what serialises.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> def nd(a, b, eps):
        ...     return (a - b) / (a + b + eps)
        >>> op = KernelOp(nd, "(),()->()", core={}, eps=1e-10)
        >>> nir = DimArray(np.full((2, 3), 0.4), ("y", "x"))
        >>> red = DimArray(np.full((2, 3), 0.1), ("y", "x"))
        >>> op(nir, red).dims
        ('y', 'x')
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        kernel: Callable[..., Any],
        spec: KernelSpec | str,
        /,
        core: dict[str, str],
        geometry: dict[str, Callable[[Any], Any]] | None = None,
        params: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.kernel = kernel
        self.spec = spec if isinstance(spec, KernelSpec) else KernelSpec(spec)
        letters = set(self.spec.letters)
        if set(core) != letters:
            raise ValueError(
                f"core must map exactly the spec's letters {sorted(letters)}, "
                f"got {sorted(core)}."
            )
        self.core = dict(core)
        if geometry is not None and not (
            isinstance(geometry, Mapping) and all(map(callable, geometry.values()))
        ):
            raise TypeError(
                "geometry= maps kernel keywords to functions of the first carrier; "
                "pass a kernel parameter named 'geometry' as params={'geometry': ...}."
            )
        self.geometry = dict(geometry or {})
        self.params = {**dict(params or {}), **kwargs}
        clash = sorted(set(self.params) & set(self.geometry))
        if clash:
            raise ValueError(
                f"{clash} are both parameters and derived-geometry keywords; "
                "the kernel can take each name once."
            )

    def _place_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        backends: tuple[str, ...] | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any], bool]:
        """The kernel's backends for NumPy and JAX inputs; other inputs pass on.

        ``run_kernel`` joins a mix the kernel takes, and checks dask and xarray.
        """
        if backends is None and self.backends is None:
            found = {backend_of(x) for x in args} - {None}
            runs_on = {*self.spec.backends}
            if not found <= {"numpy", "jax"} or found <= runs_on:
                return args, kwargs, False
            backends = self.spec.backends
        return super()._place_inputs(args, kwargs, backends)

    def _apply(self, *carriers: Any) -> Any:
        if len(carriers) != len(self.spec.inputs):
            raise TypeError(
                f"{self.spec.signature!r} takes {len(self.spec.inputs)} input(s), "
                f"got {len(carriers)}."
            )
        derived = derive_geometry(self.geometry, carriers[0])
        kwargs = self.params | derived
        return run_kernel(self.kernel, self.spec, self.core, carriers, kwargs)

    def get_config(self) -> dict[str, Any]:
        module = getattr(self.kernel, "__module__", "?")
        name = getattr(self.kernel, "__qualname__", repr(self.kernel))
        return {
            "kernel": f"{module}:{name}",
            "spec": self.spec.signature,
            "core": self.core,
            "geometry": sorted(self.geometry),
            "params": sorted(self.params),
        }

    def compute_output_signature(self, input_signature: Any) -> Any:
        """The output signature(s) for a one-input kernel; ``None`` otherwise.

        A two-output kernel gives a tuple of signatures. With several inputs
        the other inputs' dims are unknown here, so the answer is ``None``.
        The dtype is the one ``spec.dtypes`` declares (``"same"``: the
        input's), else unknown: a kernel may return a mask or an index.
        """
        return kernel_output_signature(self.spec, self.core, input_signature)
