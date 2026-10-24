"""`to_jax` — a `pipekit_array.KernelOp` as a `JaxOperator` (``pipekit-jax[array]``).

A `KernelOp` that holds arrays (a fitted background mean μ and covariance
Σ, a learned target t) needs them as pytree leaves under
``eqx.filter_jit`` and ``eqx.filter_grad``. `to_jax` generates that shell
from the `KernelOp` instead of a hand-written operator per kernel:

- **leaves** — every parameter that is an array or holds arrays (a
  `DimArray` of statistics), its arrays converted with ``jnp.asarray``;
- **static** — the kernel, its spec, the core-dim map, the geometry
  functions and every non-array parameter (``eps``, a dim name).

The call is the same dispatch as the `KernelOp`'s
(`pipekit_array.kernel.run_kernel`), so the same carriers work.

It lives here, not on `KernelOp`, because pipekit-jax already reaches
pipekit-array (through pipekit-cycle's ``[array]`` extra): an edge from
pipekit-array to pipekit-jax would close a cycle (PK-P6).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pipekit_jax.operator import JaxOperator


__all__ = ["JaxKernelOp", "to_jax"]

_HINT = "pipekit_jax.to_jax needs pipekit-array: pip install 'pipekit-jax[array]'"


def _kernel_module() -> Any:
    try:
        from pipekit_array import kernel
    except ImportError as exc:
        raise ImportError(_HINT) from exc
    return kernel


def _is_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) or eqx.is_array(value)


def _holds_arrays(value: Any) -> bool:
    """An array, or a pytree with array leaves (a `DimArray`): a parameter leaf."""
    return any(_is_array(leaf) for leaf in jax.tree_util.tree_leaves(value))


def _to_jnp(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf) if _is_array(leaf) else leaf, value
    )


class JaxKernelOp(JaxOperator):
    """A `KernelOp` whose array parameters are pytree leaves.

    Build one with `to_jax`. Compile with ``eqx.filter_jit``: the module
    holds arrays, so ``jax.jit(op)`` cannot hash it.

    Holds the kernel and geometry functions, so its config is a debug repr
    only (``forbid_in_yaml``); persist learned arrays with
    ``serialize_weights``.
    """

    forbid_in_yaml: ClassVar[bool] = True

    # leaves: the array parameters, by keyword
    params: dict[str, Any]
    # static: everything the call needs that is not an array
    kernel: Callable[..., Any] = eqx.field(static=True)
    spec: Any = eqx.field(static=True)
    core: tuple[tuple[str, str], ...] = eqx.field(static=True)
    geometry: tuple[tuple[str, Callable[[Any], Any]], ...] = eqx.field(static=True)
    static_params: tuple[tuple[str, Any], ...] = eqx.field(static=True)

    def _apply(self, *carriers: Any) -> Any:
        kernel_module = _kernel_module()
        if len(carriers) != len(self.spec.inputs):
            raise TypeError(
                f"{self.spec.signature!r} takes {len(self.spec.inputs)} input(s), "
                f"got {len(carriers)}."
            )
        derived = kernel_module.derive_geometry(dict(self.geometry), carriers[0])
        kwargs = dict(self.static_params) | dict(self.params) | derived
        return kernel_module.run_kernel(
            self.kernel, self.spec, dict(self.core), carriers, kwargs
        )


def to_jax(op: Any) -> JaxKernelOp:
    """A `JaxKernelOp` with ``op``'s kernel, spec, core and geometry.

    Parameters that are or hold arrays (a `DimArray`) become leaves, their
    arrays converted with ``jnp.asarray``; the rest are static and must be
    hashable (numbers, strings, tuples).

    Args:
        op: A `pipekit_array.KernelOp`.

    Raises:
        ImportError: pipekit-array is not installed (``pipekit-jax[array]``).
        TypeError: ``op`` is not a `KernelOp`, or a non-array parameter is
            unhashable (it would have to be static).

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray, KernelOp
        >>> def shift(x, offset):
        ...     return x + offset
        >>> op = KernelOp(shift, "()->()", core={}, offset=np.float32(1.0))
        >>> jop = to_jax(op)
        >>> sorted(jop.params)
        ['offset']
    """
    kernel_module = _kernel_module()
    if not isinstance(op, kernel_module.KernelOp):
        raise TypeError(f"to_jax takes a KernelOp, got {type(op).__name__}.")
    arrays: dict[str, Any] = {}
    static: list[tuple[str, Any]] = []
    for name, value in op.params.items():
        if _holds_arrays(value):
            arrays[name] = _to_jnp(value)
            continue
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(
                f"to_jax: parameter {name!r} ({type(value).__name__}) is neither an "
                "array (a leaf) nor hashable (a static field)."
            ) from exc
        static.append((name, value))
    return JaxKernelOp(
        params=arrays,
        kernel=op.kernel,
        spec=op.spec,
        core=tuple(sorted(op.core.items())),
        geometry=tuple(sorted(op.geometry.items())),
        static_params=tuple(sorted(static)),
    )
