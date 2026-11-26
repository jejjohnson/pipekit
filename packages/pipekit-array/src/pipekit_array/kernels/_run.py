"""Run a kernel on raw arrays, and on named carriers through their adapter."""

from __future__ import annotations

import dataclasses
import functools
import itertools
from collections.abc import Callable, Sequence
from typing import Any, cast

from pipekit_ops import NamedArrayAdapter, get_adapter
from pipekit_ops.carriers import BackendError, backend_name

from pipekit_array.kernels._spec import KernelSpec, _RunsKernels, kernel_output_dims
from pipekit_array.named._dispatch import join_backends, namespace_of


__all__ = ["derive_geometry", "run_kernel"]


def _one_backend(
    datas: Sequence[Any], kernel: Callable[..., Any], spec: KernelSpec
) -> tuple[list[Any], str]:
    """The inputs on one backend, NumPy joined to JAX, and its name.

    Raises:
        BackendError: Inputs on several other backends, a backend the spec
            does not list, or ``vectorize=True`` on dask arrays
            (``np.vectorize`` would compute the whole graph; the xarray path
            stays lazy).
    """
    joined = join_backends(datas, getattr(kernel, "__name__", "KernelOp"))
    backends = {backend_name(d) for d in joined}
    backend = "jax" if "jax" in backends else next(iter(backends), "numpy")
    if backend not in spec.backends:
        raise BackendError(
            f"the kernel declares backends {spec.backends}, not {backend!r}."
        )
    if backend == "dask" and spec.vectorize:
        raise BackendError(
            "vectorize=True on dask arrays would compute them eagerly; pass an "
            "xr.DataArray (apply_ufunc keeps it lazy) or a vectorized kernel."
        )
    return joined, backend


def _call(
    kernel: Callable[..., Any], spec: KernelSpec, kwargs: dict[str, Any], backend: str
) -> Callable[..., Any]:
    bound = functools.partial(kernel, **kwargs)
    if not spec.vectorize:
        return bound
    signature = spec.signature.replace(" ", "")
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.vectorize(bound, signature=signature)
    import numpy as np

    return np.vectorize(bound, signature=signature)


def run_kernel(
    kernel: Callable[..., Any],
    spec: KernelSpec,
    core: dict[str, str],
    carriers: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """Run ``kernel`` on raw arrays or named carriers; the dispatch behind `KernelOp`.

    Raises:
        BackendError: Carriers on several backends other than NumPy and JAX,
            or a backend the spec does not list.
        TypeError: A raw array next to named carriers.
        ValueError: A core dim is missing from its input, or a dim has two
            sizes.
    """
    adapters = [get_adapter(x) for x in carriers]
    if all(a is None for a in adapters):
        arrays, backend = _one_backend(carriers, kernel, spec)
        return _call(kernel, spec, kwargs, backend)(*arrays)
    for i, adapter in enumerate(adapters):
        if adapter is None:
            raise TypeError(
                f"input {i} is a raw array next to named carriers; wrap it as a "
                "pipekit_array.DimArray with its dims."
            )
    named: list[NamedArrayAdapter] = [a for a in adapters if a is not None]
    # a class attribute check: isinstance against a runtime Protocol costs ~10 µs
    own = [cast(_RunsKernels, a) for a in named if hasattr(type(a), "apply_kernel")]
    if own:
        if len({type(a) for a in named}) > 1:
            raise BackendError(
                f"{type(own[0]).__name__} carriers cannot mix with others in one "
                "kernel call; convert them to one family first."
            )
        return own[0].apply_kernel(kernel, spec, core, kwargs, *carriers)
    return _run_named(kernel, spec, core, list(carriers), named, kwargs)


@dataclasses.dataclass(frozen=True)
class _Plan:
    """Everything about one call that depends only on dims: computed once."""

    # per input: axes to put (present loop dims, core dims) in order (None if
    # already in order), and where to insert size-1 axes for loop dims it lacks
    arrange: tuple[tuple[tuple[int, ...] | None, tuple[int, ...]], ...]
    # per input: (axis, dim) pairs, to check each dim has one size
    size_checks: tuple[tuple[tuple[int, str], ...], ...]
    # per output: the permutation back to the target order (None if none is
    # needed), and that order
    restore: tuple[tuple[tuple[int, ...] | None, tuple[str, ...]], ...]
    # how many loop dims the inputs have together
    n_loop: int


@functools.lru_cache(maxsize=1024)
def _plan(
    spec: KernelSpec,
    core_items: tuple[tuple[str, str], ...],
    dims: tuple[tuple[str, ...], ...],
) -> _Plan:
    core = dict(core_items)
    core_dims = [tuple(core[c] for c in group) for group in spec.inputs]
    loop: list[str] = []
    for i, (x_dims, cores) in enumerate(zip(dims, core_dims, strict=True)):
        missing = [c for c in cores if c not in x_dims]
        if missing:
            raise ValueError(
                f"input {i}: core dim(s) {missing} not in its dims {x_dims}."
            )
        for d in x_dims:
            if d not in cores and d not in loop:
                loop.append(d)
    arrange = []
    for x_dims, cores in zip(dims, core_dims, strict=True):
        present = [d for d in loop if d in x_dims]
        axes = tuple(x_dims.index(d) for d in (*present, *cores))
        inserts = tuple(pos for pos, d in enumerate(loop) if d not in x_dims)
        arrange.append((_unless_identity(axes), inserts))
    size_checks = tuple(tuple(enumerate(x_dims)) for x_dims in dims)
    targets = kernel_output_dims(spec, core, dims)
    restore = []
    for group, target in zip(spec.outputs, targets, strict=True):
        produced = (*loop, *(core[c] for c in group))
        axes = tuple(produced.index(d) for d in target)
        restore.append((_unless_identity(axes), target))
    return _Plan(tuple(arrange), size_checks, tuple(restore), len(loop))


def _unless_identity(axes: tuple[int, ...]) -> tuple[int, ...] | None:
    return None if axes == tuple(range(len(axes))) else axes


def _run_named(
    kernel: Callable[..., Any],
    spec: KernelSpec,
    core: dict[str, str],
    carriers: list[Any],
    adapters: list[NamedArrayAdapter],
    kwargs: dict[str, Any],
) -> Any:
    if len(carriers) > 1:
        carriers = _check_by_kind(carriers, adapters)
    datas = [a.data(x) for a, x in zip(adapters, carriers, strict=True)]
    datas, backend = _one_backend(datas, kernel, spec)
    dims = tuple(tuple(a.dims(x)) for a, x in zip(adapters, carriers, strict=True))
    plan = _plan(spec, tuple(sorted(core.items())), dims)
    sizes: dict[str, int] = {}
    # one input cannot disagree with itself
    checked_sizes = plan.size_checks if len(datas) > 1 else ()
    for i, (data, checks) in enumerate(zip(datas, checked_sizes, strict=False)):
        shape = data.shape
        for axis, d in checks:
            n = shape[axis]
            if sizes.setdefault(d, n) != n:
                raise ValueError(f"input {i}: dim {d!r} has size {n}, not {sizes[d]}.")
    xp = namespace_of(datas[0])
    arrays = []
    for data, (axes, inserts) in zip(datas, plan.arrange, strict=True):
        moved = _permute(xp, data, axes)
        for position in inserts:
            moved = xp.expand_dims(moved, axis=position)
        arrays.append(moved)
    result = _call(kernel, spec, kwargs, backend)(*arrays)
    many = len(plan.restore) > 1
    results = tuple(result) if many else (result,)
    # as full_loop does, without its cost: every array has all the loop axes here
    loops = {x.shape[: plan.n_loop] for x in arrays}
    loop = loops.pop() if len(loops) == 1 else _loop_shape(list(loops))
    template, template_adapter = carriers[0], adapters[0]
    wrapped = []
    for out, (axes, target) in zip(results, plan.restore, strict=True):
        full = _broadcast(out, loop, len(target) - plan.n_loop)
        moved_out = _permute(xp, full, axes)
        wrapped.append(template_adapter.rewrap(template, moved_out, target))
    return tuple(wrapped) if many else wrapped[0]


def full_loop(kernel: Callable[..., Any], spec: KernelSpec) -> Callable[..., Any]:
    """``kernel`` with every output broadcast to the loop dims of all the inputs.

    An output may ignore an input (z of a position on a sphere does not
    depend on the longitude), and then lacks its dims or has them in size 1.
    """
    if spec.vectorize:  # vectorize maps over all the loop dims itself
        return kernel
    in_cores = [len(group) for group in spec.inputs]
    out_cores = [len(group) for group in spec.outputs]

    def run(*arrays: Any, **kwargs: Any) -> Any:
        result = kernel(*arrays, **kwargs)
        loops = [
            x.shape[: len(x.shape) - n] for x, n in zip(arrays, in_cores, strict=True)
        ]
        loop = _loop_shape(loops)
        if len(out_cores) == 1:
            return _broadcast(result, loop, out_cores[0])
        outs = zip(result, out_cores, strict=True)
        return tuple(_broadcast(out, loop, n) for out, n in outs)

    return run


def _loop_shape(shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    """The shapes broadcast together, aligned at the right."""
    if all(shape == shapes[0] for shape in shapes):
        return shapes[0]
    reversed_shapes = [tuple(reversed(shape)) for shape in shapes]
    sizes = itertools.zip_longest(*reversed_shapes, fillvalue=1)
    return tuple(next((n for n in axis if n != 1), 1) for axis in sizes)[::-1]


def _broadcast(out: Any, loop: tuple[int, ...], n_core: int) -> Any:
    """``out`` in the shape of the loop dims and its own core dims."""
    shape = getattr(out, "shape", None)
    if shape is None or len(shape) < n_core:
        return out
    target = (*loop, *shape[len(shape) - n_core :])
    if shape == target:
        return out
    return namespace_of(out).broadcast_to(out, target)


def _permute(xp: Any, data: Any, axes: tuple[int, ...] | None) -> Any:
    """``permute_dims``, skipped when the plan says the order is already right."""
    if axes is None:
        return data
    return xp.permute_dims(data, axes)


def _check_by_kind(carriers: list[Any], adapters: list[NamedArrayAdapter]) -> list[Any]:
    """``check_compatible`` within each carrier family; use what it returns."""
    out = list(carriers)
    families: dict[int, list[int]] = {}
    for i, adapter in enumerate(adapters):
        families.setdefault(id(adapter), []).append(i)
    for members in families.values():
        if len(members) < 2:
            continue
        checked = adapters[members[0]].check_compatible(*(carriers[i] for i in members))
        for i, x in zip(members, checked, strict=True):
            out[i] = x
    return out


def derive_geometry(
    geometry: dict[str, Callable[[Any], Any]], first: Any
) -> dict[str, Any]:
    """The kernel's derived-geometry keywords, computed from the first carrier.

    Raises:
        ValueError: The carrier's adapter reports ``batch_dims`` (one geometry
            per item), but keyword inputs are passed whole (F4); the message
            names ``map_over``.
    """
    if not geometry:
        return {}
    adapter = get_adapter(first)
    batch = getattr(adapter, "batch_dims", None)
    batched = batch(first) if batch is not None else ()
    if batched:
        raise ValueError(
            f"geometry= on a carrier batched over {batched} (one geometry per "
            "item), but keyword inputs are passed whole: use "
            f"map_over(x, op, {batched[0]!r}) so each item has its own."
        )
    return {name: fn(first) for name, fn in geometry.items()}


def anonymous(x: Any) -> Any:
    """A DataArray without its name and attributes, which were another quantity's."""
    out = x.rename(None)
    out.attrs = {}
    return out
