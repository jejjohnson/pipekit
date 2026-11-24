"""`KernelSpec` and `KernelOp` — one L1 kernel, written once, on every carrier.

A shapes-only kernel declares the axes it consumes and produces with a
generalized-ufunc signature (``"(c)->()"``, ``"(r,c),(c),(c,d)->(r)"``),
the convention NumPy (``np.vectorize``), JAX (``jnp.vectorize``) and
xarray (``apply_ufunc`` core dims) already share. `KernelOp` wraps the
kernel once and runs it:

    raw arrays        → the kernel, called as is
    named carriers    → core dims moved last by name, the kernel on the
                        data, the result rewrapped by the adapter
                        (NumPy eagerly, JAX traced under jit)
    an adapter with   → that adapter's own path (the xarray adapter:
    ``apply_kernel``    one ``apply_ufunc``, lazy on dask)

Kernel conventions:

1. **Consumed axes last.** Core dims arrive as the trailing axes.
2. **Broadcast over leading axes,** or set ``vectorize=True`` and pay for
   ``np.vectorize`` / ``jnp.vectorize``.
3. **Non-carrier inputs are keyword parameters** (a background mean, a
   threshold), passed whole and never mapped. Geometry derived from the
   carrier (``geometry=``) is passed the same way; geometry that varies
   per pixel has the carrier's dims and is a positional input instead.
4. **Write against the array namespace** (``xp.linalg.solve``) to run on
   every backend. A kernel that needs a JAX-only library (lineax)
   declares ``backends=("jax",)`` (spike finding F3).

Rules from the spike (geotoolz#516):

- **Outputs keep the first input's dim order** (F2): a gufunc puts core
  dims last, so each output is transposed back, with dims new to the
  output last.
- **Every output has the loop dims of all the inputs.** An output that
  ignores an input is broadcast (#246).
- **A DataArray's name and attributes are dropped,** because the kernel
  changes the quantity and its unit. Coordinates keep theirs, and other
  carriers keep what their adapter's ``rewrap`` keeps.
- **``geometry=`` needs one geometry per call** (F4): on a carrier whose
  adapter reports ``batch_dims`` (per-item transforms), `KernelOp` raises
  and names ``map_over``.
- **An output-only core dim declares its size** (F5): ``KernelSpec.sizes``
  maps it to an input core dim, so a lazy backend can plan the output.

Carriers of different kinds combine by dim name and size (a geo carrier
``"band row col"`` with a `DimArray` ``"col band"``); grid checks
(``check_compatible``) run between carriers of the same kind.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

from pipekit_ops import (
    NamedArrayAdapter,
    Operator,
    Signature,
    get_adapter,
    signature_of,
)

from pipekit_array.dims import (
    BackendError,
    backend_name,
    join_backends,
    namespace_of,
)


__all__ = [
    "KernelOp",
    "KernelSpec",
    "derive_geometry",
    "kernel_output_dims",
    "kernel_output_signature",
    "run_kernel",
]

_GROUP = re.compile(r"\(([^()]*)\)")
# one or more "(a,b)" groups on each side of "->"
_GROUPS = r"\([^()]*\)\s*(,\s*\([^()]*\)\s*)*"
_SIGNATURE = re.compile(rf"^\s*{_GROUPS}->\s*{_GROUPS}$")


@runtime_checkable
class _RunsKernels(Protocol):
    """An adapter with its own kernel path (an optional adapter method)."""

    def apply_kernel(
        self,
        kernel: Callable[..., Any],
        spec: KernelSpec,
        core: dict[str, str],
        kwargs: dict[str, Any],
        *carriers: Any,
    ) -> Any: ...


def _groups(part: str) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(name.strip() for name in group.split(",") if name.strip())
        for group in _GROUP.findall(part)
    )


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    """A generalized-ufunc signature over abstract core-dim letters.

    Args:
        signature: E.g. ``"(),()->()"`` (element-wise), ``"(c)->()"``
            (consumes one axis), ``"(r,c)->(c),(c,d)"`` (two outputs).
        backends: The backends the kernel is written for, from ``"numpy"``,
            ``"jax"``, ``"dask"``, ``"xarray"``.
        vectorize: The kernel handles only its core dims, and is mapped over
            the rest with ``np.vectorize`` / ``jnp.vectorize``.
        sizes: Output-only core letters and the input letter whose size each
            has, e.g. ``(("d", "c"),)``. Pairs, not a dict, so the spec stays
            hashable.
        dtypes: Each output's dtype, for output signatures and lazy
            backends: a dtype name (``"bool"``) or ``"same"`` for the first
            input's. Empty (the default) leaves it unknown in signatures;
            dask then finds it by calling the kernel once.

    Raises:
        ValueError: The signature does not parse, ``sizes`` names a letter
            that is not output-only or an input letter that does not exist,
            or ``dtypes`` is not one entry per output.

    Example:
        >>> spec = KernelSpec("(r,c)->(c),(c,d)", sizes=(("d", "c"),))
        >>> spec.inputs, spec.outputs
        ((('r', 'c'),), (('c',), ('c', 'd')))
    """

    signature: str
    backends: tuple[str, ...] = ("numpy", "jax", "dask", "xarray")
    vectorize: bool = False
    sizes: tuple[tuple[str, str], ...] = ()
    dtypes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _SIGNATURE.match(self.signature):
            raise ValueError(f"Not a gufunc signature: {self.signature!r}.")
        object.__setattr__(self, "backends", tuple(self.backends))
        object.__setattr__(self, "sizes", tuple(tuple(p) for p in self.sizes))
        object.__setattr__(self, "dtypes", tuple(map(str, self.dtypes)))
        if self.dtypes and len(self.dtypes) != len(self.outputs):
            raise ValueError(
                f"dtypes: {len(self.dtypes)} entries for {len(self.outputs)} output(s)."
            )
        in_letters = {c for group in self.inputs for c in group}
        out_only = {c for group in self.outputs for c in group} - in_letters
        declared = {out for out, _ in self.sizes}
        for out, source in self.sizes:
            if out not in out_only:
                raise ValueError(f"sizes: {out!r} is not an output-only letter.")
            if source not in in_letters:
                raise ValueError(f"sizes: {source!r} is not an input letter.")
        missing = sorted(out_only - declared)
        if missing:
            raise ValueError(
                f"Output-only core letter(s) {missing} need a size, e.g. "
                f"sizes=(({missing[0]!r}, '<input letter>'),)."
            )

    @functools.cached_property
    def inputs(self) -> tuple[tuple[str, ...], ...]:
        """Core letters of each input."""
        return _groups(self.signature.split("->")[0])

    @functools.cached_property
    def outputs(self) -> tuple[tuple[str, ...], ...]:
        """Core letters of each output."""
        return _groups(self.signature.split("->")[1])

    @functools.cached_property
    def letters(self) -> tuple[str, ...]:
        """Every core letter, in order of first appearance."""
        seen: dict[str, None] = {}
        for group in (*self.inputs, *self.outputs):
            for letter in group:
                seen.setdefault(letter, None)
        return tuple(seen)


def kernel_output_dims(
    spec: KernelSpec, core: dict[str, str], input_dims: Sequence[tuple[str, ...]]
) -> tuple[tuple[str, ...], ...]:
    """The dims of each output, from the inputs' dims, without running.

    Loop dims are every input's dims minus its core dims, in order of first
    appearance. Output k has the loop dims plus its core dims, in the first
    input's order, with dims new to the output last (F2).
    """
    loop: list[str] = []
    for dims, group in zip(input_dims, spec.inputs, strict=True):
        cores = {core[c] for c in group}
        for d in dims:
            if d not in cores and d not in loop:
                loop.append(d)
    _check_core_not_loop(loop, [tuple(core[c] for c in g) for g in spec.inputs])
    first = tuple(input_dims[0]) if input_dims else ()
    outputs = []
    for group in spec.outputs:
        dims = (*loop, *(core[c] for c in group))
        ordered = [d for d in first if d in dims] + [d for d in dims if d not in first]
        outputs.append(tuple(ordered))
    return tuple(outputs)


def kernel_output_signature(
    spec: KernelSpec, core: dict[str, str], input_signature: Any
) -> Any:
    """`KernelOp.compute_output_signature` for any operator running ``spec``.

    Shared with ``pipekit_jax.JaxKernelOp``, so a converted operator reports
    the same dims, sizes, dtypes and output count.
    """
    sig = signature_of(input_signature)
    if not isinstance(sig, Signature) or len(spec.inputs) != 1:
        return None
    sizes = dict(sig.dims)
    names = tuple(sizes)
    letter_size = {}
    for letter, dim in core.items():
        if dim in sizes:
            letter_size[letter] = sizes[dim]
    for out, source in spec.sizes:
        letter_size[out] = letter_size.get(source)
    dim_size = {core[k]: v for k, v in letter_size.items()} | {
        d: s for d, s in sizes.items()
    }
    dtypes = [sig.dtype if d == "same" else d for d in spec.dtypes] or [None] * len(
        spec.outputs
    )
    outs = tuple(
        Signature(dims=tuple((d, dim_size.get(d)) for d in dims), dtype=dtype)
        for dims, dtype in zip(
            kernel_output_dims(spec, core, [names]), dtypes, strict=True
        )
    )
    return outs if len(outs) > 1 else outs[0]


def _check_core_not_loop(
    loop: Sequence[str], core_dims: Sequence[tuple[str, ...]]
) -> None:
    """Raise when a dim is a core dim of one input and a loop dim of another."""
    cores = {d for group in core_dims for d in group}
    both = [d for d in loop if d in cores]
    if both:
        raise ValueError(
            f"dim(s) {both} are core dims of one input but loop dims of another: "
            "every input carrying a core dim must take it as a core dim (give the "
            "other input its own letter in the spec)."
        )


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
