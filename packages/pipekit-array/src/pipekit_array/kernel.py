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
import re
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

from pipekit import (
    NamedArrayAdapter,
    Operator,
    Signature,
    get_adapter,
    signature_of,
)

from pipekit_array.dims import namespace_of


__all__ = [
    "BackendError",
    "KernelOp",
    "KernelSpec",
    "derive_geometry",
    "kernel_output_dims",
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


class BackendError(TypeError):
    """Inputs on different backends, or a backend the kernel does not support."""


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

    Raises:
        ValueError: The signature does not parse, or ``sizes`` names a letter
            that is not output-only or an input letter that does not exist.

    Example:
        >>> spec = KernelSpec("(r,c)->(c),(c,d)", sizes=(("d", "c"),))
        >>> spec.inputs, spec.outputs
        ((('r', 'c'),), (('c',), ('c', 'd')))
    """

    signature: str
    backends: tuple[str, ...] = ("numpy", "jax", "dask", "xarray")
    vectorize: bool = False
    sizes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not _SIGNATURE.match(self.signature):
            raise ValueError(f"Not a gufunc signature: {self.signature!r}.")
        object.__setattr__(self, "backends", tuple(self.backends))
        object.__setattr__(self, "sizes", tuple(tuple(p) for p in self.sizes))
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
    first = tuple(input_dims[0]) if input_dims else ()
    outputs = []
    for group in spec.outputs:
        dims = (*loop, *(core[c] for c in group))
        ordered = [d for d in first if d in dims] + [d for d in dims if d not in first]
        outputs.append(tuple(ordered))
    return tuple(outputs)


def _backend(data: Any) -> str:
    module = type(data).__module__
    if module.startswith(("jax", "jaxlib")) or "Tracer" in type(data).__name__:
        return "jax"
    if module.startswith("dask."):
        return "dask"
    return "numpy"


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
        BackendError: Carriers on more than one backend, or a backend the
            spec does not list.
        TypeError: A raw array next to named carriers.
        ValueError: A core dim is missing from its input, or a dim has two
            sizes.
    """
    adapters = [get_adapter(x) for x in carriers]
    if all(a is None for a in adapters):
        backend = _backend(carriers[0]) if carriers else "numpy"
        return _call(kernel, spec, kwargs, backend)(*carriers)
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
    return _Plan(tuple(arrange), size_checks, tuple(restore))


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
    datas = [a.data(x) for a, x in zip(adapters, carriers, strict=True)]
    backends = {_backend(d) for d in datas}
    if len(backends) > 1:
        raise BackendError(
            f"inputs on {sorted(backends)}: call to_jax() on the NumPy carriers "
            "first (inputs never move between backends silently)."
        )
    backend = backends.pop()
    if backend not in spec.backends:
        raise BackendError(
            f"the kernel declares backends {spec.backends}, not {backend!r}."
        )
    if len(carriers) > 1:
        carriers = _check_by_kind(carriers, adapters)
        datas = [a.data(x) for a, x in zip(adapters, carriers, strict=True)]
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
    template, template_adapter = carriers[0], adapters[0]
    wrapped = []
    for out, (axes, target) in zip(results, plan.restore, strict=True):
        moved_out = _permute(xp, out, axes)
        wrapped.append(template_adapter.rewrap(template, moved_out, target))
    return tuple(wrapped) if many else wrapped[0]


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
        **params: Keyword parameters passed to every call.

    Raises:
        ValueError: ``core`` misses or invents a letter of the spec.

    ``KernelOp`` holds a user function, so its config is a debug repr only
    (``forbid_in_yaml``, like `pipekit.Lambda`); a named operator built on
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
        core: dict[str, str],
        geometry: dict[str, Callable[[Any], Any]] | None = None,
        **params: Any,
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
        self.geometry = dict(geometry or {})
        self.params = params

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
        The dtype is the input's: kernels preserve their input dtype (the
        caller decides precision).
        """
        sig = signature_of(input_signature)
        if not isinstance(sig, Signature) or len(self.spec.inputs) != 1:
            return None
        sizes = dict(sig.dims)
        names = tuple(sizes)
        letter_size = {}
        for letter, dim in self.core.items():
            if dim in sizes:
                letter_size[letter] = sizes[dim]
        for out, source in self.spec.sizes:
            letter_size[out] = letter_size.get(source)
        dim_size = {self.core[k]: v for k, v in letter_size.items()} | {
            d: s for d, s in sizes.items()
        }
        outs = tuple(
            Signature(dims=tuple((d, dim_size.get(d)) for d in dims), dtype=sig.dtype)
            for dims in kernel_output_dims(self.spec, self.core, [names])
        )
        return outs if len(outs) > 1 else outs[0]
