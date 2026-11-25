"""What a kernel consumes and produces: a generalized-ufunc signature."""

from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from pipekit_ops import Signature, signature_of


__all__ = ["KernelSpec", "kernel_output_dims", "kernel_output_signature"]


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
