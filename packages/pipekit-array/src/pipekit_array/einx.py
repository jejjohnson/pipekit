"""A named-dims einx bridge on any carrier (geostack OP-O2).

`einx <https://github.com/fferflo/einx>`_ writes tensor algebra as patterns
(``"band y x -> y x band"``). Here the pattern's axis names *are* the
carrier's dim names: each input is matched to its slot by name, einx runs
on the data's own backend (NumPy, JAX under ``jit``), and the result is
rewrapped through the first input's adapter with the output slot's dims, so
it keeps the labels, coordinates and (for geoarray) the grid of the dims
that survive:

    rearrange("band y x -> y x band", scene)            # transpose by name
    reduce("time band y x -> band y x", cube, op="mean") # over time
    einsum("time band, band k -> time k", cube, w)       # contract band
    repeat("y x -> member y x", field, member=8)         # broadcast

``einsum``, ``reduce`` and ``repeat`` match input slots to dims by name, in
any order. ``rearrange`` also splits and merges dims (``"(a b)"``), so its
input slot is read positionally: it lists the input's dims in order, a
group standing for one dim that is split. A merged output group
``(a b)`` becomes one dim named ``a_b``.

dask is not an einx backend: a dask-backed input raises
`pipekit.BackendError` naming ``.compute()`` rather than computing it
silently. einx is imported lazily: install the ``[einx]`` extra.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from pipekit import Operator, backend_of
from pipekit.adapters import BackendError

from pipekit_array.dims import adapter_of, transpose


__all__ = ["Einx", "PatternError", "einsum", "rearrange", "reduce", "repeat"]

Element = str | tuple[str, ...]
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_REDUCERS = ("sum", "mean", "min", "max", "prod", "std", "var", "any", "all")


class PatternError(ValueError):
    """A pattern this bridge cannot read, or one that does not fit its inputs."""


@dataclasses.dataclass(frozen=True)
class _Pattern:
    inputs: tuple[tuple[Element, ...], ...]
    output: tuple[Element, ...]

    @property
    def output_dims(self) -> tuple[str, ...]:
        return tuple(_name(el) for el in self.output)


def _name(element: Element) -> str:
    return element if isinstance(element, str) else "_".join(element)


def _parse(pattern: str) -> _Pattern:
    if pattern.count("->") != 1:
        raise PatternError(f"a pattern needs exactly one '->'; got {pattern!r}")
    lhs, rhs = pattern.split("->")
    return _Pattern(tuple(_slot(s) for s in lhs.split(",")), _slot(rhs))


def _slot(slot: str) -> tuple[Element, ...]:
    if "[" in slot or "..." in slot:
        raise PatternError(
            f"'[...]' and '...' are not supported; name every dim: {slot!r}"
        )
    elements: list[Element] = []
    i = 0
    while i < len(slot):
        ch = slot[i]
        if ch.isspace():
            i += 1
        elif ch == "(":
            close = slot.find(")", i)
            inner = slot[i + 1 : close] if close != -1 else ""
            if close == -1 or "(" in inner:
                raise PatternError(f"unbalanced or nested '(' in {slot!r}")
            members = tuple(_NAME.findall(inner))
            if not members:
                raise PatternError(f"an empty group in {slot!r}")
            elements.append(members)
            i = close + 1
        else:
            match = _NAME.match(slot, i)
            if match is None:
                raise PatternError(f"unexpected {ch!r} in {slot!r}")
            elements.append(match.group())
            i = match.end()
    return tuple(elements)


def _by_name(xs: Sequence[Any], parsed: _Pattern, what: str) -> list[Any]:
    """Each input transposed to its slot's order, its dims checked by name."""
    if len(xs) != len(parsed.inputs):
        raise PatternError(
            f"{what}: the pattern has {len(parsed.inputs)} input(s), got {len(xs)}"
        )
    out = []
    for i, (x, slot) in enumerate(zip(xs, parsed.inputs, strict=True)):
        if any(not isinstance(el, str) for el in slot):
            raise PatternError(
                f"{what}: input slots are dim names; use rearrange to split or merge"
            )
        dims = tuple(adapter_of(x, what).dims(x))
        if set(slot) != set(dims) or len(slot) != len(dims):
            raise PatternError(
                f"{what}: input {i} has dims {dims}, the pattern names {slot}"
            )
        out.append(transpose(x, tuple(str(el) for el in slot)))
    return out


def _datas(xs: Sequence[Any], what: str) -> list[Any]:
    adapters = [adapter_of(x, what) for x in xs]
    if len(xs) > 1 and len({type(a) for a in adapters}) == 1:
        xs = list(adapters[0].check_compatible(*xs))
    datas = [a.data(x) for a, x in zip(adapters, xs, strict=True)]
    backends = {backend_of(d) for d in datas}
    if "dask" in backends:
        raise BackendError(
            f"{what}: einx has no dask backend; call .compute() (or .load()) on "
            "the input first."
        )
    if len(backends) > 1:
        raise BackendError(f"{what}: inputs on {sorted(b or '?' for b in backends)}")
    return datas


def _einx() -> Any:
    """The einx module, imported on first use (the ``[einx]`` extra).

    Typed as ``Any``: einx's own ``backend=`` keyword would otherwise clash,
    for the type checker, with the axis sizes passed as ``**sizes``.
    """
    import einx

    return einx


def _wrap(template: Any, data: Any, parsed: _Pattern) -> Any:
    return adapter_of(template, "einx").rewrap(template, data, parsed.output_dims)


def rearrange(pattern: str, x: Any, **sizes: int) -> Any:
    """Transpose, split or merge dims by name.

    Args:
        pattern: One input slot and one output slot. The input slot lists
            ``x``'s dims in order; ``(a b)`` stands for one dim split into
            ``a`` and ``b`` (give their sizes in ``sizes``).
        x: A named carrier.
        **sizes: Sizes of split dims einx cannot infer (``a=2``).

    Returns:
        A carrier with the output slot's dims (a merged ``(a b)`` is
        ``a_b``), rewrapped by ``x``'s adapter.

    Raises:
        PatternError: The input slot does not describe ``x``'s dims.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> scene = DimArray(np.zeros((3, 4, 5)), ("band", "y", "x"))
        >>> rearrange("band y x -> y x band", scene).dims
        ('y', 'x', 'band')
    """
    einx = _einx()
    parsed = _parse(pattern)
    if len(parsed.inputs) != 1:
        raise PatternError("rearrange takes one input slot.")
    dims = tuple(adapter_of(x, "rearrange").dims(x))
    slot = parsed.inputs[0]
    names = [el for el in slot if isinstance(el, str)]
    if len(slot) != len(dims) or any(
        isinstance(el, str) and el != d for el, d in zip(slot, dims, strict=True)
    ):
        if not any(isinstance(el, tuple) for el in slot) and set(names) == set(dims):
            x = transpose(x, tuple(names))  # a flat slot in another order: by name
        else:
            raise PatternError(
                f"rearrange: the input slot {slot} does not describe dims {dims} "
                "in order"
            )
    (data,) = _datas([x], "rearrange")
    return _wrap(x, einx.id(pattern, data, **sizes), parsed)


def reduce(pattern: str, x: Any, *, op: str | Callable[..., Any], **sizes: int) -> Any:
    """Reduce over the dims on the input slot that the output slot leaves out.

    Args:
        pattern: ``"time band y x -> band y x"``.
        x: A named carrier.
        op: ``"sum"``, ``"mean"``, ``"min"``, ``"max"``, ``"prod"``,
            ``"std"``, ``"var"``, ``"any"``, ``"all"``, or a callable taking
            ``axis=`` (applied on the data's backend).
        **sizes: Forwarded to einx.

    Returns:
        A carrier with the output slot's dims.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.ones((2, 3, 4)), ("time", "y", "x"))
        >>> reduce("time y x -> y x", cube, op="sum").data[0, 0]
        np.float64(2.0)
    """
    einx = _einx()
    parsed = _parse(pattern)
    (x,) = _by_name([x], parsed, "reduce")
    (data,) = _datas([x], "reduce")
    if isinstance(op, str):
        if op not in _REDUCERS:
            raise PatternError(f"reduce: op must be one of {_REDUCERS} or a callable")
        result = getattr(einx, op)(pattern, data, **sizes)
    else:
        result = _backend(data).adapt_numpylike_reduce(op)(pattern, data, **sizes)
    return _wrap(x, result, parsed)


def _backend(data: Any) -> Any:
    """einx's adapter module for the data's framework (callable reductions)."""
    if backend_of(data) == "jax":
        import einx.jax as backend
    else:
        import einx.numpy as backend
    return backend


def einsum(pattern: str, *xs: Any, **sizes: int) -> Any:
    """Contract the dims named on the input slots and left off the output.

    Args:
        pattern: ``"time band, band k -> time k"``.
        *xs: One named carrier per input slot; carriers of one family are
            checked to combine first (geoarray: the same grid).
        **sizes: Forwarded to einx.

    Returns:
        A carrier with the output slot's dims, rewrapped by the first
        input's adapter.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> a = DimArray(np.ones((2, 3)), ("time", "band"))
        >>> w = DimArray(np.ones((3, 4)), ("band", "k"))
        >>> einsum("time band, band k -> time k", a, w).shape
        (2, 4)
    """
    einx = _einx()
    parsed = _parse(pattern)
    prepared = _by_name(xs, parsed, "einsum")
    datas = _datas(prepared, "einsum")
    return _wrap(prepared[0], einx.dot(pattern, *datas, **sizes), parsed)


def repeat(pattern: str, x: Any, **sizes: int) -> Any:
    """Broadcast ``x`` along new dims (sizes in ``sizes``).

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> field = DimArray(np.zeros((4, 5)), ("y", "x"))
        >>> repeat("y x -> member y x", field, member=8).shape
        (8, 4, 5)
    """
    einx = _einx()
    parsed = _parse(pattern)
    (x,) = _by_name([x], parsed, "repeat")
    (data,) = _datas([x], "repeat")
    return _wrap(x, einx.id(pattern, data, **sizes), parsed)


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "rearrange": rearrange,
    "reduce": reduce,
    "einsum": einsum,
    "repeat": repeat,
}


class Einx(Operator):
    """One einx operation as an operator: ``Einx("reduce", pattern, op="mean")``.

    Args:
        operation: ``"rearrange"``, ``"reduce"``, ``"einsum"`` or ``"repeat"``.
        pattern: The pattern, with dim names as axis names.
        op: The reduction, for ``"reduce"`` (a name, so it serialises).
        sizes: Axis sizes forwarded to einx.

    Example:
        >>> import numpy as np
        >>> from pipekit_array import DimArray
        >>> cube = DimArray(np.ones((2, 3, 4)), ("time", "y", "x"))
        >>> Einx("reduce", "time y x -> y x", op="mean")(cube).dims
        ('y', 'x')
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax")

    def __init__(
        self,
        operation: str,
        pattern: str,
        *,
        op: str | None = None,
        sizes: dict[str, int] | None = None,
    ) -> None:
        if operation not in _FUNCTIONS:
            raise ValueError(f"operation must be one of {tuple(_FUNCTIONS)}")
        if (operation == "reduce") != (op is not None):
            raise ValueError("op is given exactly when operation is 'reduce'")
        _parse(pattern)  # fail at construction, not at the first call
        self.operation = operation
        self.pattern = pattern
        self.op = op
        self.sizes = dict(sizes or {})

    def _apply(self, *xs: Any) -> Any:
        fn = _FUNCTIONS[self.operation]
        if self.operation == "reduce":
            return fn(self.pattern, *xs, op=self.op, **self.sizes)
        return fn(self.pattern, *xs, **self.sizes)
