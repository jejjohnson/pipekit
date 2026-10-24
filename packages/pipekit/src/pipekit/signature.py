"""Group I — shape inference.

A tiny value class plus a per-operator protocol method (already declared
on `Operator`) that together let `Sequential.summary` print a
Keras-style structural table without executing any data.

- `Signature(dims, dtype)` — immutable record of named dimensions and
  dtype. Helpers: `format`, `drop_dims`, `replace_dims`; `matches` tests
  it against a dims pattern (``"band y x"``, ``"... y x"``) and
  `from_pattern` builds a sizeless one from a pattern. Patterns are the
  one dims vocabulary of the stack: readable types, ``check_dims`` and
  kernel output signatures are written in them.
- `Operator.compute_output_signature(input_sig) -> Signature | None`
  — per-operator method, default passthrough. Operators that reshape /
  reduce / retype override; carriers without named dimensions return
  ``None`` to opt out.

`Signature` is xarray-flavoured (named dims). Carriers without named
dims (raw numpy, `GeoTensor`) opt out by returning ``None`` from
`compute_output_signature`; `Sequential.summary` reports ``?`` for
those steps rather than crashing.

See master plan Report 2, Group I.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Signature:
    """Immutable shape + dtype record.

    Attributes:
        dims: Tuple of ``(name, size)`` pairs. ``size`` may be ``None``
            to mark a dim as dynamic / unknown. Order matters — it's
            the carrier's axis order.
        dtype: A dtype label (e.g. ``"float32"``, ``"int64"``) or
            ``None`` if the operator doesn't track dtype.

    Example::

        sig = Signature(dims=(("time", 12), ("y", 256), ("x", 256)), dtype="float32")
        sig.format()                       # "(time=12, y=256, x=256) float32"
        sig.drop_dims(("time",))           # drops the time axis
        sig.replace_dims({"x": 128})       # change the x size
    """

    dims: tuple[tuple[str, int | None], ...]
    dtype: str | None = None

    def __post_init__(self) -> None:
        names = [d[0] for d in self.dims]
        if len(set(names)) != len(names):
            raise ValueError(f"Signature dims must have unique names, got {names!r}.")

    def format(self) -> str:
        """Return a compact ``"(name=size, …) dtype"`` string."""
        parts = ", ".join(
            f"{name}={'?' if size is None else size}" for name, size in self.dims
        )
        dtype = self.dtype if self.dtype is not None else "?"
        return f"({parts}) {dtype}"

    def drop_dims(self, names: tuple[str, ...]) -> Signature:
        """Return a new `Signature` with the given dim names removed.

        Raises:
            KeyError: if a name in ``names`` isn't present.
        """
        present = {n for n, _ in self.dims}
        missing = set(names) - present
        if missing:
            raise KeyError(f"drop_dims: unknown dim(s) {sorted(missing)}.")
        kept = tuple((n, s) for n, s in self.dims if n not in set(names))
        return Signature(dims=kept, dtype=self.dtype)

    def replace_dims(self, sizes: dict[str, int | None]) -> Signature:
        """Return a new `Signature` with the listed dim sizes overridden.

        Raises:
            KeyError: if a name in ``sizes`` isn't present.
        """
        present = {n for n, _ in self.dims}
        missing = set(sizes) - present
        if missing:
            raise KeyError(f"replace_dims: unknown dim(s) {sorted(missing)}.")
        new = tuple((n, sizes.get(n, s)) for n, s in self.dims)
        return Signature(dims=new, dtype=self.dtype)

    def matches(self, pattern: str) -> bool:
        """Whether these dims fit a dims pattern; sizes are ignored.

        A pattern is whitespace-separated dim names with at most one
        ``...``, which matches zero or more dims at its position. Names
        compare exactly and in order.

        Args:
            pattern: E.g. ``"band y x"`` (exactly these dims) or
                ``"... y x"`` (any leading dims, then ``y`` and ``x``).

        Raises:
            ValueError: if the pattern has more than one ``...``.

        Examples:
            >>> sig = Signature(dims=(("time", 12), ("band", 2), ("y", 4), ("x", 4)))
            >>> sig.matches("time band y x"), sig.matches("... y x")
            (True, True)
            >>> sig.matches("band y x"), sig.matches("time ... x")
            (False, True)
        """
        names, ellipsis = _parse_pattern(pattern)
        dims = tuple(name for name, _ in self.dims)
        if ellipsis is None:
            return dims == tuple(names)
        head = tuple(names[:ellipsis])
        tail = tuple(names[ellipsis:])
        if len(dims) < len(head) + len(tail):
            return False
        return dims[: len(head)] == head and dims[len(dims) - len(tail) :] == tail

    @classmethod
    def from_pattern(cls, pattern: str, dtype: str | None = None) -> Signature:
        """A sizeless `Signature` from a dims pattern, for annotations.

        Args:
            pattern: Whitespace-separated dim names, e.g. ``"band y x"``.
            dtype: An optional dtype label.

        Raises:
            ValueError: if the pattern has a ``...`` (a signature needs
                concrete names) or repeats a name.

        Example:
            >>> Signature.from_pattern("band y x").dims
            (('band', None), ('y', None), ('x', None))
        """
        names, ellipsis = _parse_pattern(pattern)
        if ellipsis is not None:
            raise ValueError(
                f"from_pattern needs concrete dim names, got {pattern!r} with '...'."
            )
        return cls(dims=tuple((name, None) for name in names), dtype=dtype)

    def __repr__(self) -> str:
        return f"Signature({self.format()})"


def _parse_pattern(pattern: str) -> tuple[list[str], int | None]:
    """Names in a dims pattern, and the position of its ``...`` (or None)."""
    tokens = pattern.split()
    names: list[str] = []
    ellipsis: int | None = None
    for token in tokens:
        if token == "...":
            if ellipsis is not None:
                raise ValueError(f"A dims pattern has at most one '...': {pattern!r}.")
            ellipsis = len(names)
        else:
            names.append(token)
    return names, ellipsis


def compute_output_signature(
    op: Any, input_signature: Signature | None
) -> Signature | None:
    """Free-function dispatch to ``op.compute_output_signature``.

    Convenience for code paths that don't statically know they're
    holding an `Operator`. Returns ``None`` when the operator opts out
    of shape inference.
    """
    fn = getattr(op, "compute_output_signature", None)
    if fn is None:
        return None
    return fn(input_signature)


def common_signature(*signatures: Signature | None) -> Signature | None:
    """The signature every branch agrees on, or ``None`` if any differ.

    For operators that may run one of several children (`Try`,
    `Branch`, `Switch`, `Coalesce`): the output's signature is known
    only when every child would give the same one.

    Examples:
        >>> s = Signature(dims=(("x", 4),), dtype="float32")
        >>> common_signature(s, s) == s
        True
        >>> common_signature(s, None) is None
        True
    """
    if not signatures or signatures[0] is None:
        return None
    first = signatures[0]
    for signature in signatures[1:]:
        if signature != first:
            return None
    return first
