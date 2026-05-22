"""Group I — shape inference.

A tiny value class plus a per-operator protocol method (already declared
on `Operator`) that together let `Sequential.summary` print a
Keras-style structural table without executing any data.

- `Signature(dims, dtype)` — immutable record of named dimensions and
  dtype. Helpers: `format`, `drop_dims`, `replace_dims`.
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

    def __repr__(self) -> str:
        return f"Signature({self.format()})"


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
