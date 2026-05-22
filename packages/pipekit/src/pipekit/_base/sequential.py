"""Sequential — eager linear composition of operators.

`Sequential` is the workhorse pipekit pipeline: a list of operators
applied left-to-right, each consuming the previous one's output.

    pipe = Sequential([Scale(0.5), Scale(2.0)])
    result = pipe(10.0)

Equivalent and idiomatic via the ``|`` operator on `Operator`::

    pipe = Scale(0.5) | Scale(2.0)

`Sequential` is itself an `Operator`, so it composes recursively. The
right-fold `__or__` keeps nested `Sequential`s flat.

Construction-time invariants:

1. **No terminal operators in non-terminal position.** Subclasses
   marked ``_terminal = True`` (write-to-disk, viz, etc.) can only
   appear as the last step. Catching this at construction gives a
   clear error rather than a confusing runtime failure.
2. **All steps are `Operator` instances.** Same logic — clearer error
   than an ``AttributeError`` deep in the chain.

See master plan Report 2, Group A.2.
"""

from __future__ import annotations

from typing import Any

from pipekit._base.operator import Carrier, Operator


_MISSING = object()


class Sequential(Operator):
    """Apply a list of operators in order, threading the output of each into the next.

    Args:
        operators: A list of `Operator` instances. Empty list is allowed;
            calling an empty `Sequential` requires no input and raises
            cleanly if one is passed (the pipeline has no operations to
            perform).

    Raises:
        TypeError: if any element is not an `Operator`, or if any
            non-final element is marked ``_terminal``.

    Example:
        Basic::

            pipe = Sequential([Scale(0.5), Scale(2.0)])
            assert pipe(10.0) == 10.0

        Slice into a sub-pipeline::

            head = pipe[:1]   # Sequential([Scale(0.5)])

        Index a single step::

            first = pipe[0]   # Scale(0.5)
    """

    __config_mixin_auto__ = False  # custom get_config — nested-operator list

    def __init__(self, operators: list[Operator]) -> None:
        for i, op in enumerate(operators):
            if not isinstance(op, Operator):
                raise TypeError(
                    f"Sequential[{i}] is {type(op).__name__}, expected Operator."
                )
        for i, op in enumerate(operators[:-1]):
            if op._terminal:
                raise TypeError(
                    f"Sequential[{i}] is a terminal operator "
                    f"({type(op).__name__}) — terminal operators are only "
                    "valid as the last step of a Sequential."
                )
        self.operators = list(operators)

    @property
    def _is_stateful(self) -> bool:  # type: ignore[override]
        """True if any contained operator is a `StatefulOperator`.

        Overrides the ``_is_stateful: ClassVar[bool] = False`` default
        on `Operator` with a per-instance property — the answer depends
        on what the user composed.
        """
        return any(getattr(op, "_is_stateful", False) for op in self.operators)

    def _apply(self, carrier: Carrier = _MISSING, state: Any = _MISSING) -> Any:
        if carrier is _MISSING and not self.operators:
            raise TypeError(
                "Sequential([]) cannot be called without an input: "
                "empty pipeline has no operations to perform."
            )
        # Stateful path: any stateful op present *or* state explicitly supplied.
        if state is not _MISSING or self._is_stateful:
            if state is _MISSING:
                raise TypeError(
                    "Sequential contains StatefulOperator(s) but was called "
                    "without a `state` argument. Supply the initial CarryState."
                )
            return self._apply_stateful(carrier, state)
        # Stateless path — original behaviour.
        if carrier is _MISSING:
            out = self.operators[0]()
            remaining = self.operators[1:]
        else:
            out = carrier
            remaining = self.operators
        for op in remaining:
            out = op(out)
        return out

    def _apply_stateful(self, carrier: Carrier, state: Any) -> tuple[Any, Any]:
        """Thread state through stateful ops; pass through stateless ones unchanged.

        Returns ``(carrier, state)`` so the caller can checkpoint or
        feed the state back into the next call.
        """
        out = carrier
        for op in self.operators:
            if getattr(op, "_is_stateful", False):
                out, state = op(out, state)
            else:
                out = op(out)
        return out, state

    def compute_output_signature(self, input_signature: Any) -> Any:
        """Thread the signature through every operator in order."""
        sig = input_signature
        for op in self.operators:
            sig = op.compute_output_signature(sig)
            if sig is None:
                return None
        return sig

    def get_config(self) -> dict[str, Any]:
        return {
            "operators": [
                {"class": type(op).__name__, "config": op.get_config()}
                for op in self.operators
            ]
        }

    def __or__(self, other: Operator) -> Sequential:
        """Append on the right; flatten nested `Sequential`s."""
        if not isinstance(other, Operator):
            return NotImplemented
        if isinstance(other, Sequential):
            return Sequential([*self.operators, *other.operators])
        return Sequential([*self.operators, other])

    def __repr__(self) -> str:
        if not self.operators:
            return "Sequential([])"
        inner = ", ".join(repr(op) for op in self.operators)
        return f"Sequential([{inner}])"

    def __len__(self) -> int:
        return len(self.operators)

    def __getitem__(self, key: int | slice) -> Operator | Sequential:
        """Index or slice into the underlying operator list.

        ``pipe[0]`` returns the first operator. ``pipe[1:3]`` returns a
        new `Sequential` containing the slice.
        """
        if isinstance(key, slice):
            return Sequential(self.operators[key])
        return self.operators[key]

    def describe(self, indent: int = 0) -> str:
        """Return an indented-tree pretty-print of the pipeline.

        Recurses into nested `Sequential` / `Graph` operators. Useful
        for notebook exploration; not parsed.
        """
        pad = "  " * indent
        if not self.operators:
            return f"{pad}Sequential([])"
        lines: list[str] = [f"{pad}Sequential(["]
        for op in self.operators:
            if isinstance(op, Sequential):
                lines.append(op.describe(indent=indent + 1) + ",")
            else:
                lines.append(f"{pad}  {op!r},")
        lines.append(f"{pad}])")
        return "\n".join(lines)

    def summary(self, input_signature: Any = None) -> str:
        """Render a Keras-style summary table.

        Threads ``input_signature`` through `compute_output_signature`.
        If any operator returns ``None`` (does not track shape), the
        summary still lists the operator but shows ``?`` for its output
        signature. With ``input_signature=None``, only operator names
        and types are shown.

        The return type is ``str`` rather than printing directly so the
        caller decides where it goes (notebook, log, file).
        """
        rows: list[tuple[str, str, str]] = []
        sig = input_signature
        for i, op in enumerate(self.operators):
            sig_in = "?" if sig is None else repr(sig)
            sig = op.compute_output_signature(sig)
            sig_out = "?" if sig is None else repr(sig)
            rows.append((f"[{i}] {type(op).__name__}", sig_in, sig_out))

        # Column widths
        w_name = max((len(r[0]) for r in rows), default=4)
        w_in = max((len(r[1]) for r in rows), default=8)
        w_out = max((len(r[2]) for r in rows), default=8)

        header = (
            f"{'Operator'.ljust(w_name)}  "
            f"{'Input sig'.ljust(w_in)}  "
            f"{'Output sig'.ljust(w_out)}"
        )
        rule = "-" * len(header)
        body = [
            f"{name.ljust(w_name)}  {sin.ljust(w_in)}  {sout.ljust(w_out)}"
            for name, sin, sout in rows
        ]
        return "\n".join([header, rule, *body])
