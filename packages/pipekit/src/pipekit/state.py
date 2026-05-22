"""Group M — state primitives.

Two primitives that enable stateful pipelines — the substrate for the
`pipekit-cycle` sister package (time-stepping, DA cycles, iterative
inference) and for the `pipekit-train` stateful trainer loop:

- `StatefulOperator` — subclass marker. ``_apply`` has the signature
  ``(carrier, state) -> (carrier, state)`` instead of the stateless
  ``carrier -> carrier``.
- `CarryState` — base class for state threaded through a stateful
  pipeline. Subclasses define their own fields; the base provides
  JSON-roundtrip discipline (``to_dict`` / ``from_dict``).

`Sequential` detects when any of its operators is a
`StatefulOperator` and threads state through automatically. Stateless
operators in the chain pass state through unchanged. A `Sequential`
containing only stateless operators behaves exactly as it did before
this group landed.

Why this lives in pipekit core (not in `pipekit-cycle`):

1. `Sequential` is defined here; its stateful-detection logic must be
   here too. Putting `StatefulOperator` in a sibling package would
   invert the dependency.
2. Other sibling packages may want state — `pipekit-train`'s
   `TrainingLoop` is itself stateful (optimizer / step / metrics
   threaded through epochs). Sharing the base class lets them all
   reuse the same plumbing.

See master plan Report 2, Group M.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pipekit._base.operator import Operator


class StatefulOperator(Operator):
    """Marker base for operators whose ``_apply`` threads carry-state.

    Subclasses implement ``_apply(carrier, state) -> (carrier, state)``
    rather than the stateless ``(carrier,) -> carrier``. The
    ``_is_stateful = True`` ClassVar flags the class so `Sequential`
    detects it and threads state through automatically.

    Subclasses may set ``initial_state_fn`` to a callable
    ``() -> CarryState`` that produces a fresh initial state. This is
    used by `pipekit-cycle`'s cycle wrappers to bootstrap iteration
    without forcing the caller to construct state explicitly.

    Example::

        class StreamMean(StatefulOperator):
            def __init__(self):
                self.initial_state_fn = lambda: MeanState(sum=0.0, n=0)

            def _apply(self, x, state):
                new = MeanState(sum=state.sum + x, n=state.n + 1)
                return new.sum / new.n, new

    See `pipekit-cycle` (Report 10) for the cycle / DA wrappers built
    on this primitive.
    """

    _is_stateful: ClassVar[bool] = True
    initial_state_fn: Callable[[], CarryState] | None = None

    def _apply(self, carrier: Any, state: Any) -> tuple[Any, Any]:  # type: ignore[override]
        raise NotImplementedError(
            f"{type(self).__name__} must implement _apply(carrier, state)."
        )


class CarryState:
    """Base class for state threaded through a stateful pipeline.

    Concrete subclasses define their own fields — the base class
    intentionally does not dictate a schema. Different cycle patterns
    bring their own ``CarryState`` shape (`pipekit-cycle.DAState`,
    `IterationState`, `WindowState`, ...).

    The base supplies JSON-roundtrip discipline:

    - `to_dict` returns the instance's ``__dict__`` (so any attributes
      set in ``__init__`` are captured).
    - `from_dict` reconstructs via ``cls(**d)``. Subclasses with
      non-trivial deserialisation override.

    Subclassing pattern::

        class IterationState(CarryState):
            def __init__(self, residual_norm: float, iter_count: int) -> None:
                self.residual_norm = residual_norm
                self.iter_count = iter_count

        state = IterationState(residual_norm=1.0, iter_count=0)
        state.to_dict()                  # {"residual_norm": 1.0, "iter_count": 0}
        IterationState.from_dict(...)    # rebuilds
    """

    def to_dict(self) -> dict[str, Any]:
        """Return a dict of the instance's attributes."""
        return dict(vars(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CarryState:
        """Reconstruct from ``to_dict`` output. Subclasses may override."""
        return cls(**data)

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({fields})"

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return vars(self) == vars(other)

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(vars(self).items()))))
