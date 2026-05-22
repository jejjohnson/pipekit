"""Generic time-stepping and iteration wrappers.

These are the cycle abstractions — `Cycle`, `EnsembleCycle`,
`WindowedCycle`, `Recurrence` — that lift a per-step
`StatefulOperator` (or a `ForwardModel`) into a multi-step operator.
Each is itself a `StatefulOperator`, so cycles compose into longer
cycles and into `Sequential` pipelines.

Design choice: each cycle returns ``(carrier, state)`` to preserve
the `StatefulOperator` contract. When ``save_history=True`` the
intermediate carriers are accumulated on ``self.history`` — a plain
list attribute available after the call. Treat the attribute as
write-once-per-call: re-running the operator overwrites it.

See master plan Report 10, section 2.2.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pipekit import Operator, StatefulOperator

from pipekit_cycle.protocols import ForwardModel


class _ForwardModelAdapter(StatefulOperator):
    """Wrap a `ForwardModel` instance as a `StatefulOperator`.

    Internal — `Cycle.__init__` uses this transparently when ``step_op``
    satisfies `ForwardModel` but isn't already a `StatefulOperator`.
    The state is passed through unchanged (the model is stateless
    w.r.t. carry-state; it only knows about the carrier).
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(self, model: ForwardModel) -> None:
        self.model = model

    def _apply(self, carrier: Any, state: Any) -> tuple[Any, Any]:
        return self.model.step(carrier, self.model.dt), state

    def get_config(self) -> dict[str, Any]:
        return {"model_class": type(self.model).__name__}


def _coerce_step(step_op: Any) -> StatefulOperator:
    """Return ``step_op`` as a `StatefulOperator`.

    Accepts:
      - A `StatefulOperator` instance — returned as-is.
      - A `ForwardModel` (anything satisfying the Protocol that isn't
        already a `StatefulOperator`) — wrapped in `_ForwardModelAdapter`.

    Raises:
        TypeError: if neither shape matches.
    """
    if isinstance(step_op, StatefulOperator):
        return step_op
    if isinstance(step_op, ForwardModel):
        return _ForwardModelAdapter(step_op)
    raise TypeError(
        "step_op must be a StatefulOperator or a ForwardModel, "
        f"got {type(step_op).__name__}."
    )


class Cycle(StatefulOperator):
    """Apply ``step_op`` for ``n_steps``, threading state through each step.

    Equivalent in shape to ``jax.lax.scan`` but carrier-agnostic.

    Args:
        step_op: A `StatefulOperator` or a `ForwardModel`. The latter
            is wrapped automatically so domain libraries can pass their
            forward-model adapters directly.
        n_steps: Number of steps to take. Must be ≥ 0; ``0`` is a
            no-op that returns the inputs unchanged.
        save_history: When ``True``, intermediate ``(carrier, state)``
            pairs are appended to ``self.history`` every
            ``history_stride`` steps.
        history_stride: Save every ``stride``-th step (default 1).

    Returns from ``_apply(carrier, state)``: ``(final_carrier, final_state)``.

    Example::

        cycle = Cycle(step_op=Advect(dt=3600.0), n_steps=24, save_history=True)
        carrier, state = cycle(initial_carrier, initial_state)
        # cycle.history is the per-step trajectory
    """

    __config_mixin_auto__ = False

    def __init__(
        self,
        step_op: Any,
        n_steps: int,
        save_history: bool = False,
        history_stride: int = 1,
    ) -> None:
        if n_steps < 0:
            raise ValueError(f"Cycle.n_steps must be >= 0, got {n_steps}.")
        if history_stride < 1:
            raise ValueError(
                f"Cycle.history_stride must be >= 1, got {history_stride}."
            )
        self.step_op = _coerce_step(step_op)
        self.n_steps = n_steps
        self.save_history = save_history
        self.history_stride = history_stride
        self.history: list[tuple[Any, Any]] = []

    def _apply(self, carrier: Any, state: Any) -> tuple[Any, Any]:
        self.history = []
        for i in range(self.n_steps):
            carrier, state = self.step_op(carrier, state)
            if self.save_history and (i % self.history_stride == 0):
                self.history.append((carrier, state))
        return carrier, state

    def get_config(self) -> dict[str, Any]:
        return {
            "step_op": {
                "class": type(self.step_op).__name__,
                "config": self.step_op.get_config(),
            },
            "n_steps": self.n_steps,
            "save_history": self.save_history,
            "history_stride": self.history_stride,
        }


class EnsembleCycle(StatefulOperator):
    """Run `Cycle` over a list of ensemble-member carriers.

    The carrier is expected to be a sequence (list / tuple) of
    ``n_members`` individual carriers; the same ``step_op`` is applied
    to each member in turn, sharing the carry-state across members
    (the typical EnKF arrangement, where the state holds covariances
    or inflation factors that all members read).

    Vectorisation across members is delegated to the user — for
    array-shaped carriers, wrap ``step_op`` in a batched / vmapped
    operator before passing it in.

    Args:
        step_op: Per-member step. `StatefulOperator` or `ForwardModel`.
        n_steps: Steps per member.
        n_members: Expected ensemble size; the input carrier is
            length-checked at call time.

    Returns from ``_apply(members, state)``: ``(list_of_final_members, state)``.
    """

    __config_mixin_auto__ = False

    def __init__(self, step_op: Any, n_steps: int, n_members: int) -> None:
        if n_steps < 0:
            raise ValueError(f"EnsembleCycle.n_steps must be >= 0, got {n_steps}.")
        if n_members < 1:
            raise ValueError(f"EnsembleCycle.n_members must be >= 1, got {n_members}.")
        self.step_op = _coerce_step(step_op)
        self.n_steps = n_steps
        self.n_members = n_members

    def _apply(  # ty: ignore[invalid-method-override]
        self,
        members: list[Any],
        state: Any,
    ) -> tuple[list[Any], Any]:
        if len(members) != self.n_members:
            raise ValueError(
                f"EnsembleCycle expected {self.n_members} members, got {len(members)}."
            )
        out_members = list(members)
        for _ in range(self.n_steps):
            for j, carrier in enumerate(out_members):
                out_members[j], state = self.step_op(carrier, state)
        return out_members, state

    def get_config(self) -> dict[str, Any]:
        return {
            "step_op": {
                "class": type(self.step_op).__name__,
                "config": self.step_op.get_config(),
            },
            "n_steps": self.n_steps,
            "n_members": self.n_members,
        }


class WindowedCycle(StatefulOperator):
    """Sliding-window time-stepping over a stream of initial carriers.

    The input is a list of carriers (one per stream position). For
    each window — windows start at positions ``0, stride, 2*stride, …``
    up to ``len(stream) - window`` — the cycle seeds a carrier from
    the first stream element of that window and threads it through
    ``window`` applications of ``step_op``. State threads across
    windows.

    Args:
        step_op: Per-step operator. `StatefulOperator` or `ForwardModel`.
        window: Number of ``step_op`` applications per window.
        stride: Step between successive window starts (in stream units).

    Returns from ``_apply(stream, state)``:
        ``(list_of_window_final_carriers, state)``.
    """

    __config_mixin_auto__ = False

    def __init__(self, step_op: Any, window: int, stride: int = 1) -> None:
        if window < 1:
            raise ValueError(f"WindowedCycle.window must be >= 1, got {window}.")
        if stride < 1:
            raise ValueError(f"WindowedCycle.stride must be >= 1, got {stride}.")
        self.step_op = _coerce_step(step_op)
        self.window = window
        self.stride = stride

    def _apply(  # ty: ignore[invalid-method-override]
        self,
        stream: list[Any],
        state: Any,
    ) -> tuple[list[Any], Any]:
        outputs: list[Any] = []
        n = len(stream)
        for start in range(0, n - self.window + 1, self.stride):
            carrier = stream[start]
            for _ in range(self.window):
                carrier, state = self.step_op(carrier, state)
            outputs.append(carrier)
        return outputs, state

    def get_config(self) -> dict[str, Any]:
        return {
            "step_op": {
                "class": type(self.step_op).__name__,
                "config": self.step_op.get_config(),
            },
            "window": self.window,
            "stride": self.stride,
        }


class Recurrence(StatefulOperator):
    """Apply ``step_op`` repeatedly until ``condition_op`` returns truthy.

    For iterative inference: fixed-point methods, Levenberg-Marquardt,
    EM, variational solvers. ``condition_op`` is an `Operator` that
    inspects the post-step state and returns a bool-ish value; the
    loop exits when it returns truthy or when ``max_iters`` is reached.

    Args:
        step_op: Per-iteration step.
        condition_op: `Operator` invoked as ``condition_op(state)``.
            Truthy = stop.
        max_iters: Safety cap on iterations (default 1000).

    Returns from ``_apply(carrier, state)``: ``(carrier, state)``
    after either convergence or hitting ``max_iters``.
    """

    __config_mixin_auto__ = False

    def __init__(
        self,
        step_op: Any,
        condition_op: Operator,
        max_iters: int = 1000,
    ) -> None:
        if not isinstance(condition_op, Operator):
            raise TypeError(
                "Recurrence.condition_op must be an Operator, "
                f"got {type(condition_op).__name__}."
            )
        if max_iters < 1:
            raise ValueError(f"Recurrence.max_iters must be >= 1, got {max_iters}.")
        self.step_op = _coerce_step(step_op)
        self.condition_op = condition_op
        self.max_iters = max_iters
        self.iters: int = 0
        self.converged: bool = False

    def _apply(self, carrier: Any, state: Any) -> tuple[Any, Any]:
        self.iters = 0
        self.converged = False
        for _ in range(self.max_iters):
            carrier, state = self.step_op(carrier, state)
            self.iters += 1
            if self.condition_op(state):
                self.converged = True
                break
        return carrier, state

    def get_config(self) -> dict[str, Any]:
        return {
            "step_op": {
                "class": type(self.step_op).__name__,
                "config": self.step_op.get_config(),
            },
            "condition_op": {
                "class": type(self.condition_op).__name__,
                "config": self.condition_op.get_config(),
            },
            "max_iters": self.max_iters,
        }
