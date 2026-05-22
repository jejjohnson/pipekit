"""Generic `ForwardModel` adapters.

Each class is a `pipekit.Operator` *and* satisfies the
`ForwardModel` Protocol so it can be passed wherever a protocol-typed
slot accepts forward models (e.g. `DACycle.forward_model`).

- `CallableForward(fn, dt)` — wrap any callable as a `ForwardModel`.
- `CompositeForward(components, dt)` — apply each component in turn;
  total advance is ``dt`` (each component sees its own portion).
- `NeuralForward(model_op, dt)` — wrap a neural emulator operator
  (typically a `pipekit-array.ModelOp`) as a forward model.

See master plan Report 10, section 2.6.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pipekit import Operator


class CallableForward(Operator):
    """Wrap a plain callable as a forward model.

    Args:
        fn: Callable ``(state, dt) → new_state``.
        dt: Default integration step in user units (seconds, hours, …).
        state_signature: Optional `pipekit.Signature` describing the
            state carrier. ``None`` by default.
        name: Optional name for ``__repr__``.

    Carries ``forbid_in_yaml = True`` — the callable doesn't round-trip.
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        fn: Callable[[Any, float], Any],
        dt: float,
        state_signature: Any = None,
        name: str | None = None,
    ) -> None:
        self.fn = fn
        self._dt = float(dt)
        self._state_signature = state_signature
        self.name = name or getattr(fn, "__name__", "CallableForward")

    def step(self, state: Any, dt: float) -> Any:
        return self.fn(state, dt)

    def _apply(self, state: Any) -> Any:
        """Calling the operator advances by the default ``dt``."""
        return self.step(state, self._dt)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def state_signature(self) -> Any:
        return self._state_signature

    def get_config(self) -> dict[str, Any]:
        return {"dt": self._dt, "name": self.name}

    def __repr__(self) -> str:
        return f"CallableForward(name={self.name!r}, dt={self._dt})"


class CompositeForward(Operator):
    """Compose multiple forward models — ``F_k ∘ … ∘ F_1``.

    Each component advances state by its own internal ``dt``. The
    composite's ``dt`` is the sum (the total advance per ``step`` call).
    The components run left-to-right per step.

    Args:
        components: Tuple of objects satisfying `ForwardModel`. Must
            be non-empty.

    Raises:
        ValueError: if ``components`` is empty.
    """

    __config_mixin_auto__ = False

    def __init__(self, components: tuple[Any, ...]) -> None:
        if not components:
            raise ValueError("CompositeForward requires at least one component.")
        for i, c in enumerate(components):
            if not all(hasattr(c, n) for n in ("step", "dt")):
                raise TypeError(
                    f"CompositeForward.components[{i}] does not satisfy "
                    "ForwardModel (missing `step` or `dt`)."
                )
        self.components = tuple(components)

    def step(self, state: Any, dt: float) -> Any:
        out = state
        for c in self.components:
            out = c.step(out, c.dt)
        return out

    def _apply(self, state: Any) -> Any:
        return self.step(state, self.dt)

    @property
    def dt(self) -> float:
        return float(sum(c.dt for c in self.components))

    @property
    def state_signature(self) -> Any:
        for c in self.components:
            sig = getattr(c, "state_signature", None)
            if sig is not None:
                return sig
        return None

    def get_config(self) -> dict[str, Any]:
        return {
            "components": [
                {"class": type(c).__name__, "dt": getattr(c, "dt", None)}
                for c in self.components
            ],
            "dt": self.dt,
        }


class NeuralForward(Operator):
    """Wrap a neural-emulator operator (e.g. `ModelOp`) as a `ForwardModel`.

    The wrapped ``model_op`` should be a `pipekit.Operator` that accepts
    the state carrier and returns the advanced state. ``dt`` is
    declared explicitly — neural emulators have a fixed training
    cadence that callers must respect.

    Args:
        model_op: The neural emulator `Operator`. Loaded by
            content-hash from a model registry in typical use.
        dt: The (fixed) integration step the emulator was trained for.
        state_signature: Optional `pipekit.Signature` for the state.

    Raises:
        TypeError: if ``model_op`` isn't an `Operator`.
    """

    __config_mixin_auto__ = False

    def __init__(
        self,
        model_op: Operator,
        dt: float,
        state_signature: Any = None,
    ) -> None:
        if not isinstance(model_op, Operator):
            raise TypeError(
                "NeuralForward.model_op must be an Operator, "
                f"got {type(model_op).__name__}."
            )
        self.model_op = model_op
        self._dt = float(dt)
        self._state_signature = state_signature

    def step(self, state: Any, dt: float) -> Any:
        # `dt` is accepted for protocol conformance; the emulator's
        # cadence is fixed at construction.
        return self.model_op(state)

    def _apply(self, state: Any) -> Any:
        return self.step(state, self._dt)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def state_signature(self) -> Any:
        return self._state_signature

    def get_config(self) -> dict[str, Any]:
        return {
            "model_op": {
                "class": type(self.model_op).__name__,
                "config": self.model_op.get_config(),
            },
            "dt": self._dt,
        }
