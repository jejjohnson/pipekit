"""Generic observation operators.

Each is a `pipekit.Operator` that also satisfies the
`ObservationOperator` Protocol so it can be passed wherever a
protocol-typed slot is expected (e.g. into `DACycle.obs_op`).

- `IdentityObs` — direct measurement (``H = I``).
- `LinearObs(H)` — fixed linear map, ``H @ state``. Carrier-agnostic
  via duck-typed ``@``.
- `CallableObs(fn)` — arbitrary callable (e.g. radiative-transfer
  forward). Carries ``forbid_in_yaml = True``.
- `CompositeObs(components)` — ``h_k ∘ … ∘ h_1`` applied left-to-right.

See master plan Report 10, section 2.5.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pipekit import Operator


class IdentityObs(Operator):
    """Direct measurement: ``H(x) = x``.

    The state is observable as-is — useful for synthetic-twin tests
    and as the default observation operator when one is needed but
    no real mapping applies.
    """

    def _apply(self, state: Any) -> Any:
        return state

    def linearize(self, state: Any) -> Operator:
        """Identity is its own tangent linear."""
        return self


class LinearObs(Operator):
    """Linear observation map: ``H(x) = H @ x``.

    Args:
        H: Any object supporting the ``@`` operator with the carrier.
            Typically a numpy / JAX array; pipekit-cycle stays
            backend-agnostic by relying on ``__matmul__``.
    """

    __config_mixin_auto__ = False

    def __init__(self, H: Any) -> None:
        self.H = H

    def _apply(self, state: Any) -> Any:
        return self.H @ state

    def linearize(self, state: Any) -> LinearObs:
        """The map is already linear — returns ``self``."""
        return self

    def get_config(self) -> dict[str, Any]:
        return {"H_shape": _shape_or_repr(self.H)}


class CallableObs(Operator):
    """Arbitrary callable observation operator.

    The escape hatch for nonlinear observation operators (radiative
    transfer, instrument-response convolution, …). Carries
    ``forbid_in_yaml = True`` because the callable doesn't round-trip
    through JSON / YAML.

    Args:
        fn: Callable ``state → predicted_obs``.
        linearize_fn: Optional callable ``state → linear operator``.
            If absent, `linearize` raises `NotImplementedError`.
        name: Optional name for ``__repr__``.
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        fn: Callable[[Any], Any],
        linearize_fn: Callable[[Any], Any] | None = None,
        name: str | None = None,
    ) -> None:
        self.fn = fn
        self.linearize_fn = linearize_fn
        self.name = name or getattr(fn, "__name__", "CallableObs")

    def _apply(self, state: Any) -> Any:
        return self.fn(state)

    def linearize(self, state: Any) -> Any:
        if self.linearize_fn is None:
            raise NotImplementedError(f"{self.name}: no linearisation supplied.")
        return self.linearize_fn(state)

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}

    def __repr__(self) -> str:
        return f"CallableObs(name={self.name!r})"


class CompositeObs(Operator):
    """Compose multiple observation operators: ``H_k ∘ … ∘ H_1``.

    Apply ``components[0]`` first, then ``components[1]``, etc.

    Args:
        components: One or more `Operator` instances. Each must accept
            the previous component's output as its input.

    Raises:
        TypeError: if any component isn't an `Operator`.
        ValueError: if ``components`` is empty.
    """

    __config_mixin_auto__ = False

    def __init__(self, components: tuple[Operator, ...]) -> None:
        if not components:
            raise ValueError("CompositeObs requires at least one component.")
        for i, op in enumerate(components):
            if not isinstance(op, Operator):
                raise TypeError(
                    f"CompositeObs.components[{i}] must be an Operator, "
                    f"got {type(op).__name__}."
                )
        self.components = tuple(components)

    def _apply(self, state: Any) -> Any:
        out = state
        for op in self.components:
            out = op(out)
        return out

    def linearize(self, state: Any) -> Any:
        """Compose component linearisations (chain rule).

        Each component must expose a `linearize`. If a non-linearisable
        component is present, raises `NotImplementedError`.
        """
        lins: list[Operator] = []
        carrier = state
        for op in self.components:
            linearize = getattr(op, "linearize", None)
            if linearize is None:
                raise NotImplementedError(
                    f"CompositeObs: component {type(op).__name__} has no "
                    "`linearize` — cannot build composite tangent linear."
                )
            lins.append(linearize(carrier))
            carrier = op(carrier)
        return CompositeObs(tuple(lins))

    def get_config(self) -> dict[str, Any]:
        return {
            "components": [
                {"class": type(op).__name__, "config": op.get_config()}
                for op in self.components
            ]
        }


def _shape_or_repr(obj: Any) -> Any:
    """Return ``obj.shape`` if it has one, else ``repr(obj)``."""
    shape = getattr(obj, "shape", None)
    if shape is not None:
        return list(shape)
    return repr(obj)
