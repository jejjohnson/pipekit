"""Operator base class — the composition primitive.

Every concrete operator (`MaskClouds`, `NDVI`, `ModelOp`, …) subclasses
`Operator` and implements `_apply`. The base class handles three
behaviours every operator inherits:

1. **Dual-mode `__call__`** — `op(x)` runs eagerly on data; `op(node)`
   builds a `Node` in a `Graph`. Subclasses only implement `_apply`.
2. **Config round-trip via `ConfigMixin`** — `get_config()` is
   auto-derived from `__init__` parameter names by reading matching
   instance attributes. Subclasses opt out with
   `__config_mixin_auto__ = False` or exclude fields with
   `__config_exclude__ = ("foo",)`.
3. **JSON-safe `state` / `from_state`** — a serialisable record that
   round-trips through any subclass of `Operator` that's been imported.

Operators that themselves hold non-serialisable runtime state — user
closures (`Tap`, `Lambda`, `Branch`, …) or runtime arrays — set
`forbid_in_yaml = True` to signal that `get_config()` is a debug repr,
not a faithful round-trip; `from_state` refuses flagged classes and
`parallel.check_pickleable` surfaces flagged instances as pickling
hazards. Operators that merely *nest* other operators (`Cache`, `Try`,
the parallel maps, …) do not set the flag: nesting is discovered
structurally by `check_pickleable`, and their nested-dict configs are
already rejected cleanly by `from_state`'s primitive check.

See master plan Report 2, Group A.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, ClassVar


if TYPE_CHECKING:
    from pipekit._base.sequential import Sequential


# The carrier that flows through a pipeline. `Carrier = Any` because
# pipekit is carrier-agnostic — sister libraries narrow this at their
# own signatures (geotoolz → GeoTensor, xr_toolz → xr.Dataset, etc.).
Carrier = Any

# Shared "argument was omitted" sentinel — distinguishes an omitted
# parameter from an explicit None (used by `Sequential._apply` and
# `qc.AssertAttr`).
_MISSING = object()


class ConfigMixin:
    """Auto-derive `get_config()` from `__init__` parameter names.

    Reads `inspect.signature(type(self).__init__)`, then for each
    declared parameter name looks up the matching instance attribute on
    `self`. The discipline: store `__init__` kwargs as attributes of the
    same name (the geotoolz / xr_toolz convention).

    Opt-out: set ``__config_mixin_auto__ = False`` to disable
    auto-derivation entirely (subclasses then must override
    `get_config` themselves).

    Exclude: set ``__config_exclude__ = ("field_a",)`` to skip specific
    parameters (useful for runtime-only state).

    Subclasses can override `get_config` directly when auto-derivation
    isn't appropriate (e.g. `Sequential` whose config is a list of
    nested operators, not constructor parameters).
    """

    __config_mixin_auto__: ClassVar[bool] = True
    __config_exclude__: ClassVar[tuple[str, ...]] = ()

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of constructor args."""
        if not type(self).__config_mixin_auto__:
            return {}
        try:
            sig = inspect.signature(type(self).__init__)
        except (TypeError, ValueError):
            return {}
        excluded = set(type(self).__config_exclude__)
        config: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name in excluded:
                continue
            if hasattr(self, name):
                config[name] = getattr(self, name)
        return config


class Operator(ConfigMixin):
    """Base class for pipekit operators.

    Subclasses implement ``_apply(self, *args, **kwargs)``. The base
    class dispatches ``__call__`` to either ``_apply`` (eager) or
    `Node` construction (symbolic) based on whether any positional
    argument is a graph node.

    Attributes:
        forbid_in_yaml: ``True`` on subclasses that *directly* hold
            non-serialisable user state (callables, closures, runtime
            arrays, RNGs). YAML loaders and `from_state` refuse to
            instantiate flagged operators; `parallel.check_pickleable`
            surfaces them as pickleability warnings. Operators that only
            nest other operators leave this ``False`` — the walker finds
            flagged children structurally. Default ``False``.
        _terminal: ``True`` on subclasses that legitimately return
            ``None`` (or otherwise break the carrier-in / carrier-out
            contract). `Sequential` rejects terminal operators except
            as the last step. Default ``False``.

    Example:
        Define and compose::

            class Scale(Operator):
                def __init__(self, factor: float) -> None:
                    self.factor = factor

                def _apply(self, x):
                    return x * self.factor

            pipe = Scale(0.5) | Scale(2.0)   # Sequential([Scale(0.5), Scale(2.0)])
            assert pipe(10.0) == 10.0
    """

    forbid_in_yaml: ClassVar[bool] = False
    _terminal: ClassVar[bool] = False
    # Stateful-pipeline marker — `StatefulOperator` overrides to True; the
    # `Sequential` plumbing reads this to decide whether to thread state.
    # See `pipekit.state` (Group M).
    _is_stateful: ClassVar[bool] = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch eager vs symbolic on positional arguments.

        If any positional argument is a `Node` (which `Input` subclasses),
        all positional args must be `Node`s and the call returns a new
        `Node` recording this operator and its parents. Otherwise the
        call routes to `_apply`. Subclasses override `_apply`, not
        `__call__`.

        Raises:
            TypeError: if graph-construction mode is triggered (at
                least one `Node` arg) but other positional args are not
                `Node`s. Mixing nodes and literals would crash
                downstream when `Graph` walks ``parents``; reject it at
                construction with a clear message.
        """
        # Lazy import to break the operator <-> graph circular dependency.
        from pipekit._base.graph import Node

        if any(isinstance(a, Node) for a in args):
            non_node = [
                (i, type(a).__name__)
                for i, a in enumerate(args)
                if not isinstance(a, Node)
            ]
            if non_node:
                raise TypeError(
                    f"{type(self).__name__}: graph-construction mode requires "
                    "every positional arg to be a Node. Non-Node arg(s) at "
                    f"position(s) {non_node!r}. Wrap literals in `Const(value)` "
                    "or supply them as a separate `Input`."
                )
            return Node(operator=self, parents=tuple(args))
        out = self._apply(*args, **kwargs)
        # Reserved hook dispatch — no-op in v0.1; Spy/Hook family lands later
        # without requiring every Operator subclass to be edited.
        self._dispatch_post_apply_hooks(args, out)
        return out

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        """Implement the operator's behaviour. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement _apply().")

    def _dispatch_post_apply_hooks(self, args: tuple[Any, ...], out: Any) -> None:
        """No-op in v0.1. Reserved for the Spy/Hook family (Group E v0.2)."""
        return None

    def compute_output_signature(self, input_signature: Any) -> Any:
        """Return the signature of this operator's output given an input signature.

        Default is passthrough — most carrier-agnostic operators preserve
        shape and dtype. Subclasses that reshape / reduce / retype override.
        Returning ``None`` signals "this operator does not track shape"
        (carriers without named dimensions); `Sequential.summary` raises
        a clean error rather than blowing up downstream.

        Note: the annotation is `Any` for v0.1; it narrows to
        `Signature | None` once Group I (shape inference) lands.
        """
        return input_signature

    @property
    def state(self) -> dict[str, Any]:
        """Return a JSON-serialisable state record for this operator."""
        return {
            "module": type(self).__module__,
            "class": type(self).__name__,
            "config": self.get_config(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Operator:
        """Reconstruct an operator from a `state` record.

        Walks `cls` plus all transitive subclasses so calling
        `Operator.from_state(state)` resolves any imported subclass.
        Config values must be JSON primitives (None / bool / int / float
        / str / list / tuple of the same); operators that nest other
        operators in their config emit debug payloads and are not
        round-trippable through this path — use a YAML / Hydra loader
        instead.

        Raises:
            ValueError: if the state record is malformed.
            TypeError: if the class isn't a loaded `Operator` subclass.
            RuntimeError: if the class is marked ``forbid_in_yaml`` or the
                config contains non-primitive values.
        """
        module_name = state.get("module")
        class_name = state.get("class")
        config = state.get("config", {})
        if not isinstance(module_name, str):
            raise ValueError("Operator state must include a module name")
        if not isinstance(class_name, str):
            raise ValueError("Operator state must include a class name")
        if not isinstance(config, dict):
            raise ValueError("Operator state config must be a dictionary")

        for op_type in (cls, *_operator_subclasses(cls)):
            if op_type.__module__ == module_name and op_type.__name__ == class_name:
                if op_type.forbid_in_yaml:
                    # A flagged operator's get_config() is a debug payload —
                    # its keys need not match the constructor, so calling
                    # op_type(**config) would raise a confusing TypeError.
                    raise RuntimeError(
                        f"from_state cannot reconstruct {op_type.__name__}: "
                        "it is marked forbid_in_yaml (its config is a debug "
                        "payload, not a faithful round-trip). Rebuild it in "
                        "code instead."
                    )
                non_primitive = [
                    k for k, v in config.items() if not _is_json_primitive(v)
                ]
                if non_primitive:
                    raise RuntimeError(
                        f"from_state cannot reconstruct {op_type.__name__}: "
                        f"config contains non-primitive values {non_primitive}. "
                        "Use a YAML / Hydra loader for nested-operator configs."
                    )
                return op_type(**config)
        raise TypeError(f"{module_name}.{class_name} is not a loaded Operator")

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_config().items())
        return f"{type(self).__name__}({params})"

    def __or__(self, other: Operator) -> Sequential:
        """``op1 | op2`` returns ``Sequential([op1, op2])``, flattening on the right.

        ``a | (b | c)`` produces a single three-element `Sequential`.
        """
        # Lazy import — Sequential subclasses Operator.
        from pipekit._base.sequential import Sequential

        if not isinstance(other, Operator):
            return NotImplemented
        if isinstance(other, Sequential):
            return Sequential([self, *other.operators])
        return Sequential([self, other])


def nested_config(op: Operator) -> dict[str, Any]:
    """Debug-config payload for an operator nested inside another's config.

    The canonical ``{"class": ..., "config": ...}`` shape every container
    operator (`Sequential`, `Cache`, `Branch`, the parallel maps, …) emits
    for its children. One helper instead of one inline dict per module, so
    the payload shape can never drift between operators.
    """
    return {"class": type(op).__name__, "config": op.get_config()}


def callable_name(fn: Any) -> str:
    """Best-effort display name for a user-supplied callable.

    ``__name__`` when present (functions, classes), else ``repr`` (lambdas
    assigned through functools.partial, callable instances, …).
    """
    return getattr(fn, "__name__", repr(fn))


def require_operator(value: Any, owner: str, field: str) -> None:
    """Raise a standardized ``TypeError`` unless ``value`` is an `Operator`.

    Args:
        value: The candidate object.
        owner: The validating class's name (for the message).
        field: The parameter/attribute name (may include an index, e.g.
            ``"sources[2]"``).
    """
    if not isinstance(value, Operator):
        raise TypeError(
            f"{owner}.{field} must be an Operator, got {type(value).__name__}."
        )


def _operator_subclasses(cls: type[Operator]) -> tuple[type[Operator], ...]:
    """All transitive subclasses of ``cls``, in definition order."""
    subclasses: list[type[Operator]] = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(_operator_subclasses(subclass))
    return tuple(subclasses)


def _is_json_primitive(value: Any) -> bool:
    """Return ``True`` if ``value`` is a safe `from_state` constructor payload.

    Accepts JSON primitives (None / bool / int / float / str) and
    list / tuple of the same, recursively. Rejects dicts — operators
    that nest other operators in their config emit ``{"class", "config"}``
    debug payloads that the plain constructor can't consume.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_primitive(v) for v in value)
    return False
