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
   round-trips through any subclass of `Operator` that's been imported,
   nested operators included: a child operator appears in its parent's
   config as its own `state` record, and `from_state` rebuilds the tree
   from the leaves up.

Operators that themselves hold non-serialisable runtime state — user
closures (`Tap`, `Lambda`, `Branch`, …) or runtime arrays — set
`forbid_in_yaml = True` to signal that `get_config()` is a debug repr,
not a faithful round-trip; `from_state` refuses flagged classes, at any
depth, and `parallel.check_pickleable` surfaces flagged instances as
pickling hazards. Operators that merely *nest* other operators (`Cache`,
`Try`, the parallel maps, …) do not set the flag: they round-trip, and
nesting is discovered structurally by `check_pickleable`.

See master plan Report 2, Group A.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pipekit.adapters import BackendError, backend_of, signature_of


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
        backends: The array backends the operator runs on (``"numpy"``,
            ``"jax"``, ``"dask"``), or ``None`` (the default) for an
            operator that does not care. When set, an eager call whose
            carriers or arrays sit on another backend, or on several,
            raises `pipekit.adapters.BackendError` before ``_apply``
            runs: inputs never move between backends silently. Values
            with no backend (numbers, strings, tables) are not checked.

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
    backends: ClassVar[tuple[str, ...] | None] = None
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
            BackendError: ``backends`` is set and an input sits on another
                backend, or the inputs sit on several.
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
        self._check_backends(args, kwargs)
        out = self._apply(*args, **kwargs)
        # Reserved hook dispatch — no-op in v0.1; Spy/Hook family lands later
        # without requiring every Operator subclass to be edited.
        self._dispatch_post_apply_hooks(args, out)
        return out

    def _check_backends(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Raise `BackendError` when the inputs break ``backends`` (if set).

        `__call__` runs it before ``_apply``; anything else that calls
        ``_apply`` directly (`AsyncMap` awaiting a coroutine ``_apply``)
        runs it too.
        """
        if self.backends is not None:
            _require_backends(type(self).__name__, self.backends, args, kwargs)

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
        (carriers without named dimensions); `Sequential.summary` and
        `Graph.summary` show ``?`` for it, and every later step in a
        `Sequential` is unknown too.

        Handed a carrier instead of a signature, the default reads the
        carrier's `Signature` through its registered adapter
        (`pipekit.adapters.signature_of`), so ``Sequential.summary`` can
        start from data.

        The annotation is `Any` rather than `Signature | None` so domain
        libraries can thread their own signature types.
        """
        return signature_of(input_signature)

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
        """Reconstruct an operator, and any operators nested in it, from `state`.

        Resolves classes among `cls` and all its transitive subclasses,
        so ``Operator.from_state(state)`` resolves any imported subclass;
        nothing is imported. Config values are rebuilt recursively: a
        nested `state` record becomes an operator, an
        ``{"exception": "module.Name"}`` record becomes that (already
        imported) exception class, lists and dicts are rebuilt item by
        item, and JSON primitives pass through. The class's
        `from_config` then builds the operator.

        Raises:
            ValueError: if a state record is malformed.
            TypeError: if a class isn't a loaded `Operator` subclass, or an
                exception record doesn't name a loaded exception class.
            RuntimeError: if any class in the tree is marked
                ``forbid_in_yaml``, or a config holds a value that is not
                JSON (or a nested payload without a ``module``, as
                written before nested round-trips existed).
        """
        module_name, class_name, _ = _check_record(state)
        op_type = resolve_loaded(module_name, class_name)
        if not issubclass(op_type, cls):
            raise TypeError(f"{module_name}.{class_name} is not a {cls.__name__}.")
        return build_operator(state, resolve_loaded)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Build an operator from its decoded `get_config()` record.

        The default calls ``cls(**config)``, which is right for every
        operator whose config keys are its constructor's arguments.
        Operators whose config is not a constructor call (`Graph`,
        whose config is a topology) override this.

        Args:
            config: The config with nested operators already rebuilt.
        """
        return cls(**config)

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


def _require_backends(
    name: str,
    backends: tuple[str, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Raise `BackendError` unless every input sits on one listed backend.

    The inputs are the positional and keyword arguments, and the items of a
    list or tuple among them (a mosaic's scenes).
    """
    found: set[str] = set()
    for value in (*args, *kwargs.values()):
        items = value if isinstance(value, list | tuple) else (value,)
        for item in items:
            backend = backend_of(item)
            if backend is not None:
                found.add(backend)
    if len(found) > 1:
        raise BackendError(
            f"{name}: inputs on {sorted(found)}; put them on one backend first "
            "(inputs never move between backends silently)."
        )
    if found and (got := found.pop()) not in backends:
        runs = " and ".join(backends)
        raise BackendError(
            f"{name} runs on {runs} only, not {got}. Convert the input first "
            "(a carrier's to_numpy() or to_jax()); inputs never move between "
            "backends silently."
        )


def nested_config(op: Operator) -> dict[str, Any]:
    """Config payload for an operator nested inside another's config.

    The child's full `state` record (``module``, ``class``, ``config``),
    which `Operator.from_state` rebuilds. Every container operator
    (`Sequential`, `Cache`, `Branch`, the parallel maps, …) emits its
    children through this one helper, so the payload can never drift
    between operators.
    """
    return op.state


def exception_config(exc: type[BaseException]) -> dict[str, str]:
    """Config payload for an exception class (`Try.on`, `Retry.on`).

    ``{"exception": "module.QualName"}``, which `Operator.from_state`
    resolves back to the class among already-imported modules.
    """
    return {"exception": f"{exc.__module__}.{exc.__qualname__}"}


Resolver = Callable[[str, str], "type[Operator]"]


def build_operator(state: dict[str, Any], resolve: Resolver) -> Operator:
    """Rebuild an operator tree from a `state` record.

    The engine behind `Operator.from_state` and the sandboxed loader:
    ``resolve(module, class)`` maps each class name in the tree to a
    class (and may refuse, by raising); everything else is shared.

    Args:
        state: A ``{"module", "class", "config"}`` record.
        resolve: Maps a module and class name to an `Operator` subclass.

    Raises:
        ValueError / TypeError / RuntimeError: as `Operator.from_state`.
    """
    module_name, class_name, config = _check_record(state)
    op_type = resolve(module_name, class_name)
    if op_type.forbid_in_yaml:
        # A flagged operator's get_config() is a debug payload: its keys
        # need not match the constructor.
        raise RuntimeError(
            f"from_state cannot reconstruct {op_type.__name__}: it is marked "
            "forbid_in_yaml (its config is a debug payload, not a faithful "
            "round-trip). Rebuild it in code instead."
        )
    decoded: dict[str, Any] = {}
    for key, value in config.items():
        decoded[key] = _decode(value, resolve, f"{op_type.__name__}.{key}")
    return op_type.from_config(decoded)


def _check_record(state: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Validate a state record's shape; return its module, class and config."""
    module_name = state.get("module")
    class_name = state.get("class")
    config = state.get("config", {})
    if not isinstance(module_name, str):
        raise ValueError("Operator state must include a module name")
    if not isinstance(class_name, str):
        raise ValueError("Operator state must include a class name")
    if not isinstance(config, dict):
        raise ValueError("Operator state config must be a dictionary")
    return module_name, class_name, config


def _decode(value: Any, resolve: Resolver, where: str) -> Any:
    """Rebuild one config value: operators, exception classes, containers."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        for i, item in enumerate(value):
            items.append(_decode(item, resolve, f"{where}[{i}]"))
        # keep a tuple a tuple: a state built in memory round-trips exactly
        if isinstance(value, tuple):
            return tuple(items)
        return items
    if isinstance(value, dict):
        keys = set(value)
        if keys == {"module", "class", "config"}:
            return build_operator(value, resolve)
        if keys == {"class", "config"}:
            raise RuntimeError(
                f"from_state cannot reconstruct {where}: it is a nested "
                "operator payload without a `module`, written before nested "
                "round-trips existed. Dump the pipeline again."
            )
        if keys == {"exception"}:
            return _resolve_exception(value["exception"])
        rebuilt: dict[str, Any] = {}
        for key, item in value.items():
            rebuilt[key] = _decode(item, resolve, f"{where}[{key!r}]")
        return rebuilt
    raise RuntimeError(
        f"from_state cannot reconstruct {where}: {type(value).__name__} is "
        "not a JSON value."
    )


def _resolve_exception(path: Any) -> type[BaseException]:
    """``"module.QualName"`` ↦ an exception class from an imported module."""
    if not isinstance(path, str) or "." not in path:
        raise ValueError(f"Exception record must be 'module.Name', got {path!r}.")
    module_name, _, qualname = path.rpartition(".")
    # Already-imported modules only: loading a pipeline never imports code.
    target: Any = sys.modules.get(module_name)
    for part in qualname.split("."):
        target = getattr(target, part, None)
    if not (isinstance(target, type) and issubclass(target, BaseException)):
        raise TypeError(f"{path} is not a loaded exception class.")
    return target


def resolve_loaded(module_name: str, class_name: str) -> type[Operator]:
    """The loaded `Operator` subclass named ``module.class``; nothing is imported.

    A class that moved lists its old modules in ``__former_modules__``, so
    states saved before the move still load.

    Raises:
        TypeError: if no loaded `Operator` subclass has that name.
    """
    for op_type in (Operator, *_operator_subclasses(Operator)):
        if op_type.__name__ != class_name:
            continue
        # a class that moved keeps loading states saved under its old module
        former = getattr(op_type, "__former_modules__", ())
        if op_type.__module__ == module_name or module_name in former:
            return op_type
    raise TypeError(f"{module_name}.{class_name} is not a loaded Operator")


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
