"""Group C — building blocks.

Tiny operators that on their own look trivial but in combination
unlock common patterns:

| Operator                   | Purpose                                            |
| -------------------------- | -------------------------------------------------- |
| `Identity`                 | Explicit no-op; the default ``if_false`` for `Branch`. |
| `Const(value)`             | Return a fixed value, ignoring input.              |
| `Lambda(fn, name=...)`     | Wrap a callable as an `Operator`. The escape hatch. |
| `Sink(write_fn, name=...)` | Side-effect write; returns input unchanged.        |

`Lambda` and `Sink` set ``forbid_in_yaml = True`` — their state is a
user-supplied callable that doesn't round-trip through JSON / YAML.
They're the documented escape hatches; the rest of the operator
catalogue should round-trip cleanly.

See master plan Report 2, Group C.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pipekit._base.operator import Operator


class Identity(Operator):
    """Explicit no-op. Returns its input unchanged.

    Used as the default value for slots that need an `Operator` but
    where "do nothing" is the right behaviour — `Branch.if_false`,
    `Switch.default`, training-mode-off augmentation, etc.

    Example::

        Branch(predicate=is_clean, if_true=Identity(), if_false=Despeckle())
    """

    def _apply(self, x: Any) -> Any:
        return x


class Const(Operator):
    """Return a fixed value, ignoring any input.

    Useful for test fixtures, `Switch` default cases that should emit a
    sentinel, and golden values used in `Try` fallbacks.

    Example::

        zero = Const(0.0)
        assert zero() == 0.0
        assert zero("anything") == 0.0

    The value should be JSON-primitive for round-trip via `from_state`.
    """

    def __init__(self, value: Any) -> None:
        self.value = value

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        return self.value


class Lambda(Operator):
    """Wrap an arbitrary callable as an `Operator`.

    The escape hatch for one-off operations that don't warrant a named
    subclass. Carries ``forbid_in_yaml = True`` because the wrapped
    callable typically captures closure state that doesn't survive
    JSON / YAML round-trip.

    Args:
        fn: The callable to wrap. Receives the carrier and any extra
            positional / keyword arguments forwarded through ``__call__``.
        name: Optional name for ``__repr__``. Defaults to the
            callable's ``__name__`` if available, otherwise ``"Lambda"``.

    Example::

        twice = Lambda(lambda x: x * 2, name="double")
        assert twice(3) == 6
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False  # auto-config would emit the closure

    def __init__(self, fn: Callable[..., Any], name: str | None = None) -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "Lambda")

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def get_config(self) -> dict[str, Any]:
        # Debug-only — closures don't round-trip.
        return {"name": self.name}

    def __repr__(self) -> str:
        return f"Lambda(name={self.name!r})"


class Sink(Operator):
    """Apply a side-effecting write and pass the input through unchanged.

    The canonical checkpointing primitive: log, persist, broadcast, etc.
    The return value of ``write_fn`` is discarded; the original input
    flows on.

    `Sink` is **not** marked ``_terminal`` because it returns the
    carrier — a `Sequential` can continue past it. For operators that
    legitimately return ``None`` (write-to-disk, render-and-exit),
    define a subclass with ``_terminal = True`` and put it at the end.

    Args:
        write_fn: Callable that consumes the carrier for side effects.
            Its return value is ignored.
        name: Optional name for ``__repr__``. Defaults to the
            callable's ``__name__`` if available, otherwise ``"Sink"``.

    Example::

        log = Sink(lambda gt: print(gt.shape), name="log_shape")
        pipe = Scale(2.0) | log | AddConst(1.0)
        # log fires between Scale and AddConst; the carrier is untouched
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(self, write_fn: Callable[..., Any], name: str | None = None) -> None:
        self.write_fn = write_fn
        self.name = name or getattr(write_fn, "__name__", "Sink")

    def _apply(self, x: Any, /, *args: Any, **kwargs: Any) -> Any:
        self.write_fn(x, *args, **kwargs)
        return x

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}

    def __repr__(self) -> str:
        return f"Sink(name={self.name!r})"
