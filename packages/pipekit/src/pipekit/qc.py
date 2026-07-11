"""Group H — quality control.

Assertions as composable operators. Every QC check is "I bet this
won't break" turned into a permanent runtime guard. The pipekit-level
operators stay carrier-agnostic — they look at ``shape`` / ``dtype``
via ``getattr``, never inside the array. Numeric checks
(`AssertValueRange`, `AssertNoNaN`, etc.) that need to look at array
contents live in sister libraries (Report 3, `pipekit-array`).

- `Quarantine(check, sentinel, on_quarantine=None)` — non-raising;
  failure → sentinel.
- `AssertShape(expected_shape)` — pass-through; raise on shape mismatch.
- `AssertDType(expected_dtype)` — pass-through; raise on dtype mismatch.
- `AssertHasAttribute(name, value=None)` — pass-through; check
  attribute presence (and value, if supplied).
- `AssertCallable(predicate, message=...)` — pass-through; user-supplied
  predicate; raise on failure.

See master plan Report 2, Group H.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pipekit._base.operator import _MISSING, Operator, callable_name


class QCError(AssertionError):
    """Raised when a QC assertion fails. Subclass of `AssertionError`."""


class Quarantine(Operator):
    """Non-raising QC: on failure log and return ``sentinel``; on success pass through.

    The "soft" QC pattern. Useful in batched / catalog-driven pipelines
    where one bad input shouldn't abort the whole run — quarantine it,
    keep going, and surface the count in the run report.

    Args:
        check: Callable ``carrier → bool``. ``True`` means OK.
        sentinel: Value returned when ``check`` fails.
        on_quarantine: Optional callable ``carrier → None`` invoked
            when ``check`` fails. Use for logging / metrics.

    Example::

        safe = Quarantine(
            check=lambda gt: gt.cloud_fraction < 0.5,
            sentinel=None,
            on_quarantine=lambda gt: log.warning("quarantined %s", gt.id),
        )
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        check: Callable[[Any], bool],
        sentinel: Any,
        on_quarantine: Callable[[Any], None] | None = None,
    ) -> None:
        self.check = check
        self.sentinel = sentinel
        self.on_quarantine = on_quarantine

    def _apply(self, x: Any) -> Any:
        if self.check(x):
            return x
        if self.on_quarantine is not None:
            self.on_quarantine(x)
        return self.sentinel

    def get_config(self) -> dict[str, Any]:
        return {
            "check": callable_name(self.check),
            "sentinel": _safe_repr(self.sentinel),
            "on_quarantine": (
                callable_name(self.on_quarantine)
                if self.on_quarantine is not None
                else None
            ),
        }


class AssertShape(Operator):
    """Pass-through; raise `QCError` if ``getattr(x, 'shape')`` mismatches.

    Args:
        expected_shape: Tuple of ints. ``None`` entries are wildcards.

    Example::

        AssertShape((None, 3, 256, 256))   # batch dim is free, rest are pinned
    """

    def __init__(self, expected_shape: tuple[int | None, ...]) -> None:
        self.expected_shape = tuple(expected_shape)

    def _apply(self, x: Any) -> Any:
        actual = getattr(x, "shape", None)
        if actual is None:
            raise QCError(
                f"AssertShape: input has no `shape` attribute (got {type(x).__name__})."
            )
        actual = tuple(actual)
        if len(actual) != len(self.expected_shape):
            raise QCError(
                f"AssertShape: expected rank {len(self.expected_shape)}, "
                f"got rank {len(actual)} (shape={actual})."
            )
        for axis, (e, a) in enumerate(zip(self.expected_shape, actual, strict=True)):
            if e is not None and e != a:
                raise QCError(
                    f"AssertShape: axis {axis} expected {e}, got {a} "
                    f"(expected={self.expected_shape}, actual={actual})."
                )
        return x


class AssertDType(Operator):
    """Pass-through; raise `QCError` if ``getattr(x, 'dtype')`` mismatches.

    Args:
        expected_dtype: A dtype label. Compared via ``str(actual) == str(expected)``
            so it accepts both string labels (``"float32"``) and numpy
            dtype objects without importing numpy at this layer.
    """

    def __init__(self, expected_dtype: Any) -> None:
        self.expected_dtype = expected_dtype

    def _apply(self, x: Any) -> Any:
        actual = getattr(x, "dtype", None)
        if actual is None:
            raise QCError(
                f"AssertDType: input has no `dtype` attribute (got {type(x).__name__})."
            )
        if str(actual) != str(self.expected_dtype):
            raise QCError(
                f"AssertDType: expected {self.expected_dtype!r}, got {actual!r}."
            )
        return x


class AssertHasAttribute(Operator):
    """Pass-through; raise `QCError` if attribute missing (or value mismatches).

    Args:
        name: Attribute name to look up via ``getattr``.
        value: Optional expected value. **Omit** the argument (it
            defaults to a private ``_MISSING`` sentinel) to check
            presence only; **pass** any value — including ``None`` — to
            additionally require equality with the attribute. ``None``
            as a *default* is therefore not a sentinel: it's a real
            value the user might assert against.
    """

    def __init__(self, name: str, value: Any = _MISSING) -> None:
        self.name = name
        self.value = value

    def _apply(self, x: Any) -> Any:
        if not hasattr(x, self.name):
            raise QCError(
                f"AssertHasAttribute: {type(x).__name__} has no attribute "
                f"{self.name!r}."
            )
        if self.value is not _MISSING:
            actual = getattr(x, self.name)
            if actual != self.value:
                raise QCError(
                    f"AssertHasAttribute: attribute {self.name!r} expected "
                    f"{self.value!r}, got {actual!r}."
                )
        return x

    def get_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"name": self.name}
        if self.value is not _MISSING:
            cfg["value"] = self.value
        return cfg


class AssertCallable(Operator):
    """Pass-through; raise `QCError` if ``predicate(x)`` is falsy.

    The escape hatch for QC checks that don't fit the typed asserts.
    Carries ``forbid_in_yaml = True`` because the predicate is a
    user callable.

    Args:
        predicate: Callable ``carrier → bool``. ``True`` means OK.
        message: Error message attached to the raised `QCError`. The
            predicate's name (if any) is included for context.
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        predicate: Callable[[Any], bool],
        message: str = "predicate returned False",
    ) -> None:
        self.predicate = predicate
        self.message = message

    def _apply(self, x: Any) -> Any:
        if not self.predicate(x):
            name = callable_name(self.predicate)
            raise QCError(f"AssertCallable({name}): {self.message}")
        return x

    def get_config(self) -> dict[str, Any]:
        return {
            "predicate": callable_name(self.predicate),
            "message": self.message,
        }


def _safe_repr(value: Any) -> Any:
    """Return ``value`` if it's a JSON-primitive; otherwise ``repr(value)``."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)
