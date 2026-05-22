"""Group D — control flow.

Conditionals and exception handling as first-class composable
operators. The whole point of this module: these are **not** Python
``if`` / ``try`` statements outside the pipeline — they're operators
*inside* it. That makes them serializable, profilable, and observable
through the same `Tap` / `Snapshot` / `ShapeTrace` machinery as any
other step.

- `Branch(predicate, if_true, if_false=Identity())` — binary conditional.
- `Switch(key, cases, default=Identity())` — multi-way dispatch on ``key(x)``.
- `Try(primary, fallback, on)` — direct map from ``toolz.excepts``.
- `Coalesce(sources, is_ok)` — first source whose output passes ``is_ok`` wins.
- `Retry(op, attempts, base_delay, on)` — exponential-backoff retry loop.

All five carry ``forbid_in_yaml = True`` — they hold callables /
exception types that don't JSON / YAML round-trip.

See master plan Report 2, Group D.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable
from typing import Any, ClassVar

from pipekit._base.operator import Operator
from pipekit.blocks import Identity


class Branch(Operator):
    """Binary conditional: run ``if_true`` or ``if_false`` based on a predicate.

    Args:
        predicate: Callable ``carrier → bool``. The decision function.
        if_true: `Operator` applied when ``predicate(carrier)`` is truthy.
        if_false: `Operator` applied when ``predicate(carrier)`` is falsy.
            Defaults to `Identity` — the carrier flows through unchanged.

    Example::

        clean_if_dirty = Branch(
            predicate=has_nans,
            if_true=GapFill(),
            if_false=Identity(),
        )
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        predicate: Callable[..., bool],
        if_true: Operator,
        if_false: Operator | None = None,
    ) -> None:
        if not isinstance(if_true, Operator):
            raise TypeError(
                f"Branch.if_true must be an Operator, got {type(if_true).__name__}."
            )
        if if_false is not None and not isinstance(if_false, Operator):
            raise TypeError(
                f"Branch.if_false must be an Operator, got {type(if_false).__name__}."
            )
        self.predicate = predicate
        self.if_true = if_true
        self.if_false = if_false if if_false is not None else Identity()

    def _apply(self, x: Any) -> Any:
        return self.if_true(x) if self.predicate(x) else self.if_false(x)

    def get_config(self) -> dict[str, Any]:
        return {
            "predicate": getattr(self.predicate, "__name__", repr(self.predicate)),
            "if_true": {
                "class": type(self.if_true).__name__,
                "config": self.if_true.get_config(),
            },
            "if_false": {
                "class": type(self.if_false).__name__,
                "config": self.if_false.get_config(),
            },
        }


class Switch(Operator):
    """Multi-way dispatch on ``key(carrier)``.

    Args:
        key: Callable ``carrier → hashable``. The dispatch key.
        cases: Mapping from key-value to `Operator`. Whichever case
            matches ``key(carrier)`` is applied.
        default: `Operator` for the unmatched case. Defaults to `Identity`.

    Example::

        by_sensor = Switch(
            key=lambda gt: gt.attrs["sensor"],
            cases={
                "sentinel-2": Sentinel2Pipeline(),
                "landsat-8":  Landsat8Pipeline(),
            },
            default=Identity(),
        )
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        key: Callable[..., Hashable],
        cases: dict[Hashable, Operator],
        default: Operator | None = None,
    ) -> None:
        for k, op in cases.items():
            if not isinstance(op, Operator):
                raise TypeError(
                    f"Switch.cases[{k!r}] must be an Operator, got {type(op).__name__}."
                )
        if default is not None and not isinstance(default, Operator):
            raise TypeError(
                f"Switch.default must be an Operator, got {type(default).__name__}."
            )
        self.key = key
        self.cases = dict(cases)
        self.default = default if default is not None else Identity()

    def _apply(self, x: Any) -> Any:
        k = self.key(x)
        op = self.cases.get(k, self.default)
        return op(x)

    def get_config(self) -> dict[str, Any]:
        return {
            "key": getattr(self.key, "__name__", repr(self.key)),
            "cases": {
                str(k): {"class": type(op).__name__, "config": op.get_config()}
                for k, op in self.cases.items()
            },
            "default": {
                "class": type(self.default).__name__,
                "config": self.default.get_config(),
            },
        }


class Try(Operator):
    """Run ``primary``; on caught exceptions run ``fallback``.

    Direct analogue of ``toolz.excepts``. If ``primary(carrier)`` raises
    one of the exception types listed in ``on``, ``fallback(carrier)``
    runs and its result is returned. Uncaught exception types
    propagate.

    Args:
        primary: `Operator` to try first.
        fallback: `Operator` invoked on caught exceptions.
        on: Tuple of exception types to catch. Required — no implicit
            ``Exception`` catch-all.

    Example::

        safe_read = Try(
            primary=ReadCOG(uri),
            fallback=Const(empty_tile),
            on=(IOError, FileNotFoundError),
        )
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        primary: Operator,
        fallback: Operator,
        on: tuple[type[BaseException], ...],
    ) -> None:
        if not isinstance(primary, Operator):
            raise TypeError(
                f"Try.primary must be an Operator, got {type(primary).__name__}."
            )
        if not isinstance(fallback, Operator):
            raise TypeError(
                f"Try.fallback must be an Operator, got {type(fallback).__name__}."
            )
        if not on:
            raise ValueError(
                "Try.on must list at least one exception type — "
                "Try is not a catch-all; use Exception explicitly if intended."
            )
        for exc in on:
            if not (isinstance(exc, type) and issubclass(exc, BaseException)):
                raise TypeError(
                    f"Try.on entries must be exception classes, got {exc!r}."
                )
        self.primary = primary
        self.fallback = fallback
        self.on = tuple(on)

    def _apply(self, x: Any) -> Any:
        try:
            return self.primary(x)
        except self.on:
            return self.fallback(x)

    def get_config(self) -> dict[str, Any]:
        return {
            "primary": {
                "class": type(self.primary).__name__,
                "config": self.primary.get_config(),
            },
            "fallback": {
                "class": type(self.fallback).__name__,
                "config": self.fallback.get_config(),
            },
            "on": [exc.__name__ for exc in self.on],
        }


class Coalesce(Operator):
    """Try each source in order; return the first output that satisfies ``is_ok``.

    The "first-good-result-wins" pattern. Each source runs on the same
    input; the first whose output passes ``is_ok`` is returned. If all
    sources fail the check, `Coalesce` raises ``RuntimeError``.

    Args:
        sources: Operators to try in order.
        is_ok: Predicate ``result → bool``. The acceptance criterion.

    Example::

        pick_best = Coalesce(
            sources=[ReadCloudFree(t), ReadPartialCloud(t), ReadAny(t)],
            is_ok=lambda gt: gt.cloud_fraction < 0.1,
        )
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        sources: list[Operator],
        is_ok: Callable[[Any], bool],
    ) -> None:
        if not sources:
            raise ValueError("Coalesce requires at least one source.")
        for i, op in enumerate(sources):
            if not isinstance(op, Operator):
                raise TypeError(
                    f"Coalesce.sources[{i}] must be an Operator, "
                    f"got {type(op).__name__}."
                )
        self.sources = list(sources)
        self.is_ok = is_ok

    def _apply(self, x: Any) -> Any:
        for op in self.sources:
            result = op(x)
            if self.is_ok(result):
                return result
        raise RuntimeError(
            f"Coalesce: none of {len(self.sources)} sources produced an output "
            "passing `is_ok`."
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "sources": [
                {"class": type(op).__name__, "config": op.get_config()}
                for op in self.sources
            ],
            "is_ok": getattr(self.is_ok, "__name__", repr(self.is_ok)),
        }


class Retry(Operator):
    """Retry an operator on transient failures with exponential backoff.

    Args:
        op: The `Operator` to invoke.
        attempts: Total number of attempts (≥ 1). The first attempt is
            attempt 1; subsequent retries are 2, 3, ... up to ``attempts``.
        base_delay: Base sleep in seconds; attempt ``i`` waits
            ``base_delay * (2 ** (i - 1))`` before retrying. The last
            attempt does not sleep before re-raising.
        on: Tuple of exception types to consider transient.
            Other exceptions propagate immediately.

    Example::

        sturdy_read = Retry(
            op=ReadFromS3(uri),
            attempts=5,
            base_delay=0.5,
            on=(ConnectionError, TimeoutError),
        )
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        op: Operator,
        attempts: int,
        base_delay: float,
        on: tuple[type[BaseException], ...],
    ) -> None:
        if not isinstance(op, Operator):
            raise TypeError(f"Retry.op must be an Operator, got {type(op).__name__}.")
        if attempts < 1:
            raise ValueError(f"Retry.attempts must be >= 1, got {attempts}.")
        if base_delay < 0:
            raise ValueError(f"Retry.base_delay must be >= 0, got {base_delay}.")
        if not on:
            raise ValueError(
                "Retry.on must list at least one exception type — "
                "Retry is not a catch-all."
            )
        for exc in on:
            if not (isinstance(exc, type) and issubclass(exc, BaseException)):
                raise TypeError(
                    f"Retry.on entries must be exception classes, got {exc!r}."
                )
        self.op = op
        self.attempts = attempts
        self.base_delay = base_delay
        self.on = tuple(on)

    def _apply(self, x: Any) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                return self.op(x)
            except self.on as exc:
                last_exc = exc
                if attempt < self.attempts:
                    self._sleep(self.base_delay * (2 ** (attempt - 1)))
        # Exhausted attempts — re-raise the last seen exception.
        assert last_exc is not None
        raise last_exc

    def _sleep(self, seconds: float) -> None:
        """Hook for tests to monkey-patch out real sleeping."""
        time.sleep(seconds)

    def get_config(self) -> dict[str, Any]:
        return {
            "op": {
                "class": type(self.op).__name__,
                "config": self.op.get_config(),
            },
            "attempts": self.attempts,
            "base_delay": self.base_delay,
            "on": [exc.__name__ for exc in self.on],
        }
