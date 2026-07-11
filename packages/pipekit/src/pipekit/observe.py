"""Group E — observers.

Identity-with-side-effect operators. The carrier flows through
unchanged; something useful happens on the side. Critical for
notebook exploration and debug, useful in production logging.

- `Tap(fn, name=...)` — fire a callback, return input unchanged.
  Distinct from `Sink`: `Tap` is named for observation (logging,
  metrics); `Sink` for persistence. Same machinery, different intent.
- `Snapshot()` — controller. ``Snapshot.at(key)`` returns an
  operator that captures the input under ``key``. Inspect via
  ``snapshot.captures`` (dict).
- `ShapeTrace(printer=print, mode="every"|"diff_only")` — log
  shape / dtype at every step. Carrier-agnostic via ``getattr``.
- `Profile()` — controller. ``Profile.wrap(op)`` returns a wrapped
  operator that records timings. Inspect via ``profile.report()``.
- `Histogram(bins=10, to_array=...)` — controller. ``Histogram.at(key)``
  returns an operator that captures a distribution; ``to_array`` is a
  user-supplied callable adapting the carrier to a flat sequence of
  numbers (default: ``list``).

`Snapshot`, `Profile`, and `Histogram` use the controller-not-operator
pattern: the class itself isn't an `Operator`, but its ``.at()`` /
``.wrap()`` methods return small operators that close over the
controller's state. This keeps the per-step operators tiny while the
collected data lives on a single object the user can inspect.

See master plan Report 2, Group E.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

from pipekit._base.operator import Operator, nested_config, require_operator


class Tap(Operator):
    """Fire a callback on the carrier; return the carrier unchanged.

    Distinct from `Sink` only by naming convention — `Tap` is for
    observation (logging, metrics, breakpoint hooks); `Sink` is for
    persistence. Both pass the carrier through; both carry
    ``forbid_in_yaml = True``.

    Args:
        fn: Callable invoked with the carrier. Its return value is
            ignored.
        name: Optional name for ``__repr__``. Defaults to the
            callable's ``__name__`` if available, otherwise ``"Tap"``.

    Example::

        log = Tap(lambda x: print(getattr(x, "shape", x)), name="log_shape")
        pipe = Scale(2.0) | log | AddConst(1.0)
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(self, fn: Callable[[Any], Any], name: str | None = None) -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "Tap")

    def _apply(self, x: Any) -> Any:
        self.fn(x)
        return x

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}

    def __repr__(self) -> str:
        return f"Tap(name={self.name!r})"


class _SnapshotAt(Operator):
    """Internal: operator returned by `Snapshot.at`."""

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(self, controller: Snapshot, key: str) -> None:
        self.controller = controller
        self.key = key

    def _apply(self, x: Any) -> Any:
        self.controller.captures[self.key] = x
        return x

    def get_config(self) -> dict[str, Any]:
        return {"key": self.key}

    def __repr__(self) -> str:
        return f"Snapshot.at({self.key!r})"


class Snapshot:
    """Controller that captures intermediate carriers by name.

    Not an `Operator` itself. Call ``.at(key)`` to obtain a per-step
    operator that stores the carrier under that key.

    Attributes:
        captures: Dict ``{key: last-seen carrier}``. Repeated keys
            overwrite — the most recent value wins.

    Example::

        snap = Snapshot()
        pipe = snap.at("pre") | Scale(2.0) | snap.at("post") | AddConst(1.0)
        pipe(3.0)
        snap.captures
        # {"after_scale": 3.0, "after_add": 6.0}
    """

    def __init__(self) -> None:
        self.captures: dict[str, Any] = {}

    def at(self, key: str) -> Operator:
        return _SnapshotAt(self, key)

    def clear(self) -> None:
        self.captures.clear()

    def __repr__(self) -> str:
        return f"Snapshot(keys={list(self.captures)!r})"


class ShapeTrace(Operator):
    """Log ``shape`` / ``dtype`` of each carrier passing through.

    Args:
        printer: Callable invoked with the formatted line. Defaults to
            the built-in ``print``.
        mode: ``"every"`` logs every call; ``"diff_only"`` logs only
            when the (shape, dtype) tuple changes from the previous
            call. ``"diff_only"`` is the typical notebook mode.
        label: Optional prefix for the printed line (useful when
            multiple `ShapeTrace` instances co-exist in one pipeline).

    Example::

        trace = ShapeTrace(mode="diff_only", label="post-resample")
        pipe = Resample() | trace | NDVI()
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        printer: Callable[[str], None] = print,
        mode: str = "every",
        label: str | None = None,
    ) -> None:
        if mode not in ("every", "diff_only"):
            raise ValueError(
                f"ShapeTrace.mode must be 'every' or 'diff_only', got {mode!r}."
            )
        self.printer = printer
        self.mode = mode
        self.label = label
        self._last: tuple[Any, Any] | None = None

    def _apply(self, x: Any) -> Any:
        shape = getattr(x, "shape", None)
        dtype = getattr(x, "dtype", None)
        current = (tuple(shape) if shape is not None else None, dtype)
        if self.mode == "diff_only" and current == self._last:
            return x
        self._last = current
        prefix = f"[{self.label}] " if self.label else ""
        self.printer(f"{prefix}shape={current[0]} dtype={current[1]}")
        return x

    def get_config(self) -> dict[str, Any]:
        return {"mode": self.mode, "label": self.label}

    def __repr__(self) -> str:
        return f"ShapeTrace(mode={self.mode!r}, label={self.label!r})"


class _ProfileWrap(Operator):
    """Internal: operator returned by `Profile.wrap`."""

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(self, controller: Profile, inner: Operator) -> None:
        self.controller = controller
        self.inner = inner

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = self.inner(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        name = type(self.inner).__name__
        self.controller.timings.setdefault(name, []).append(elapsed)
        return out

    def get_config(self) -> dict[str, Any]:
        return {"inner": nested_config(self.inner)}

    def __repr__(self) -> str:
        return f"Profile.wrap({self.inner!r})"


class Profile:
    """Controller that records per-step wall-clock timings.

    Not an `Operator` itself. ``Profile.wrap(op)`` returns a wrapped
    operator that times each call and appends the elapsed seconds to
    ``timings[type(op).__name__]``.

    Attributes:
        timings: Dict ``{operator-class-name: [elapsed_seconds, …]}``.

    Example::

        prof = Profile()
        pipe = prof.wrap(Scale(2.0)) | prof.wrap(AddConst(1.0))
        for v in [1.0, 2.0, 3.0]:
            pipe(v)
        prof.report()
        # {"Scale": {"calls": 3, "total": ..., "mean": ...}, ...}
    """

    def __init__(self) -> None:
        self.timings: dict[str, list[float]] = {}

    def wrap(self, op: Operator) -> Operator:
        require_operator(op, "Profile.wrap", "op")
        return _ProfileWrap(self, op)

    def report(self) -> dict[str, dict[str, float]]:
        """Return a ``{name: {calls, total, mean}}`` summary."""
        out: dict[str, dict[str, float]] = {}
        for name, samples in self.timings.items():
            total = sum(samples)
            out[name] = {
                "calls": float(len(samples)),
                "total": total,
                "mean": total / len(samples) if samples else 0.0,
            }
        return out

    def clear(self) -> None:
        self.timings.clear()

    def __repr__(self) -> str:
        return f"Profile(ops={list(self.timings)!r})"


class _HistogramAt(Operator):
    """Internal: operator returned by `Histogram.at`."""

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(self, controller: Histogram, key: str) -> None:
        self.controller = controller
        self.key = key

    def _apply(self, x: Any) -> Any:
        values = self.controller.to_array(x)
        self.controller._record(self.key, values)
        return x

    def get_config(self) -> dict[str, Any]:
        return {"key": self.key, "bins": self.controller.bins}

    def __repr__(self) -> str:
        return f"Histogram.at({self.key!r})"


class Histogram:
    """Controller that captures simple histograms by key.

    Not an `Operator` itself. ``Histogram.at(key)`` returns an
    operator that, on each call, runs ``to_array(carrier)`` and bins
    the result. Bin counts accumulate across calls.

    The default ``to_array`` (``list``) handles any iterable of numbers
    — including a plain ``list[float]``. Pass a carrier-specific
    adapter for numpy / xarray (e.g. ``to_array=lambda a: a.ravel().tolist()``).

    Args:
        bins: Number of equal-width bins.
        to_array: Adapter ``carrier → Iterable[float]``. Default: ``list``.

    Attributes:
        results: Dict ``{key: {"counts": [...], "edges": [...]}}``.

    Example::

        h = Histogram(bins=10)
        pipe = h.at("ndvi_pre") | Clean() | h.at("ndvi_post")
        for tile in tiles:
            pipe(tile)
        h.results["ndvi_post"]
    """

    def __init__(
        self,
        bins: int = 10,
        to_array: Callable[[Any], Iterable[float]] = list,
    ) -> None:
        if bins < 1:
            raise ValueError(f"Histogram.bins must be >= 1, got {bins}.")
        self.bins = bins
        self.to_array = to_array
        self.results: dict[str, dict[str, list[float]]] = {}

    def at(self, key: str) -> Operator:
        return _HistogramAt(self, key)

    def clear(self) -> None:
        self.results.clear()

    def _record(self, key: str, values: Iterable[float]) -> None:
        sample = [float(v) for v in values]
        if not sample:
            return
        lo, hi = min(sample), max(sample)
        if lo == hi:
            # Degenerate range — bin the lone value into bin 0.
            edges = [lo, lo + 1.0]
            counts = [len(sample)]
        else:
            width = (hi - lo) / self.bins
            edges = [lo + i * width for i in range(self.bins + 1)]
            counts = [0] * self.bins
            for v in sample:
                # Right-open bins, with the right edge inclusive on the last.
                idx = min(int((v - lo) / width), self.bins - 1)
                counts[idx] += 1
        prior = self.results.get(key)
        if prior is None or prior["edges"] != edges:
            self.results[key] = {"counts": list(counts), "edges": list(edges)}
        else:
            prior["counts"] = [
                a + b for a, b in zip(prior["counts"], counts, strict=True)
            ]

    def __repr__(self) -> str:
        return f"Histogram(bins={self.bins}, keys={list(self.results)!r})"
