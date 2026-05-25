"""Metric writer protocol + default JSON Lines implementation.

`MetricWriter` is the carrier-agnostic seam where per-step metrics
land when no `ExperimentTracker` is attached. `JSONLWriter` is the
default; W&B, TensorBoard, and other writers are user-supplied per
``docs/design/examples/integration.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetricWriter(Protocol):
    """Where per-step metrics go when no tracker is attached.

    Implementations: JSONL (default), in-memory list, W&B,
    TensorBoard. The trainer calls ``write`` per logged step.

    See ``docs/design/api/components.md``.
    """

    def write(self, step: int, metrics: dict[str, float]) -> None: ...

    def close(self) -> None: ...


@dataclass
class JSONLWriter:
    """Append-only JSON Lines metric writer.

    Each ``write`` appends one line of the form::

        {"step": int, "metrics": {name: float, ...}, "timestamp": iso8601}

    The file is opened lazily on the first ``write`` and flushed every
    ``flush_every`` writes.

    Args:
        path: Output file path. Parent directories are created if
            missing.
        flush_every: Number of writes between explicit ``flush()``
            calls. Higher = fewer syscalls; lower = less data lost
            on crash.

    See ``docs/design/api/components.md``.
    """

    path: str | Path
    flush_every: int = 100

    _file: Any = None
    _writes_since_flush: int = 0

    def write(self, step: int, metrics: dict[str, float]) -> None:
        """Append one record."""
        if self._file is None:
            path = Path(self.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")
        record = {
            "step": int(step),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._file.write(json.dumps(record) + "\n")
        self._writes_since_flush += 1
        if self._writes_since_flush >= self.flush_every:
            self._file.flush()
            self._writes_since_flush = 0

    def close(self) -> None:
        """Close the underlying file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writes_since_flush = 0
