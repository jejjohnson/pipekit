"""Run records, per-step metrics, and named artifacts.

- `Run` — a single pipeline execution; identified by ``run_id`` and
  ``pipeline_hash``.
- `RunMetrics` — per-step metric series.
- `RunArtifacts` — named artifact URIs attached to a run.

See master plan Report 12, section 2.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


RunStatus = Literal["running", "completed", "failed"]


@dataclass
class Run:
    """A single execution of a pipeline (training or inference).

    Attributes:
        run_id: Tracker-assigned identifier (or a uuid for local runs).
        pipeline_hash: Pipekit-derived content hash of the pipeline.
        config: Flat dict of pipeline parameters logged at start.
        metrics: Latest metric values (the running snapshot;
            full per-step history lives in `RunMetrics`).
        artifacts: ``name → URI`` map.
        started_at: UTC timestamp.
        ended_at: UTC timestamp; ``None`` until `end_run`.
        status: One of ``"running"``, ``"completed"``, ``"failed"``.
    """

    run_id: str
    pipeline_hash: str
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    status: RunStatus = "running"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict snapshot (datetimes → isoformat)."""
        return {
            "run_id": self.run_id,
            "pipeline_hash": self.pipeline_hash,
            "config": dict(self.config),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "status": self.status,
        }


class RunMetrics:
    """Per-step metric series.

    Attributes:
        values: ``metric_name → [(step, value), …]``.
    """

    def __init__(self) -> None:
        self.values: dict[str, list[tuple[int, float]]] = {}

    def log(self, name: str, value: float, step: int) -> None:
        """Append ``(step, value)`` to the ``name`` series."""
        self.values.setdefault(name, []).append((int(step), float(value)))

    def latest(self) -> dict[str, float]:
        """Return ``{name: latest_value}`` across all series."""
        return {n: pairs[-1][1] for n, pairs in self.values.items() if pairs}

    def clear(self) -> None:
        self.values.clear()


class RunArtifacts:
    """Named artifacts (files, models, plots) attached to a `Run`.

    Attributes:
        items: ``name → URI``. URI is whatever the storage backend
            understands — a local path, an ``s3://`` URI, etc.
    """

    def __init__(self) -> None:
        self.items: dict[str, str] = {}

    def attach(self, name: str, uri: str) -> None:
        """Record ``uri`` under ``name``. Re-attaching overwrites."""
        self.items[name] = uri

    def get(self, name: str) -> str | None:
        return self.items.get(name)

    def clear(self) -> None:
        self.items.clear()
