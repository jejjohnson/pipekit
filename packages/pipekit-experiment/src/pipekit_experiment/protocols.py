"""Two protocols — `ExperimentTracker` and `ModelRegistry`.

The seam between pipekit pipelines and external orchestration tools.
Adapters (MLflow, W&B, DVC, Hydra, Metaflow) satisfy these protocols
structurally — no inheritance from pipekit-experiment is required.

Both protocols are runtime-checkable.

See master plan Report 12, section 2.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from pipekit import Operator

    from pipekit_experiment.run import Run


@runtime_checkable
class ExperimentTracker(Protocol):
    """Log metrics, parameters, and artifacts from a training run.

    Implementations wrap external trackers — MLflow, W&B, ClearML,
    Neptune, ad-hoc CSV writers. Pipekit pipelines call these methods;
    the adapter translates to the tool's API.
    """

    def start_run(self, name: str, config: dict[str, Any]) -> Run: ...

    def log_metrics(self, run: Run, metrics: dict[str, float], step: int) -> None: ...

    def log_artifact(self, run: Run, path: str, name: str) -> None: ...

    def end_run(self, run: Run, status: str) -> None: ...


@runtime_checkable
class ModelRegistry(Protocol):
    """Store and retrieve trained models by content hash.

    Implementations: filesystem-backed, S3-backed, MLflow-registry-backed.
    The canonical identifier is the content hash; ``name`` is a
    mutable tag that resolves to a hash.
    """

    def store(
        self,
        model_op: Operator,
        *,
        name: str | None = None,
        tags: dict[str, Any] | None = None,
        weights: bytes | None = None,
    ) -> str:
        """Persist ``model_op`` (and optional weights). Return the hash."""

    def load(self, ref: str) -> Operator:
        """Resolve ``ref`` (hash or name) and return the rebuilt operator."""

    def list(self, *, tags: dict[str, Any] | None = None) -> list[str]:
        """Return content hashes, optionally filtered by tag values."""

    def tag(self, hash: str, name: str, *, force: bool = False) -> None:
        """Bind ``name`` to ``hash``. ``force=True`` allows re-tagging."""
