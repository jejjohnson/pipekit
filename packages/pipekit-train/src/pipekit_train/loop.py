"""Layer 3 — TrainingLoop.

The headline composable. `TrainingLoop` is a
`pipekit.StatefulOperator` carrying ``(model, opt_state, step, epoch,
metrics, rng)``; running it produces ``(trained_model_op, artifact)``.

This module also ships the supporting `TrainerCarryState` (subclass of
`pipekit.CarryState`) and `ValidationStep` (an `Operator` that
evaluates a trained model against a held-out dataset).

See ``docs/design/api/models.md``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from importlib import import_module
from typing import Any, ClassVar, Literal

from pipekit import CarryState, Operator, StatefulOperator, dumps

from pipekit_train.dataset import TrainingDataset
from pipekit_train.loss import Loss
from pipekit_train.writer import MetricWriter


# ---------------------------------------------------------------------
# TrainerCarryState
# ---------------------------------------------------------------------


class TrainerCarryState(CarryState):
    """Carry-state for `TrainingLoop`.

    The optimizer state and RNG are backend-specific opaque types
    (an `optax.OptState` and a `jax.Array` PRNG key for the Equinox
    adapter; a `torch.optim.Optimizer.state_dict()` + tuple for
    Lightning). The pipekit-train side carries them as `Any` and
    delegates serialisation to the adapter's native checkpoint
    machinery.

    Attributes:
        model: Current model operator (possibly partially trained).
        opt_state: Backend-specific optimizer state.
        step: Global step counter.
        epoch: Derived from step // steps_per_epoch (0 if not set).
        metrics: Latest per-step metric snapshot (cleared on epoch end).
        rng: Backend-specific PRNG state.
    """

    def __init__(
        self,
        model: Operator,
        opt_state: Any,
        step: int = 0,
        epoch: int = 0,
        metrics: dict[str, float] | None = None,
        rng: Any = None,
    ) -> None:
        self.model = model
        self.opt_state = opt_state
        self.step = int(step)
        self.epoch = int(epoch)
        self.metrics = dict(metrics) if metrics is not None else {}
        self.rng = rng

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot.

        Backend-specific fields (``model``, ``opt_state``, ``rng``)
        are stored as repr strings — the adapter's native checkpoint
        is the round-trip path for those.
        """
        return {
            "model": repr(self.model),
            "opt_state": repr(self.opt_state),
            "step": self.step,
            "epoch": self.epoch,
            "metrics": dict(self.metrics),
            "rng": repr(self.rng),
        }


# ---------------------------------------------------------------------
# ValidationStep
# ---------------------------------------------------------------------


class ValidationStep(Operator):
    """Compute validation metrics on a held-out dataset.

    Iterates ``dataset``, calls ``model_op`` on each input, applies
    each metric `Operator` to ``(predicted, target)``, and returns
    the mean-over-batches metric values.

    Args:
        model_op: The model `Operator` (trained or otherwise).
        dataset: The held-out `TrainingDataset`.
        metrics: Tuple of `Operator`s. Each takes ``(predicted, target)``
            and returns either a scalar or ``(scalar, aux_dict)``; the
            scalar is averaged. The aux dict from `MSE` etc. is
            *also* averaged per-key.

    Returns from ``__call__``:
        ``dict[str, float]`` with one entry per metric (plus any aux
        keys), each averaged over the dataset.
    """

    __config_mixin_auto__ = False
    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        model_op: Operator,
        dataset: TrainingDataset,
        metrics: tuple[Operator, ...],
    ) -> None:
        if not isinstance(model_op, Operator):
            raise TypeError(
                "ValidationStep.model_op must be an Operator; got "
                f"{type(model_op).__name__}."
            )
        if not isinstance(dataset, TrainingDataset):
            raise TypeError(
                "ValidationStep.dataset must be a TrainingDataset; got "
                f"{type(dataset).__name__}."
            )
        if not metrics:
            raise ValueError("ValidationStep.metrics must be non-empty.")
        self.model_op = model_op
        self.dataset = dataset
        self.metrics = tuple(metrics)

    def _apply(self) -> dict[str, float]:
        sums: dict[str, float] = {}
        count = 0
        for x, y in self.dataset:
            predicted = self.model_op(x)
            for metric in self.metrics:
                result = metric(predicted, y)
                if isinstance(result, tuple):
                    _, aux = result
                    for k, v in aux.items():
                        sums[k] = sums.get(k, 0.0) + float(v)
                else:
                    name = type(metric).__name__
                    sums[name] = sums.get(name, 0.0) + float(result)
            count += 1
        if count == 0:
            return {}
        return {k: v / count for k, v in sums.items()}

    def get_config(self) -> dict[str, Any]:
        return {
            "model_op": type(self.model_op).__name__,
            "dataset": type(self.dataset).__name__,
            "metrics": [type(m).__name__ for m in self.metrics],
        }


# ---------------------------------------------------------------------
# TrainingArtifact (local dataclass; promoted to pipekit_experiment's
# TrainingArtifact when [experiment] is installed)
# ---------------------------------------------------------------------


@dataclass
class _LocalArtifact:
    """Drop-in stand-in for pipekit_experiment.TrainingArtifact.

    When ``pipekit-train[experiment]`` is installed,
    `TrainingLoop._build_artifact` returns a real
    `pipekit_experiment.TrainingArtifact`. Otherwise it returns this
    local dataclass (same field names) so users without the extra
    still get a structured return value.
    """

    training_pipeline_yaml: str
    dataset_hash: str
    trained_model_hash: str
    tracker_run_id: str | None
    model_registry_uri: str
    backend_info: dict[str, Any] = field(default_factory=dict)
    deps_lock: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# TrainingLoop
# ---------------------------------------------------------------------


_BACKEND_MODULES: dict[str, str] = {
    "equinox": "pipekit_train.adapters.equinox",
    "lightning": "pipekit_train.adapters.lightning",
    "keras": "pipekit_train.adapters.keras",
}


class TrainingLoop(StatefulOperator):
    """Train a ``model_op`` on a ``dataset``, producing a trained model_op.

    The headline composable. Delegates the actual training to a
    backend adapter (``pipekit_train.adapters.<backend>.run``). The
    adapter sees the full loop config and is responsible for the
    per-step gradient update and the per-eval validation pass. The
    `TrainingLoop` itself owns the artifact assembly and the callback
    threading.

    Args:
        model_op: Untrained model Operator. Backend-specific (an
            equinox/lightning/keras module wrapped in a
            pipekit.Operator).
        dataset: A `TrainingDataset`.
        val_dataset: Optional held-out `TrainingDataset` for eval.
        loss: A `Loss`. Ignored if ``task`` is supplied.
        task: Optional backend-native task interface. Lets power users
            bypass the auto-synthesised `Loss` path.
        optimizer_config: Flat dict, e.g.
            ``{"name": "adamw", "lr": 3e-4, "weight_decay": 0.01}``.
            Translated by the adapter into the backend's optimizer.
        max_steps: Total optimisation steps. (Per-step is the unit;
            see ADR D10.)
        steps_per_epoch: Optional. Controls when on_epoch_end fires.
        batch_size: Per-step minibatch size.
        backend: One of ``"equinox"``, ``"lightning"``, ``"keras"``.
        callbacks: Tuple of `Callback` instances.
        metric_writer: Optional `MetricWriter`.
        eval_every_n_steps: Validation cadence.
        log_every_n_steps: Metric logging cadence.
        checkpoint_dir: Optional. When set, the adapter wires up its
            native checkpoint machinery here.
        seed: Master PRNG seed.

    Raises (at run-time):
        ValueError: if neither ``loss`` nor ``task`` is supplied.
        ValueError: if ``backend`` isn't recognised.
    """

    __config_mixin_auto__ = False
    # The training loop carries closures (model_op, dataset, callbacks)
    # that often don't round-trip cleanly; mark accordingly.
    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        model_op: Operator,
        dataset: TrainingDataset,
        val_dataset: TrainingDataset | None = None,
        loss: Loss | None = None,
        task: Any | None = None,
        optimizer_config: dict[str, Any] | None = None,
        max_steps: int = 10_000,
        steps_per_epoch: int | None = None,
        batch_size: int = 32,
        backend: Literal["equinox", "lightning", "keras"] = "equinox",
        callbacks: tuple[Any, ...] = (),
        metric_writer: MetricWriter | None = None,
        eval_every_n_steps: int = 500,
        log_every_n_steps: int = 100,
        checkpoint_dir: str | None = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(model_op, Operator):
            raise TypeError(
                "TrainingLoop.model_op must be an Operator; got "
                f"{type(model_op).__name__}."
            )
        if not isinstance(dataset, TrainingDataset):
            raise TypeError(
                "TrainingLoop.dataset must be a TrainingDataset; got "
                f"{type(dataset).__name__}."
            )
        if val_dataset is not None and not isinstance(val_dataset, TrainingDataset):
            raise TypeError(
                "TrainingLoop.val_dataset must be a TrainingDataset or None; "
                f"got {type(val_dataset).__name__}."
            )
        if backend not in _BACKEND_MODULES:
            raise ValueError(
                f"TrainingLoop.backend must be one of "
                f"{sorted(_BACKEND_MODULES)}; got {backend!r}."
            )
        if max_steps < 1:
            raise ValueError(
                f"TrainingLoop.max_steps must be >= 1; got {max_steps}."
            )
        if batch_size < 1:
            raise ValueError(
                f"TrainingLoop.batch_size must be >= 1; got {batch_size}."
            )
        self.model_op = model_op
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.loss = loss
        self.task = task
        self.optimizer_config = dict(optimizer_config) if optimizer_config else {
            "name": "adamw",
            "lr": 3e-4,
        }
        self.max_steps = int(max_steps)
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = int(batch_size)
        self.backend = backend
        self.callbacks = tuple(callbacks)
        self.metric_writer = metric_writer
        self.eval_every_n_steps = int(eval_every_n_steps)
        self.log_every_n_steps = int(log_every_n_steps)
        self.checkpoint_dir = checkpoint_dir
        self.seed = int(seed)

    # --- StatefulOperator hook ----------------------------------------

    def _apply(
        self,
        carrier: Any = None,
        state: TrainerCarryState | None = None,
    ) -> tuple[Operator, TrainerCarryState]:
        """Run training; return ``(trained_model_op, final_state)``.

        v0.1 scope: ``carrier`` is ignored (the loop's own ``model_op``
        is always used). ``state`` is also ignored (resume happens
        through the adapter's native checkpoint manager via
        ``checkpoint_dir``). Sequential composition with prior-stage
        model handoff is documented but unsupported until v0.2.
        """
        del carrier, state
        trained_model_op, backend_info = self._run_adapter()
        final_state = TrainerCarryState(
            model=trained_model_op,
            opt_state=None,
            step=self.max_steps,
            epoch=(
                self.max_steps // self.steps_per_epoch if self.steps_per_epoch else 0
            ),
            metrics={},
            rng=None,
        )
        # The artifact build path uses _run_adapter's backend_info too.
        # Re-use the backend_info dict so .run() can pull it without
        # re-running adapter.run().
        self._last_backend_info = backend_info
        return trained_model_op, final_state

    # --- Convenience entry --------------------------------------------

    def run(self) -> tuple[Operator, Any]:
        """Run training end-to-end and return ``(trained_model_op, artifact)``.

        Equivalent to ``self._apply(None, None)`` followed by an
        artifact build. The artifact is a
        `pipekit_experiment.TrainingArtifact` when that package is
        installed; otherwise a local `_LocalArtifact` dataclass with
        the same field names.
        """
        trained_model_op, _ = self._apply()
        artifact = self._build_artifact(
            trained_model_op,
            getattr(self, "_last_backend_info", {}),
        )
        return trained_model_op, artifact

    # --- Adapter dispatch ---------------------------------------------

    def _run_adapter(self) -> tuple[Operator, dict[str, Any]]:
        """Resolve and call the configured backend adapter."""
        module_name = _BACKEND_MODULES[self.backend]
        module = import_module(module_name)
        return module.run(self)

    # --- Artifact assembly --------------------------------------------

    def _build_artifact(
        self,
        trained_model_op: Operator,
        backend_info: dict[str, Any],
    ) -> Any:
        """Assemble the `TrainingArtifact` (or `_LocalArtifact` fallback)."""
        # Pull tracker/registry references from callbacks (if any).
        tracker_run_id: str | None = None
        model_registry_uri: str = ""
        trained_model_hash: str = ""
        for cb in self.callbacks:
            run = getattr(cb, "run", None)
            if run is not None and hasattr(run, "__getitem__"):
                # LogToExperiment.run is a tracker-specific record.
                # We surface a stable string representation.
                tracker_run_id = repr(run)
            last_uri = getattr(cb, "last_uri", None)
            last_hash = getattr(cb, "last_hash", None)
            if last_uri:
                model_registry_uri = last_uri
            if last_hash:
                trained_model_hash = last_hash

        artifact_cls: Any
        try:
            artifact_cls = import_module(
                "pipekit_experiment.artifacts"
            ).TrainingArtifact
        except ImportError:
            artifact_cls = _LocalArtifact

        try:
            training_pipeline_yaml = json.dumps(dumps(self), indent=2, default=repr)
        except Exception:  # noqa: BLE001
            # forbid_in_yaml means we may not round-trip cleanly;
            # surface a JSON repr of get_config() instead.
            training_pipeline_yaml = json.dumps(
                self.get_config(), indent=2, default=repr
            )

        return artifact_cls(
            training_pipeline_yaml=training_pipeline_yaml,
            dataset_hash=self.dataset.content_hash(),
            trained_model_hash=trained_model_hash,
            tracker_run_id=tracker_run_id,
            model_registry_uri=model_registry_uri,
            backend_info=backend_info,
            deps_lock=_capture_deps_lock(),
            metadata={},
        )

    # --- Config / repr -------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        return {
            "model_op": type(self.model_op).__name__,
            "dataset": type(self.dataset).__name__,
            "dataset_hash": self.dataset.content_hash(),
            "val_dataset": (
                type(self.val_dataset).__name__ if self.val_dataset is not None else None
            ),
            "loss": type(self.loss).__name__ if self.loss is not None else None,
            "task": type(self.task).__name__ if self.task is not None else None,
            "optimizer_config": dict(self.optimizer_config),
            "max_steps": self.max_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "batch_size": self.batch_size,
            "backend": self.backend,
            "callbacks": [type(c).__name__ for c in self.callbacks],
            "eval_every_n_steps": self.eval_every_n_steps,
            "log_every_n_steps": self.log_every_n_steps,
            "checkpoint_dir": self.checkpoint_dir,
            "seed": self.seed,
        }


def _capture_deps_lock() -> str:
    """Best-effort capture of the env lock for reproducibility.

    Tries ``uv pip freeze`` first; falls back to empty string. The
    artifact stores the result verbatim — it's a reproducibility hint,
    not a hard contract.
    """
    try:
        result = subprocess.run(
            ["uv", "pip", "freeze"],  # noqa: S603, S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return ""
