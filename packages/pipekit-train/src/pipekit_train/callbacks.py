"""Callback protocol + bundled callbacks.

The five-hook callback surface and three bundled callbacks
(`EarlyStopping`, `Checkpoint`, `LogToExperiment`) per
``docs/design/api/components.md``.

The Callback Protocol is intentionally **not** ``@runtime_checkable``:
structural ``isinstance`` would reject partial implementations (a
metric-logger that only cares about ``on_step_end``), which
contradicts the design's optional-hook model. Adapters dispatch via
``getattr(cb, hook, None)``. See ADR D9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from pipekit import Operator


if TYPE_CHECKING:
    from pipekit import CarryState


# ---------------------------------------------------------------------
# Callback Protocol
# ---------------------------------------------------------------------


class Callback(Protocol):
    """Per-step / per-epoch / per-eval hooks.

    Each of the five hooks is **optional** at runtime. Adapters
    dispatch via ``getattr(cb, hook_name, None)`` so a callback that
    only cares about one hook (say, ``on_step_end`` for a metric
    logger) implements just that method.

    This Protocol is intentionally **not** ``@runtime_checkable``:
    structural ``isinstance`` would require every callback to
    implement all five methods, which contradicts the
    partial-implementation model documented in
    ``docs/design/api/components.md``. The Protocol exists as the
    type-checker contract describing the *full* hook surface; the
    runtime contract is "ducks with the methods you actually use".

    See ADR D9 in ``docs/design/decisions.md``.
    """

    def on_train_begin(self, loop: Any, state: CarryState) -> None: ...

    def on_step_end(
        self,
        loop: Any,
        state: CarryState,
        metrics: dict[str, float],
    ) -> None: ...

    def on_epoch_end(
        self,
        loop: Any,
        state: CarryState,
        metrics: dict[str, float],
    ) -> None: ...

    def on_eval_end(
        self,
        loop: Any,
        state: CarryState,
        eval_metrics: dict[str, float],
    ) -> None: ...

    def on_train_end(self, loop: Any, state: CarryState) -> None: ...


# ---------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------


class EarlyStopping(Operator):
    """Signal stop when a monitored metric hasn't improved for N evals.

    The callback only triggers on ``on_eval_end``. It updates internal
    "best so far" state and sets ``should_stop`` to True once
    ``patience`` consecutive eval rounds pass without a meaningful
    improvement (``min_delta``).

    Args:
        metric: Metric name as it appears in the on_eval_end dict.
        patience: Number of consecutive evals without improvement
            before signalling stop.
        mode: ``"min"`` for losses, ``"max"`` for accuracies.
        min_delta: Minimum change to count as improvement.

    Attributes (runtime state, not part of the operator config):
        should_stop: Read by the adapter's outer loop on each
            on_eval_end to decide whether to break early.
        best: Current best value of the monitored metric.
        wait: Consecutive eval rounds since last improvement.
    """

    # Runtime state lives on the operator but doesn't participate in
    # get_config (the dataclass-style discipline; tested below).
    __config_exclude__: ClassVar[tuple[str, ...]] = ("should_stop", "best", "wait")

    def __init__(
        self,
        metric: str,
        patience: int = 10,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(
                f"EarlyStopping.mode must be 'min' or 'max'; got {mode!r}."
            )
        if patience < 1:
            raise ValueError(f"EarlyStopping.patience must be >= 1; got {patience}.")
        if min_delta < 0:
            raise ValueError(
                f"EarlyStopping.min_delta must be non-negative; got {min_delta}."
            )
        self.metric = metric
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        # Runtime state. These are deliberately not in get_config.
        self.should_stop: bool = False
        self.best: float | None = None
        self.wait: int = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_eval_end(
        self,
        loop: Any,
        state: Any,
        eval_metrics: dict[str, float],
    ) -> None:
        del loop, state
        if self.metric not in eval_metrics:
            return  # silently skip; design choice — adapter need not error
        value = float(eval_metrics[self.metric])
        if self._is_improvement(value):
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True

    def reset(self) -> None:
        """Clear runtime state — useful for re-running the same loop."""
        self.should_stop = False
        self.best = None
        self.wait = 0

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            "EarlyStopping is a callback, not a pipeline operator. "
            "Pass it via TrainingLoop(callbacks=...)."
        )


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------


class Checkpoint(Operator):
    """Save (model_op, optimizer_state) every N steps; final to a registry.

    On each ``on_step_end``, when ``state.step`` is a multiple of
    ``every_n_steps``, the adapter is expected to save its native
    checkpoint into ``save_dir``. (The actual save call lives in the
    adapter; this callback just signals when to fire.)

    When ``registry`` is set, ``on_train_end`` stores the final model
    operator in the registry by content hash, and the URI is recorded
    on ``last_uri`` / ``last_hash`` for the loop to fold into its
    `TrainingArtifact`.

    Args:
        every_n_steps: Cadence for on-disk snapshots.
        keep_last: Number of recent checkpoints to retain on disk.
            Honored by the adapter's native checkpoint manager
            (Orbax `CheckpointManager.max_to_keep`, Lightning
            `ModelCheckpoint.save_top_k`).
        save_dir: Local directory or fsspec URI.
        registry: Optional anything satisfying
            ``pipekit_experiment.ModelRegistry``. When set, the final
            checkpoint is also stored by content hash and the URI is
            wired back into `TrainingArtifact.model_registry_uri` /
            `.trained_model_hash`.

    Attributes (runtime state):
        last_uri: Set after ``on_train_end`` if a registry was provided.
        last_hash: Set after ``on_train_end`` if a registry was provided.
    """

    __config_exclude__: ClassVar[tuple[str, ...]] = ("last_uri", "last_hash")
    # `registry` is a runtime object that doesn't round-trip cleanly.
    # We surface it as a constructor argument but mark the operator
    # forbid_in_yaml so YAML loaders refuse to round-trip a Checkpoint
    # that carries a non-serialisable registry handle.
    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        every_n_steps: int = 1000,
        keep_last: int = 3,
        save_dir: str = "./ckpts",
        registry: Any | None = None,
    ) -> None:
        if every_n_steps < 1:
            raise ValueError(
                f"Checkpoint.every_n_steps must be >= 1; got {every_n_steps}."
            )
        if keep_last < 1:
            raise ValueError(f"Checkpoint.keep_last must be >= 1; got {keep_last}.")
        self.every_n_steps = int(every_n_steps)
        self.keep_last = int(keep_last)
        self.save_dir = str(save_dir)
        self.registry = registry
        self.last_uri: str | None = None
        self.last_hash: str | None = None

    def should_snapshot(self, step: int) -> bool:
        """Pure predicate the adapter can call from on_step_end."""
        return step > 0 and step % self.every_n_steps == 0

    def on_train_end(self, loop: Any, state: Any) -> None:
        """Persist the final model to the registry, if one is configured."""
        del loop
        if self.registry is None:
            return
        model = getattr(state, "model", None)
        if model is None:
            return
        # `registry.store` is the pipekit_experiment.ModelRegistry protocol.
        # We don't import pipekit_experiment to keep this dependency
        # optional (`pipekit-train[experiment]`); duck-type instead.
        if not hasattr(self.registry, "store"):
            raise TypeError(
                "Checkpoint.registry must satisfy pipekit_experiment."
                "ModelRegistry (missing `.store`). Got "
                f"{type(self.registry).__name__}."
            )
        h = self.registry.store(model)
        self.last_hash = h
        self.last_uri = f"registry://{h}"

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            "Checkpoint is a callback, not a pipeline operator. "
            "Pass it via TrainingLoop(callbacks=...)."
        )


# ---------------------------------------------------------------------
# LogToExperiment
# ---------------------------------------------------------------------


class LogToExperiment(Operator):
    """Forward metrics / params / artifacts to a pipekit-experiment tracker.

    On ``on_train_begin``, calls ``tracker.start_run`` with the loop's
    config (plus any user-supplied extra config). On ``on_step_end`` /
    ``on_eval_end``, forwards metrics to ``tracker.log_metrics``. On
    ``on_train_end``, calls ``tracker.end_run``.

    Args:
        tracker: Anything satisfying
            ``pipekit_experiment.ExperimentTracker``.
        run_name: Name passed to ``tracker.start_run``.
        config: Extra config dict merged into the run start. The
            tracker also sees the TrainingLoop's ``get_config()``.

    Attributes (runtime state):
        run: The Run object returned by ``tracker.start_run``,
            available after ``on_train_begin``.
    """

    __config_exclude__: ClassVar[tuple[str, ...]] = ("run",)
    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        tracker: Any,
        run_name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        for method in ("start_run", "log_metrics", "end_run"):
            if not hasattr(tracker, method):
                raise TypeError(
                    "LogToExperiment.tracker must satisfy pipekit_experiment."
                    f"ExperimentTracker (missing `.{method}`). Got "
                    f"{type(tracker).__name__}."
                )
        self.tracker = tracker
        self.run_name = run_name
        self.config = dict(config) if config is not None else None
        self.run: Any = None

    def on_train_begin(self, loop: Any, state: Any) -> None:
        del state
        run_config: dict[str, Any] = {}
        if loop is not None and hasattr(loop, "get_config"):
            run_config.update(loop.get_config())
        if self.config is not None:
            run_config.update(self.config)
        self.run = self.tracker.start_run(name=self.run_name, config=run_config)

    def on_step_end(
        self,
        loop: Any,
        state: Any,
        metrics: dict[str, float],
    ) -> None:
        del loop
        if self.run is None or not metrics:
            return
        step = getattr(state, "step", 0)
        self.tracker.log_metrics(self.run, dict(metrics), step=step)

    def on_eval_end(
        self,
        loop: Any,
        state: Any,
        eval_metrics: dict[str, float],
    ) -> None:
        del loop
        if self.run is None or not eval_metrics:
            return
        step = getattr(state, "step", 0)
        tagged = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.tracker.log_metrics(self.run, tagged, step=step)

    def on_train_end(self, loop: Any, state: Any) -> None:
        del loop, state
        if self.run is None:
            return
        self.tracker.end_run(self.run, status="completed")

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            "LogToExperiment is a callback, not a pipeline operator. "
            "Pass it via TrainingLoop(callbacks=...)."
        )
