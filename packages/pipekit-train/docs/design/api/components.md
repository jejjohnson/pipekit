---
status: draft
version: 0.1.0
---

# Layer 2 — Components

Callbacks, metric writers, validation. The Layer-2 modules are
carrier-agnostic — they work for any adapter that drives the
hooks below.

---

## `Callback` Protocol

```python
@runtime_checkable
class Callback(Protocol):
    """Per-step / per-epoch / per-eval hooks.

    The adapter translates these calls into the backend's native
    callback API (Lightning: lit.Callback; Equinox: explicit hook
    points in the outer loop; Keras: keras.callbacks.Callback).
    """

    def on_train_begin(self, loop: TrainingLoop, state: CarryState) -> None: ...

    def on_step_end(
        self,
        loop: TrainingLoop,
        state: CarryState,
        metrics: dict[str, float],
    ) -> None: ...

    def on_epoch_end(
        self,
        loop: TrainingLoop,
        state: CarryState,
        metrics: dict[str, float],
    ) -> None: ...

    def on_eval_end(
        self,
        loop: TrainingLoop,
        state: CarryState,
        eval_metrics: dict[str, float],
    ) -> None: ...

    def on_train_end(self, loop: TrainingLoop, state: CarryState) -> None: ...
```

Five hooks. The merge between eqx_trainer's four hooks and the
master plan's three: `on_train_begin` + `on_train_end` at the ends,
`on_step_end` for every step, `on_epoch_end` for the (optional)
epoch boundaries, `on_eval_end` after each eval pass. Each hook is
optional — adapter dispatch uses `getattr(cb, hook, None)`.

## Bundled callbacks

All are `pipekit.Operator`s so they round-trip through YAML.

### `Checkpoint`

```python
class Checkpoint(Operator):
    """Save (model_op, optimizer_state) every N steps.

    With `registry` set, also stores to a ModelRegistry and records
    the model URI on the TrainingArtifact.

    Args:
        every_n_steps: Cadence.
        keep_last: Number of recent checkpoints to retain on disk.
        save_dir: Local directory or fsspec URI.
        registry: Optional pipekit-experiment.ModelRegistry; when set,
            the final checkpoint is also stored by content hash and
            the URI is wired back into TrainingArtifact.
    """
    every_n_steps: int = 1000
    keep_last: int = 3
    save_dir: str = "./ckpts"
    registry: Any | None = None     # pipekit_experiment.ModelRegistry
```

The adapter is responsible for the actual save. The Equinox adapter
uses Orbax `CheckpointManager.save_interval_steps`; Lightning uses
its built-in `ModelCheckpoint`. The `registry` field is the
pipekit-train ↔ pipekit-experiment bridge — it fills
`TrainingArtifact.model_registry_uri` and `trained_model_hash`.

### `EarlyStopping`

```python
class EarlyStopping(Operator):
    """Signal stop when a monitored metric hasn't improved for N evals.

    Args:
        metric: Metric name as it appears in the on_eval_end dict.
        patience: Number of consecutive evals without improvement
            before signalling stop.
        mode: "min" for losses, "max" for accuracies.
        min_delta: Minimum change to count as improvement.
    """
    metric: str
    patience: int = 10
    mode: Literal["min", "max"] = "min"
    min_delta: float = 0.0
```

The adapter checks the returned "should_stop" flag on each
`on_eval_end` and exits cleanly if set. Lightning's native
`EarlyStopping` callback is used directly when the Lightning adapter
is in play.

### `LogToExperiment`

The bridge to `pipekit-experiment`. Imports `pipekit_experiment`
lazily — only when this callback is constructed.

```python
class LogToExperiment(Operator):
    """Forward metrics / params / artifacts to a pipekit-experiment tracker.

    Args:
        tracker: Anything satisfying ExperimentTracker.
        run_name: Name passed to tracker.start_run.
        config: Extra config dict merged into the run start. The
            tracker also sees the TrainingLoop's get_config().
    """
    tracker: Any                    # pipekit_experiment.ExperimentTracker
    run_name: str
    config: dict[str, Any] | None = None
```

Per-step metrics flow through `on_step_end → tracker.log_metrics`.
Per-eval metrics flow through `on_eval_end → tracker.log_metrics`
(with a `phase="eval"` tag merged in). The final
`on_train_end` calls `tracker.end_run` and the resulting
`tracker_run_id` is stamped into `TrainingArtifact.tracker_run_id`.

This callback is the **only** point at which pipekit-train imports
`pipekit_experiment`. Users without `[experiment]` extras can omit
this callback and use a plain `MetricWriter` instead.

---

## `MetricWriter` Protocol

```python
@runtime_checkable
class MetricWriter(Protocol):
    """Where per-step metrics go when no tracker is attached.

    Implementations: JSONL, in-memory list, W&B, TensorBoard.
    The trainer calls write() per logged step.
    """
    def write(self, step: int, metrics: dict[str, float]) -> None: ...
    def close(self) -> None: ...
```

### `JSONLWriter`

```python
@dataclass
class JSONLWriter:
    """Append-only JSON Lines metric writer.

    Args:
        path: Output file path. Created if missing.
        flush_every: How many writes between fsync() calls.
    """
    path: str
    flush_every: int = 100

    def write(self, step: int, metrics: dict[str, float]) -> None: ...
    def close(self) -> None: ...
```

The default writer. No external dependency. Per-line schema:
`{"step": int, "metrics": dict[str, float], "timestamp": iso8601}`.
A W&B adapter is ten lines (see `examples/integration.md`).

---

## `ValidationStep`

```python
class ValidationStep(Operator):
    """Compute validation metrics on a held-out dataset.

    Used inside TrainingLoop on eval cadence, and standalone after
    training for the final-model evaluation report.

    Args:
        model_op: The (trained) model operator.
        dataset: The validation TrainingDataset.
        metrics: List of metric Operators (each takes (predicted,
            target) and returns a scalar). Composes with the rest of
            the pipekit ecosystem — pipekit-evaluate metrics, custom
            user metrics, …
    """
    model_op: Operator
    dataset: TrainingDataset
    metrics: tuple[Operator, ...]

    def __call__(self) -> dict[str, float]: ...
```

`metrics` is an `Operator` tuple, not callables, so the validation
step round-trips through YAML and the user can swap in
`pipekit-evaluate` metrics (when that package ships) without
changing the TrainingLoop config.
