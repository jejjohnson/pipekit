---
status: draft
version: 0.1.0
---

# Layer 3 — Models (The Training Loop)

`TrainingLoop` is the headline composable. It is a
`pipekit.StatefulOperator` carrying `(model, opt_state, step, epoch,
metrics, rng)`. Running it produces `(trained_model_op,
TrainingArtifact)`.

---

## `TrainerCarryState`

The state threaded through the loop, subclass of `pipekit.CarryState`.

```python
class TrainerCarryState(CarryState):
    """Carry-state for TrainingLoop.

    Attributes:
        model: The current model operator (possibly partially trained).
            Round-trips through ModelRegistry on checkpoint.
        opt_state: Backend-specific optimizer state. The Equinox
            adapter stores an optax.OptState; Lightning stores a
            torch.optim.Optimizer.state_dict(); Keras stores a
            dict of variable names → arrays.
        step: Global step counter.
        epoch: Derived from step // steps_per_epoch (0 if
            steps_per_epoch is None).
        metrics: Latest per-step metric snapshot — what `on_step_end`
            saw most recently. Cleared on `on_epoch_end`.
        rng: Backend-agnostic PRNG state. A (seed, counter) tuple
            for backends without native PRNG state, or the backend's
            native key for those that have one (JAX).
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
        self.step = step
        self.epoch = epoch
        self.metrics = metrics or {}
        self.rng = rng
```

`to_dict` / `from_dict` are inherited from `CarryState`; the
`opt_state` and `rng` fields are encoded by the adapter (which knows
their structure). Checkpoint round-trip therefore goes
`carry_state → dict → JSON file + per-adapter weight blob`.

---

## `TrainingLoop`

```python
class TrainingLoop(StatefulOperator):
    """Train a model_op on a dataset, producing a trained model_op.

    Delegates the actual training to a backend adapter
    (`pipekit_train.adapters.<backend>.run`). The adapter sees the
    full loop config and is responsible for the per-step gradient
    update and the per-eval validation pass.

    Args:
        model_op: Untrained model Operator. Typically wraps an
            equinox.Module, lightning.LightningModule, or
            keras.Model. Must be picklable / serializable through its
            backend-specific channel.
        dataset: A pipekit_train.TrainingDataset.
        val_dataset: Optional held-out TrainingDataset for eval.
        loss: A pipekit_train.Loss (or any callable satisfying the
            Protocol). Ignored if the user supplies a backend-native
            task interface via task=...
        task: Optional backend-native task interface — for Equinox,
            an instance of adapters.equinox.TrainTask. Lets power
            users bypass the auto-synthesised loss path.
        optimizer_config: Flat dict, e.g.
            {"name": "adamw", "lr": 3e-4, "weight_decay": 0.01}.
            Translated by the adapter into the backend's optimizer.
        max_steps: Total optimisation steps. (Per-step is the unit;
            see ADR D10.)
        steps_per_epoch: Optional. Controls when on_epoch_end fires.
            None means epoch boundaries are never reported.
        batch_size: Per-step minibatch size. The dataset's loader is
            wrapped with the appropriate Batch op.
        backend: One of "equinox", "lightning", "keras", and the planned
            Bayesian backends "numpyro-svi", "numpyro-mcmc", "blackjax".
            Maps to the adapter module.
        callbacks: Tuple of pipekit_train.Callback instances.
            Default: empty tuple. (Note: tuple, not list, to avoid
            mutable default + to preserve operator hashability.)
        metric_writer: Optional pipekit_train.MetricWriter. Falls
            back to a no-op writer when None.
        eval_every_n_steps: Validation cadence.
        log_every_n_steps: Metric logging cadence.
        checkpoint_dir: Optional. When set, the adapter wires up
            its native checkpoint machinery to this directory.
        seed: Master PRNG seed.
    """

    model_op: Operator
    dataset: TrainingDataset
    val_dataset: TrainingDataset | None = None
    loss: Loss | None = None
    task: Any | None = None
    optimizer_config: dict[str, Any] = field(default_factory=lambda: {"name": "adamw", "lr": 3e-4})
    max_steps: int = 10_000
    steps_per_epoch: int | None = None
    batch_size: int = 32
    backend: Literal[
        "equinox", "lightning", "keras",
        "numpyro-svi", "numpyro-mcmc", "blackjax",   # planned (Bayesian)
    ] = "equinox"
    callbacks: tuple[Callback, ...] = ()
    metric_writer: MetricWriter | None = None
    eval_every_n_steps: int = 500
    log_every_n_steps: int = 100
    checkpoint_dir: str | None = None
    seed: int = 0

    def _apply(self, carrier: Any, state: TrainerCarryState | None) -> tuple[Any, TrainerCarryState]:
        """StatefulOperator hook — delegates to adapter.run + advances state."""
        ...

    def run(self) -> tuple[Operator, TrainingArtifact | dict[str, Any]]:
        """Run the loop end-to-end, returning the trained model and artifact.

        Equivalent to `_apply(None, initial_state)` followed by an
        artifact build pass. Most users call this directly rather than
        composing TrainingLoop into a Sequential.
        """
        ...
```

### The outer loop (adapter-driven)

`TrainingLoop` does not itself call `train_step`. It delegates to:

```python
_BACKEND_MODULES = {
    "equinox": "pipekit_train.adapters.equinox",
    "lightning": "pipekit_train.adapters.lightning",
    "keras": "pipekit_train.adapters.keras",
    # planned Bayesian backends (see design/numpyro_adapter.md, blackjax_adapter.md):
    "numpyro-svi": "pipekit_train.adapters.numpyro_svi",
    "numpyro-mcmc": "pipekit_train.adapters.numpyro_mcmc",
    "blackjax": "pipekit_train.adapters.blackjax",
}

def run(self):
    adapter = import_module(_BACKEND_MODULES[self.backend])
    trained_model_op, backend_info = adapter.run(self)
    artifact = self._build_artifact(trained_model_op, backend_info)
    return trained_model_op, artifact
```

The adapter receives the full `TrainingLoop` and is responsible for:

1. Constructing the optimizer from `optimizer_config`.
2. Building the backend-native loader from `dataset` (and
   `val_dataset` if set).
3. Initialising the carry-state.
4. Running the per-step loop up to `max_steps`, calling callbacks at
   the documented hooks, evaluating every `eval_every_n_steps`.
5. Returning `(trained_model_op, backend_info_dict)`.

### Artifact assembly (`_build_artifact`)

`TrainingLoop` itself fills these fields:

- `training_pipeline_yaml` ← `pipekit.dumps(self)`
- `dataset_hash` ← `self.dataset.content_hash()`
- `backend_info` ← `backend_info` from adapter
- `deps_lock` ← best-effort capture of `uv.lock` or `pip freeze`

These fields come from callbacks:

- `tracker_run_id` ← `LogToExperiment` callback, if attached
- `model_registry_uri` ← `Checkpoint(registry=...)`, if attached
- `trained_model_hash` ← same callback path

If `pipekit-experiment` is not installed (no `[experiment]` extra),
`_build_artifact` returns a `dict` with the same field names instead
of a `TrainingArtifact`.

---

## `Sequential` composition example

Because `TrainingLoop` is a `StatefulOperator`, multi-stage training
is a single `Sequential`:

```python
pretrain = TrainingLoop(model_op=base_model, dataset=large_unlabeled, ...)
finetune = TrainingLoop(model_op=..., dataset=labeled, ...)

# Sequential threads carry-state (including model) through both stages
full = pipekit.Sequential([pretrain, finetune])
trained_model_op, artifact = full.run()
```

This is the deepest reason `TrainingLoop` is a `StatefulOperator`
and not a plain `Operator` — it lets pretrain → fine-tune ship as
one YAML file with one carry-state thread.
