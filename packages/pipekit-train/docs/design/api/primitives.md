---
status: draft
version: 0.1.0
---

# Layer 1 — Primitives

The carrier-agnostic loss surface, plus the per-adapter primitives
(`train_step`, `save_state`, `restore_state`) that are reused inside
the backend adapters.

---

## `Loss` Protocol

```python
@runtime_checkable
class Loss(Protocol):
    """Loss function. Per-batch scalar, called by the adapter.

    Implementations may return either a scalar loss directly, or a
    (scalar, aux_metrics_dict) tuple. The adapter distinguishes by
    introspection.
    """
    def __call__(
        self,
        predicted: Any,
        target: Any,
    ) -> float | tuple[float, dict[str, float]]: ...
```

Loss is the *contract* the adapter sees. The adapter is free to wrap
it in its own native form (Equinox: into a synthesised `TrainTask`;
Lightning: into `training_step`). Power users who want per-backend
custom training behaviour supply the adapter's native interface
directly and `Loss` is ignored.

## Common `Loss` operators

```python
class MSE(Operator):
    """Mean squared error. Carrier-agnostic over array-like leaves."""
    reduction: Literal["mean", "sum", "none"] = "mean"

    def __call__(self, predicted, target) -> float: ...

class NLL(Operator):
    """Negative log-likelihood for distributional outputs.

    Used when the model predicts a distribution (mean + variance) or
    when the model is a conditional density (amortized inference).
    The `predicted` carrier is expected to be either a tuple
    (mean, log_var) or an object with .log_prob(target).
    """
    def __call__(self, predicted, target) -> float: ...

class KL(Operator):
    """KL divergence — for variational training."""
    def __call__(self, q_dist, p_dist) -> float: ...

class Composite(Operator):
    """Weighted sum: loss = sum(w_i * L_i)."""
    components: tuple[tuple[float, Loss], ...]

    def __call__(self, predicted, target) -> tuple[float, dict[str, float]]:
        terms = {f"{type(L).__name__}_{i}": L(predicted, target)
                 for i, (w, L) in enumerate(self.components)}
        total = sum(w * v for (w, _), v in zip(self.components, terms.values()))
        return total, terms
```

Each is an `Operator` so it round-trips through YAML and composes
with the rest of pipekit (e.g. `Tap` for inspection,
`AssertShape` for QC, …).

---

## Per-adapter primitives — Equinox

The eqx_trainer primitives live inside the Equinox adapter
unchanged. They are documented here for completeness; Lightning and
Keras adapters define their own equivalents.

### `train_step` (Equinox)

The compiled inner loop. Pure function, `eqx.filter_jit`-wrapped.

```python
@eqx.filter_jit
def train_step(
    state: TrainState,
    batch: Any,
    key: jax.Array,
    task: TrainTask,
    optimizer: optax.GradientTransformation,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """One optimisation step. Pure function."""

    @eqx.filter_grad(has_aux=True)
    def grad_fn(model, batch, key):
        return task.loss_fn(model, batch, key)

    grads, metrics = grad_fn(state.model, batch, key)
    params = eqx.filter(state.model, eqx.is_array)
    updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
    new_model = eqx.apply_updates(state.model, updates)
    return (
        TrainState(model=new_model, opt_state=new_opt_state, step=state.step + 1),
        metrics,
    )
```

Key properties (ported from eqx_trainer `api/primitives.md`):

- `eqx.filter_grad` differentiates only array leaves — frozen params,
  activation functions, and config ignored automatically.
- `task` and `optimizer` are arguments, not closures, so they appear
  as static PyTree leaves to the tracer. Structural change forces a
  recompile; identical structure does not.
- The full `TrainState` is returned. No mutation; pure functional.

### `save_state` / `restore_state` (Equinox)

Orbax `CheckpointManager` ↔ Equinox `eqx.Module` bridge via
`eqx.partition`. Identical to the eqx_trainer primitive.

```python
def save_state(mngr: ocp.CheckpointManager, state: TrainState, step: int) -> None:
    arrays, static = eqx.partition(state, eqx.is_array)
    mngr.save(step, args=ocp.args.StandardSave(arrays))

def restore_state(
    mngr: ocp.CheckpointManager,
    template: TrainState,
    step: int,
) -> TrainState:
    arrays, static = eqx.partition(template, eqx.is_array)
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, arrays)
    restored = mngr.restore(step, args=ocp.args.StandardRestore(abstract))
    return eqx.combine(restored, static)
```

The pipekit-train extension on top of these is:

- The Grain iterator is also checkpointed. v0.2 ships a JSON
  side-car (`<directory>/<step>/data_iter.json`) written next to
  the Orbax checkpoint, capturing the iterator's `get_state()`
  return value (base64-encoded for bytes leaves). Preemption
  recovery restores both the weights AND the data-iter position
  via the side-car on `restore_state` → `train_iter.set_state(state)`.
  A full Orbax `CompositeCheckpointHandler` integration (using
  `grain.checkpoint.CheckpointHandler` natively) is a v0.3 polish.
- The `CheckpointManager.save_interval_steps` is wired to the
  pipekit-train `Checkpoint` callback's `every_n_steps` (Layer 2),
  so users configure cadence in one place.

---

## Per-adapter primitives — Lightning (v0.2 scaffold)

Lightning's idiom is `LightningModule.training_step` +
`LightningDataModule`. The Lightning adapter (when implemented)
synthesises both from `Loss` and `TrainingDataset`. The primitive
this adapter ships is `_build_lightning_module(loss, model_op)`
which returns a `LightningModule` whose `training_step` calls
`loss(model_op(x), y)`. Documented in
[adapters.md](adapters.md#lightning-v02).

## Per-adapter primitives — Keras (v0.3 scaffold)

Keras's idiom is `model.compile(loss=..., optimizer=...)` +
`model.fit(...)`. The Keras adapter (when implemented) maps `Loss`
to a `keras.losses.Loss` subclass and configures `model.compile` from
the rest of the `TrainingLoop` config. Documented in
[adapters.md](adapters.md#keras-v03).
