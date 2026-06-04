---
status: draft
version: 0.1.0
---

# Layer 4 — Backend Adapters

One module per backend under `pipekit_train.adapters.*`. Each
exports a single `run(loop) → (trained_model_op, backend_info)`
function. v0.1 ships the Equinox adapter as the reference; Lightning
and Keras are scaffolded.

---

## Equinox (v0.1 reference)

`pipekit_train.adapters.equinox`. Built on the `eqx_trainer` design:
Grain DataLoader, Orbax CheckpointManager, Optax optimiser,
`eqx.filter_jit` train step, `eqx.partition` checkpoint bridge.

Extras: `pipekit-train[equinox]`.

### Module surface

```python
# pipekit_train/adapters/equinox.py

class TrainState(eqx.Module):
    """Equinox-side training state. Stored inside TrainerCarryState.opt_state.

    A nested-state design: TrainerCarryState (the pipekit-side carry
    that everyone sees) holds an opaque opt_state field; for the
    Equinox adapter, that field is itself a TrainState. The split
    keeps the JAX-specific PyTree concerns inside the adapter and
    the carrier-agnostic concerns inside the loop.
    """
    model: eqx.Module
    opt_state: optax.OptState
    step: int

    @staticmethod
    def create(model: eqx.Module, optimizer: optax.GradientTransformation) -> "TrainState":
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        return TrainState(model=model, opt_state=opt_state, step=0)


@runtime_checkable
class TrainTask(Protocol):
    """User-defined Equinox training target. Mirrors eqx_trainer D4.

    Provide `loss_fn`; optionally provide `eval_fn`. The adapter
    synthesises a TrainTask from `loop.loss` if none is supplied.
    """
    def loss_fn(
        self,
        model: eqx.Module,
        batch: Any,
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]: ...

    def eval_fn(
        self,
        model: eqx.Module,
        batch: Any,
        key: jax.Array,
    ) -> dict[str, jax.Array]: ...


@eqx.filter_jit
def train_step(
    state: TrainState,
    batch: Any,
    key: jax.Array,
    task: TrainTask,
    optimizer: optax.GradientTransformation,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """One Equinox optimisation step (see api/primitives.md)."""


def save_state(mngr: ocp.CheckpointManager, state: TrainState, step: int) -> None: ...
def restore_state(mngr: ocp.CheckpointManager, template: TrainState, step: int) -> TrainState: ...


def run(loop: TrainingLoop) -> tuple[Operator, dict[str, Any]]:
    """Train `loop` end-to-end with the Equinox+Optax+Grain+Orbax stack.

    Returns:
        trained_model_op: A pipekit_jax.JaxModelOp (or domain
            subclass) wrapping the trained eqx.Module.
        backend_info: Dict with keys: backend ("equinox"), jax_version,
            equinox_version, optax_version, devices, total_seconds.
    """
```

### What `run` does

1. **Resolve the optimizer.** `optimizer_config = {"name": "adamw",
   "lr": 3e-4, ...}` → `optax.chain(optax.adamw(**kwargs), ...)`.
   Schedule support: if `lr` is a dict like `{"schedule":
   "cosine_decay", "init_value": 3e-4, "decay_steps":
   loop.max_steps}`, build an `optax.cosine_decay_schedule` and
   nest the optimiser inside `optax.inject_hyperparams`.
2. **Resolve the task.** If `loop.task` is set, use it. Otherwise
   synthesise: `class SynthTask: def loss_fn(self, model, batch,
   key): pred = jax.vmap(model)(batch["x"]); l = loop.loss(pred,
   batch["y"]); return l, {"loss": l}`.
3. **Build the data iterator.** Wrap `loop.dataset` as a
   `grain.MapDataset` source if it isn't already one; add a
   `grain.Batch(loop.batch_size)` operation; wrap in a
   `grain.DataLoader(worker_count=2, ...)`. Same for
   `loop.val_dataset`.
4. **Set up the checkpoint manager.** If `loop.checkpoint_dir`,
   build an `ocp.CheckpointManager` with
   `save_interval_steps=loop.callbacks[Checkpoint].every_n_steps`
   and `max_to_keep=loop.callbacks[Checkpoint].keep_last`, using
   a `CompositeCheckpointHandler` so the Grain iterator state
   checkpoints alongside the model.
5. **Initialise `TrainState` and `TrainerCarryState`.** Pull the
   model from `loop.model_op` (it must be a `JaxModelOp` or a
   subclass exposing `.module` returning the bare `eqx.Module`).
   Build the carry-state with the right RNG split.
6. **Resume.** If checkpoint manager has a `latest_step`, restore
   `TrainState` + Grain iterator state.
7. **Run the per-step loop.** For each step up to `loop.max_steps`:
   call `train_step`; periodically log (`log_every_n_steps`),
   evaluate (`eval_every_n_steps`), checkpoint, run callbacks.
8. **Build the trained `model_op`.** Wrap the final `eqx.Module`
   back into a `JaxModelOp` (or the original `model_op`'s class) and
   return it together with `backend_info`.

### Sharding (multi-device / multi-host) — v0.2

Pass an optional `ShardingSpec` to enable data-parallel (and uniform
model-parallel) training (issue #15, ADR D12):

```python
import jax
from jax.sharding import Mesh, PartitionSpec
from pipekit_train import TrainingLoop
from pipekit_train.adapters.equinox import ShardingSpec

mesh = Mesh(jax.devices(), axis_names=("data",))
spec = ShardingSpec(
    mesh=mesh,
    model_pspec=PartitionSpec(),        # replicate the model on every device
    data_pspec=PartitionSpec("data"),   # shard each batch along its leading axis
)

loop = TrainingLoop(
    model_op=model_op,
    dataset=dataset,
    loss=loss,
    batch_size=256,                     # must be divisible by the "data" axis size
    sharding=spec,                      # None => single-device (the default)
)
trained_op, artifact = loop.run()
```

How it threads through `run`:

- **`TrainState`** — model + optimiser array leaves are placed on
  `spec.model_sharding`; the step counter is replicated.
- **`train_step`** — unchanged; `eqx.filter_jit` compiles sharding-aware
  once its inputs are sharded (replicated model + a `data`-sharded batch
  give data-parallel SPMD, XLA inserting the collectives).
- **Data iterator** — each batch is placed on `spec.data_sharding`. For
  multi-host (an explicit spec **and** `process_count > 1`), each process
  reads a disjoint shard of the `grain` `MapDataset`
  (`map_ds[process_index::process_count]`, the modern equivalent of
  `grain.ShardByJaxProcess`) and batches the *local* size
  `batch_size // process_count`, so the per-process batches assemble via
  `jax.make_array_from_process_local_data` into a global batch of exactly
  `batch_size`. Without a spec the single-device path is preserved even
  under a distributed-initialised JAX. (`batch_size` must divide
  `process_count`; multi-host needs an indexable dataset, not a streaming
  one.)
- **Checkpointing** — Orbax restore is sharding-aware automatically: the
  restore template is the already-sharded `TrainState`.
- **`backend_info["sharding"]`** records the mesh shape and partition
  specs for the artifact.

Verified on a CPU-simulated 2-device mesh (`tests/adapters/
test_equinox_sharding.py`) and a scripted 2-process `jax.distributed` smoke
(`scripts/multihost_sharding_smoke.py`).

### Notes

- `sharding=None` (the default) is the original single-host, single-device
  path, unchanged.
- The adapter does not require the user to install `pipekit-jax`. If
  it isn't installed, the adapter returns a minimal `JaxModelOp`-like
  wrapper defined inline (the wrapper just calls the `eqx.Module`'s
  `__call__`). The full `pipekit-jax.JaxModelOp` adds shape inference
  and weight-blob round-trip.
- The Grain iterator is the *only* JAX-side data-loading dependency.
  PyTorch DataLoader is explicitly not used.

---

## Lightning (v0.2 scaffold)

`pipekit_train.adapters.lightning`. Extras:
`pipekit-train[lightning]`.

v0.1 ships the module with a single function that raises:

```python
def run(loop: TrainingLoop) -> tuple[Operator, dict[str, Any]]:
    raise NotImplementedError(
        "The Lightning adapter is scheduled for v0.2. Use "
        "backend='equinox' for v0.1, or pin pipekit-train when "
        "v0.2 lands."
    )
```

### Planned design

When v0.2 lands, the adapter will:

1. Synthesise a `LightningModule` from `loop.model_op` + `loop.loss`
   (or accept a user-supplied `task` that is itself a
   `LightningModule` factory).
2. Synthesise a `LightningDataModule` from `loop.dataset` /
   `loop.val_dataset`.
3. Build a `Trainer` with `max_steps=loop.max_steps`, mapping
   pipekit callbacks to Lightning's `pl.Callback` subclass surface
   (e.g. `LogToExperiment → MLFlowLogger` or `WandbLogger`).
4. Call `trainer.fit(module, datamodule)`.
5. Wrap the final module's `nn.Module` in `pipekit-array.ModelOp` and
   return.

The Lightning adapter inherits all of Lightning's distributed,
mixed-precision, and accumulation features without pipekit-train
adding any code — they're configured via `optimizer_config` or
`extra_trainer_kwargs`.

---

## Keras (v0.3 scaffold)

`pipekit_train.adapters.keras`. Extras: `pipekit-train[keras]`.

v0.1 ships with the same `NotImplementedError` stub.

### Planned design

When v0.3 lands, the adapter will:

1. Translate `loop.loss` into a `keras.losses.Loss` subclass (or
   accept the user's pre-built `tf.keras.losses` instance via
   `loop.task`).
2. Configure the optimiser via `keras.optimizers.get` using
   `optimizer_config`.
3. Wrap `loop.dataset` as a `tf.data.Dataset.from_generator(...)`.
4. Call `model.compile(...)` then `model.fit(...)`.
5. Wrap the trained `keras.Model` in `pipekit-array.ModelOp` and
   return.

Keras 3's multi-backend (TF / JAX / torch) means the Keras adapter
gives a *second* JAX path orthogonal to the Equinox one — useful for
projects that already standardise on Keras layers.

---

## NumPyro (planned) — Bayesian inference as a backend

`pipekit_train.adapters.numpyro`. Extras: `pipekit-train[numpyro]`
(`numpyro>=0.15`, `optax`, `grain`; JAX comes via NumPyro). See ADR D13.

### Why it fits

"Training" in NumPyro is **Bayesian inference**, and it lands on the same
adapter contract as every other backend — a module exposing
`run(loop) -> (trained_model_op, backend_info)`, registered in
`_BACKEND_MODULES` + the `backend` `Literal`, gated by an extra (D5). It
also closes a loop with the benchmark ladder (`pipekit-evaluate`): NumPyro
**NUTS** is the "model-based inference / rung-2 oracle" and **SVI** is the
"emulator-based inference / rung-4" — so those estimators become trainable
through the same `TrainingLoop`. (`plumax`, `xtremax`, `somax` all use
NumPyro for their oracles.)

### The one real difference: task-first, not loss-first

The carrier-agnostic `Loss(pred, target)` (D4) does **not** map to NumPyro —
its objective is the **ELBO** (SVI) or the joint log-density (NUTS) over a
probabilistic `model(x, y=None)`, defined by the model itself. So the
NumPyro adapter is **task-first**: it requires `loop.task` (a NumPyro model
+ guide), which D9 ("TrainTask Protocol per backend") already anticipates.
Passing only `loss=` raises a clean error (there is no `_SynthesisedTask`
analog).

```python
@dataclass
class NumpyroTask:
    """User-defined NumPyro inference target (loop.task for backend='numpyro')."""
    model: Callable          # def model(x, y=None): numpyro.sample(...)
    guide: Callable | None = None        # default: AutoNormal(model) for SVI
    method: Literal["svi", "nuts"] = "svi"
    loss: Any = None                     # default: numpyro.infer.Trace_ELBO()
    num_warmup: int = 1000               # NUTS only
    num_samples: int = 1000              # NUTS only
    num_chains: int = 1                  # NUTS only
```

### Two inference modes, one adapter (method-dispatched)

**SVI — the per-step path (primary).** `svi.update(state, *batch)` is *one
gradient step returning `(state, loss)`*, so it drops straight into the
existing per-step loop — callbacks, `metric_writer`, periodic eval
(`svi.evaluate`), and checkpoint cadence all work unchanged. NumPyro's
`SVI(optim=...)` accepts an **optax `GradientTransformation` directly**, so
`optimizer_config` reuses the Equinox translation. Minibatching uses
`numpyro.plate(subsample_size=batch_size)` fed by the existing Grain
iterator (full-batch is the v1 default).

**NUTS — the single-call path (oracle).** `mcmc.run(key)` is one blocking
call (warmup + sampling), not a per-gradient loop. The adapter runs it as a
single `fit`, firing `on_train_begin` / `on_train_end` only; `max_steps` is
ignored in favour of the task's `num_warmup` / `num_samples`. Diagnostics
(`num_divergences`, `r_hat` summary) go into `backend_info`.

### What `run` does

1. **Validate the task.** Require `loop.task` to be a `NumpyroTask`
   (error if only `loop.loss` is set). Default `guide=AutoNormal(model)`
   for SVI; `loss=Trace_ELBO()`.
2. **Build the data iterator.** Reuse `_BatchSource` (Grain / streaming)
   exactly as the Equinox adapter does; batches are `(x, y)` arrays.
3. **Dispatch on `task.method`:**
   - **svi:** `svi = SVI(model, guide, optax_optim, loss)`;
     `state = svi.init(key, *first_batch)`; per-step `svi.update`; log the
     ELBO; eval via `svi.evaluate` on `val_dataset`; checkpoint the
     `SVIState` params (Orbax).
   - **nuts:** `mcmc = MCMC(NUTS(model), num_warmup, num_samples,
     num_chains)`; `mcmc.run(key, X, y)`; collect `mcmc.get_samples()`.
4. **Wrap the trained artifact.** Return a `NumpyroPredictiveOp`
   (`pipekit.Operator`, `forbid_in_yaml=True`) carrying
   `Predictive(model, guide=guide, params=svi_params)` (SVI) or
   `Predictive(model, posterior_samples=samples)` (NUTS). `_apply(x)`
   returns the posterior-predictive mean by default (samples optionally).
   The weight-blob round-trip through the model registry is the
   params / posterior-samples PyTree — directly analogous to
   `pipekit-jax.JaxModelOp`.

### Mapping summary

| `TrainingLoop` field | SVI | NUTS |
|----------------------|-----|------|
| `task` (required) | `NumpyroTask(model, guide, method="svi")` | `…method="nuts", num_warmup, num_samples` |
| `loss` | N/A — objective is the model's ELBO | N/A — joint log-density |
| `optimizer_config` | → optax → `SVI(optim=...)` | ignored |
| `max_steps` | SVI steps | → `num_samples` |
| `batch_size` | `plate(subsample_size=…)` minibatch | full-batch (subsample = HMCECS) |
| per-step loop / callbacks | `svi.update` + `svi.evaluate` | one `mcmc.run`, begin/end only |
| `checkpoint_dir` | `SVIState` params (Orbax) | posterior samples |
| trained `model_op` | `NumpyroPredictiveOp(Predictive(model, guide, params))` | `…(Predictive(model, posterior_samples))` |
| `backend_info` | numpyro/jax ver, method, final ELBO | num_warmup/samples, divergences, r̂ |

### Build order (when implementation lands)

```
1. [numpyro] extra + registry/Literal entry + scaffold run() raising
   NotImplementedError (matches lightning/keras scaffolds).
2. NumpyroTask + NumpyroPredictiveOp + task-vs-loss validation.
3. SVI per-step path (callbacks/writer/eval/checkpoint reuse the loop).
   -> test: Bayesian linear regression, ELBO decreases, slope recovered.
4. NUTS single-call path + diagnostics in backend_info.
   -> test: NUTS posterior covers the true slope; predictive op round-trips.
5. Minibatch SVI via numpyro.plate; docs + ADR finalised.
```

Tests gate on `importorskip("numpyro")`, like the Equinox suite.

---

## Adapter selection

`TrainingLoop.run()` selects the adapter by string:

```python
loop = TrainingLoop(..., backend="equinox")   # uses adapters.equinox
loop = TrainingLoop(..., backend="lightning") # uses adapters.lightning
loop = TrainingLoop(..., backend="keras")     # uses adapters.keras
loop = TrainingLoop(..., backend="numpyro")   # uses adapters.numpyro (planned)
```

The selection happens at `run()` time, not at construction, so a
`TrainingLoop` YAML can be authored with `backend="lightning"` even
on a machine that hasn't installed `[lightning]` — the missing extra
is reported at run time only.
