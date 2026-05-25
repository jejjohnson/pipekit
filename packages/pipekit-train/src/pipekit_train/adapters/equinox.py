"""Equinox + Optax + Orbax adapter — the v0.1 reference.

Concrete JAX-native stack as described in
``docs/design/api/adapters.md`` (Equinox section). The adapter
synthesises a ``TrainTask`` from a `pipekit_train.Loss` if the user
doesn't supply one, builds the optimiser from ``optimizer_config``,
runs an ``eqx.filter_jit``'d train step, and threads the user's
callbacks through the loop.

Module surface:

- `TrainState(eqx.Module)` — model + opt_state + step.
- `TrainTask` (Protocol) — user-defined training target, ported from
  the eqx_trainer design.
- `train_step` — pure function, ``@eqx.filter_jit`` wrapped.
- `save_state` / `restore_state` — Orbax bridge via ``eqx.partition``.
- `_EquinoxModelOp(Operator)` — wraps a trained ``eqx.Module`` as a
  `pipekit.Operator`. v0.1 in-package replacement for the future
  `pipekit-jax.JaxModelOp`.
- `run(loop)` — full end-to-end training entry point.

See ADRs D8, D9, D10 in ``docs/design/decisions.md``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, runtime_checkable

import equinox as eqx  # ty: ignore[unresolved-import]
import jax  # ty: ignore[unresolved-import]
import jax.numpy as jnp  # ty: ignore[unresolved-import]
import numpy as np
import optax  # ty: ignore[unresolved-import]
import orbax.checkpoint as ocp  # ty: ignore[unresolved-import]
from pipekit import Operator


if TYPE_CHECKING:
    from pipekit_train.loop import TrainingLoop


# ---------------------------------------------------------------------
# _EquinoxModelOp — pipekit.Operator wrapper around an eqx.Module
# ---------------------------------------------------------------------


class _EquinoxModelOp(Operator):
    """Wrap an `eqx.Module` as a `pipekit.Operator`.

    Drop-in replacement for the future ``pipekit-jax.JaxModelOp``.
    Same constructor signature; same ``_apply`` shape (calls the
    module on the input). When ``pipekit-jax`` ships, users swap the
    import and everything still works.

    The wrapper is marked ``forbid_in_yaml = True`` because an
    ``eqx.Module`` is a JAX PyTree of arrays — the round-trip path is
    the model registry's ``weights`` blob plus the module's
    structural config, not JSON of the operator itself.

    Args:
        module: An ``eqx.Module``.
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(self, module: eqx.Module) -> None:
        if not isinstance(module, eqx.Module):
            raise TypeError(
                "_EquinoxModelOp.module must be an eqx.Module; got "
                f"{type(module).__name__}."
            )
        self.module = module

    def _apply(self, x: Any) -> Any:
        # eqx.Module is callable (instance method __call__), but ty
        # doesn't know that statically.
        return cast(Any, self.module)(x)

    def get_config(self) -> dict[str, Any]:
        return {"module_class": type(self.module).__name__}


# ---------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------


class TrainState(eqx.Module):
    """Equinox-side training state.

    A PyTree-clean container holding the bare ``eqx.Module``
    (without the ``_EquinoxModelOp`` wrapper), the Optax optimizer
    state, and the global step counter.
    """

    model: eqx.Module
    opt_state: optax.OptState
    step: jax.Array

    @staticmethod
    def create(
        model: eqx.Module,
        optimizer: optax.GradientTransformation,
    ) -> TrainState:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        return TrainState(
            model=model,
            opt_state=opt_state,
            step=jnp.asarray(0, dtype=jnp.int32),
        )


# ---------------------------------------------------------------------
# TrainTask Protocol + synthesis from carrier-agnostic Loss
# ---------------------------------------------------------------------


@runtime_checkable
class TrainTask(Protocol):
    """User-defined Equinox training target.

    Provide ``loss_fn`` (required); optionally provide ``eval_fn``.
    The adapter synthesises a `TrainTask` from ``loop.loss`` when no
    explicit task is supplied.

    Signatures:
        loss_fn(model, batch, key) -> (scalar_loss, aux_metrics_dict)
        eval_fn(model, batch, key) -> metrics_dict  (optional)
    """

    def loss_fn(
        self,
        model: eqx.Module,
        batch: Any,
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]: ...


class _SynthesisedTask:
    """Adapt a `pipekit_train.Loss` into a `TrainTask`.

    Expects batches as ``(stacked_x, stacked_y)`` tuples. Calls
    ``jax.vmap(model)(x)`` so the model is written per-sample and
    the adapter handles batching. Forwards the loss's ``(scalar,
    aux_dict)`` shape through.
    """

    def __init__(self, loss: Any) -> None:
        self.loss = loss

    def loss_fn(
        self,
        model: eqx.Module,
        batch: tuple[jax.Array, jax.Array],
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del key
        x, y = batch
        pred = jax.vmap(model)(x)
        result = self.loss(pred, y)
        if isinstance(result, tuple):
            scalar, aux = result
            aux = dict(aux)
        else:
            scalar, aux = result, {"loss": result}
        # JAX grad requires a 0-D scalar, and the outer training loop
        # calls float() on every aux entry; both will crash if a
        # `Loss` configured with ``reduction='none'`` (or otherwise
        # returning a non-scalar array) flows through. Reduce
        # everything to a scalar here so the synthesised path is
        # robust for all common Loss shapes.
        scalar = _reduce_to_scalar(scalar)
        aux = {k: _reduce_to_scalar(v) for k, v in aux.items()}
        return scalar, aux


def _reduce_to_scalar(x: Any) -> jax.Array:
    """Cast to jax.Array and average down to a 0-D scalar if needed."""
    x = jnp.asarray(x)
    if x.ndim > 0:
        x = jnp.mean(x)
    return x


# ---------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------


@eqx.filter_jit
def train_step(
    state: TrainState,
    batch: Any,
    key: jax.Array,
    task: TrainTask,
    optimizer: optax.GradientTransformation,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """One Equinox optimisation step. Pure function.

    Differentiates only array leaves (`eqx.filter_grad`); passes
    ``task`` and ``optimizer`` as static arguments so structural
    changes recompile.
    """

    @eqx.filter_grad(has_aux=True)
    def grad_fn(
        model: eqx.Module,
        batch: Any,
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        return task.loss_fn(model, batch, key)

    grads, metrics = grad_fn(state.model, batch, key)
    params = eqx.filter(state.model, eqx.is_array)
    updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
    new_model = eqx.apply_updates(state.model, updates)
    return (
        TrainState(model=new_model, opt_state=new_opt_state, step=state.step + 1),
        metrics,
    )


# ---------------------------------------------------------------------
# Orbax bridge — save_state / restore_state via eqx.partition
# ---------------------------------------------------------------------


def save_state(
    mngr: ocp.CheckpointManager,
    state: TrainState,
    step: int,
) -> None:
    """Persist the array leaves of ``state`` at ``step``.

    Static (non-array) leaves are reconstructed on restore from the
    in-memory template; see ``restore_state``.
    """
    arrays, _ = eqx.partition(state, eqx.is_array)
    mngr.save(step, args=ocp.args.StandardSave(arrays))


def restore_state(
    mngr: ocp.CheckpointManager,
    template: TrainState,
    step: int,
) -> TrainState:
    """Restore array leaves into the static skeleton of ``template``."""
    arrays, static = eqx.partition(template, eqx.is_array)
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, arrays)
    restored = mngr.restore(step, args=ocp.args.StandardRestore(abstract))
    return cast(TrainState, eqx.combine(restored, static))


# ---------------------------------------------------------------------
# Optimiser construction
# ---------------------------------------------------------------------


_OPTIMIZER_CONSTRUCTORS: dict[str, Any] = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "sgd": optax.sgd,
    "rmsprop": optax.rmsprop,
}


def _build_optimizer(config: dict[str, Any]) -> optax.GradientTransformation:
    """Translate ``optimizer_config`` into an `optax.GradientTransformation`.

    Supported names: ``adam``, ``adamw``, ``sgd``, ``rmsprop``. The
    rest of the config is forwarded as kwargs to the constructor.
    Conveniences:

    - ``lr`` is normalised to ``learning_rate`` (the design uses the
      shorter form; optax uses the longer one).
    """
    config = dict(config)
    name = config.pop("name", "adamw")
    if "lr" in config and "learning_rate" not in config:
        config["learning_rate"] = config.pop("lr")
    if name not in _OPTIMIZER_CONSTRUCTORS:
        raise ValueError(
            f"Unknown optimizer {name!r}. v0.1 supports "
            f"{sorted(_OPTIMIZER_CONSTRUCTORS)}."
        )
    return _OPTIMIZER_CONSTRUCTORS[name](**config)


# ---------------------------------------------------------------------
# Data iteration — Grain or direct fallback
# ---------------------------------------------------------------------


def _is_indexable(dataset: Any) -> bool:
    return hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")


def _stack_pair(samples: list[tuple[Any, Any]]) -> tuple[jax.Array, jax.Array]:
    xs = jnp.stack([jnp.asarray(s[0]) for s in samples])
    ys = jnp.stack([jnp.asarray(s[1]) for s in samples])
    return xs, ys


def _iter_batches(
    dataset: Any,
    batch_size: int,
    seed: int,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    """Yield (stacked_x, stacked_y) batches, infinite-loop with reshuffle.

    Uses Grain's MapDataset when the dataset is indexable, otherwise
    falls back to direct Python iteration with manual batching
    (single epoch then re-iteration).
    """
    if _is_indexable(dataset):
        yield from _iter_indexable(dataset, batch_size, seed)
    else:
        yield from _iter_streaming(dataset, batch_size)


def _iter_indexable(
    dataset: Any,
    batch_size: int,
    seed: int,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    """Sequence-based iteration with epoch shuffle.

    Grain's MapDataset.source(seq) imports the heavy data loader; for
    v0.1 with toy-scale datasets we just shuffle indices via numpy and
    iterate. (Grain's worker_count / prefetch are v0.2 — see Q7 in
    boundaries.md.)

    Raises:
        ValueError: when ``len(dataset) < batch_size``. With a fixed
            batch_size we drop trailing partials so the JIT'd
            train_step doesn't recompile per epoch boundary; if the
            dataset is smaller than one batch the iterator would
            yield nothing and the training loop would hang waiting
            for ``next(train_iter)``. Fail fast instead.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    if n < batch_size:
        raise ValueError(
            f"Dataset has {n} samples but batch_size={batch_size}. "
            "Reduce batch_size, grow the dataset, or wrap as "
            "IterableDataset for streaming (which doesn't have this "
            "constraint)."
        )
    indices = np.arange(n)
    while True:
        rng.shuffle(indices)
        batch: list[tuple[Any, Any]] = []
        for idx in indices:
            batch.append(dataset[int(idx)])
            if len(batch) == batch_size:
                yield _stack_pair(batch)
                batch = []
        # Drop the partial trailing batch — keeps batch_size constant
        # so train_step doesn't recompile per epoch boundary.


def _iter_streaming(
    dataset: Any,
    batch_size: int,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    """Direct iteration for non-indexable datasets (e.g. `IterableDataset`)."""
    while True:
        batch: list[tuple[Any, Any]] = []
        for x, y in dataset:
            batch.append((x, y))
            if len(batch) == batch_size:
                yield _stack_pair(batch)
                batch = []


# ---------------------------------------------------------------------
# Eval — runs the validation dataset through the model, returns metrics
# ---------------------------------------------------------------------


def _evaluate(
    model: eqx.Module,
    val_dataset: Any,
    batch_size: int,
    task: TrainTask,
) -> dict[str, float]:
    """Run model on val_dataset, return averaged metric dict."""
    sums: dict[str, float] = {}
    n = 0
    # We always do a single-pass (one epoch) over the val dataset.
    # Use direct iteration; the loss_fn we already have can score
    # mini-batches as if they were training batches.
    iterator = _eval_pass(val_dataset, batch_size)
    for batch in iterator:
        _, aux = task.loss_fn(model, batch, jax.random.key(0))
        for k, v in aux.items():
            sums[k] = sums.get(k, 0.0) + float(np.asarray(v))
        n += 1
    if n == 0:
        return {}
    return {k: v / n for k, v in sums.items()}


def _eval_pass(
    dataset: Any,
    batch_size: int,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    """One pass over the validation dataset."""
    batch: list[tuple[Any, Any]] = []
    for x, y in dataset:
        batch.append((x, y))
        if len(batch) == batch_size:
            yield _stack_pair(batch)
            batch = []
    if batch:
        yield _stack_pair(batch)


# ---------------------------------------------------------------------
# Callback dispatch helpers
# ---------------------------------------------------------------------


def _dispatch(callbacks: tuple[Any, ...], hook: str, *args: Any) -> None:
    """Call ``hook`` on each callback that implements it."""
    for cb in callbacks:
        fn = getattr(cb, hook, None)
        if fn is not None:
            fn(*args)


def _any_should_stop(callbacks: tuple[Any, ...]) -> bool:
    """True if any callback signals ``should_stop`` (e.g. EarlyStopping)."""
    return any(getattr(cb, "should_stop", False) for cb in callbacks)


# ---------------------------------------------------------------------
# run — the orchestration entry point
# ---------------------------------------------------------------------


def run(loop: TrainingLoop) -> tuple[Operator, dict[str, Any]]:
    """Train ``loop`` end-to-end. The entry point for the Equinox backend.

    Returns:
        ``(trained_model_op, backend_info)`` where:
        - ``trained_model_op`` is a `_EquinoxModelOp` wrapping the
          trained ``eqx.Module``.
        - ``backend_info`` is a dict with ``backend``, ``jax_version``,
          ``equinox_version``, ``optax_version``, ``devices``, and
          ``total_seconds`` for inclusion in the TrainingArtifact.
    """
    t0 = time.time()
    # --- Unwrap the model ----------------------------------------------
    model_op = loop.model_op
    if isinstance(model_op, _EquinoxModelOp):
        eqx_module = model_op.module
    elif isinstance(model_op, eqx.Module):
        eqx_module = model_op
    elif hasattr(model_op, "module") and isinstance(model_op.module, eqx.Module):
        # Future pipekit-jax.JaxModelOp would land here.
        eqx_module = model_op.module
    else:
        raise TypeError(
            "Equinox adapter expects loop.model_op to be a pipekit_train."
            "adapters.equinox._EquinoxModelOp (or any operator exposing "
            ".module: eqx.Module). Got "
            f"{type(model_op).__name__}."
        )

    # --- Task / loss validation ----------------------------------------
    task: Any = loop.task
    if task is None:
        if loop.loss is None:
            raise ValueError(
                "Equinox adapter requires loop.loss or loop.task to be "
                "supplied. Pass a Loss (carrier-agnostic) or your own "
                "TrainTask (loss_fn-shaped) when constructing TrainingLoop."
            )
        task = _SynthesisedTask(loop.loss)
    elif not isinstance(task, TrainTask):
        raise TypeError(
            "Equinox adapter: loop.task must satisfy the TrainTask Protocol "
            f"(missing `loss_fn`); got {type(task).__name__}."
        )

    # --- Optimiser, RNG -------------------------------------------------
    optimizer = _build_optimizer(loop.optimizer_config)

    state = TrainState.create(eqx_module, optimizer)
    rng = jax.random.key(loop.seed)

    # --- Checkpoint manager (if checkpoint_dir set) --------------------
    mngr: ocp.CheckpointManager | None = None
    if loop.checkpoint_dir:
        # Look for a Checkpoint callback to learn cadence + retention.
        every_n = loop.eval_every_n_steps
        keep_last = 3
        for cb in loop.callbacks:
            if hasattr(cb, "every_n_steps") and hasattr(cb, "keep_last"):
                every_n = int(cb.every_n_steps)
                keep_last = int(cb.keep_last)
                break
        mngr = ocp.CheckpointManager(
            directory=os.path.abspath(loop.checkpoint_dir),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=keep_last,
                save_interval_steps=every_n,
            ),
        )
        # Resume from latest checkpoint if one exists.
        latest = mngr.latest_step()
        if latest is not None:
            state = restore_state(mngr, state, latest)

    # --- Data iterator -------------------------------------------------
    train_iter = _iter_batches(loop.dataset, loop.batch_size, loop.seed)

    # --- Initial-state callback dispatch -------------------------------
    initial_state = _carry_state(state, loop)
    _dispatch(loop.callbacks, "on_train_begin", loop, initial_state)

    # --- Main loop -----------------------------------------------------
    last_metrics: dict[str, float] = {}
    while int(state.step) < loop.max_steps:
        rng, step_key = jax.random.split(rng)
        batch = next(train_iter)
        state, metrics = train_step(state, batch, step_key, task, optimizer)
        last_metrics = {k: float(np.asarray(v)) for k, v in metrics.items()}

        step = int(state.step)

        # Per-step log to the writer (if any).
        if loop.metric_writer is not None and step % loop.log_every_n_steps == 0:
            loop.metric_writer.write(step, last_metrics)

        # Callbacks on_step_end.
        carry = _carry_state(state, loop, last_metrics)
        _dispatch(loop.callbacks, "on_step_end", loop, carry, last_metrics)

        # Periodic epoch boundary (cosmetic; controls on_epoch_end firing).
        if loop.steps_per_epoch and step > 0 and step % loop.steps_per_epoch == 0:
            _dispatch(loop.callbacks, "on_epoch_end", loop, carry, last_metrics)

        # Periodic evaluation.
        if (
            loop.val_dataset is not None
            and step > 0
            and step % loop.eval_every_n_steps == 0
        ):
            eval_metrics = _evaluate(
                state.model, loop.val_dataset, loop.batch_size, task
            )
            _dispatch(loop.callbacks, "on_eval_end", loop, carry, eval_metrics)

        # Checkpoint — gated on the manager's configured cadence
        # (CheckpointManager.save_interval_steps, derived from the
        # Checkpoint callback's every_n_steps). Calling save() every
        # step would defeat the configured cadence and add I/O on
        # the fast path.
        if mngr is not None and step > 0 and mngr.should_save(step):
            save_state(mngr, state, step)

        # Early-stopping check — break before next step.
        if _any_should_stop(loop.callbacks):
            break

    # --- Final eval (if val_dataset and we didn't just eval) ----------
    if loop.val_dataset is not None and int(state.step) % loop.eval_every_n_steps != 0:
        final_eval = _evaluate(state.model, loop.val_dataset, loop.batch_size, task)
        carry = _carry_state(state, loop, last_metrics)
        _dispatch(loop.callbacks, "on_eval_end", loop, carry, final_eval)

    # --- Wait for any async checkpoint -------------------------------
    if mngr is not None:
        mngr.wait_until_finished()

    # --- Wrap the trained module + dispatch on_train_end -------------
    trained_model_op = _EquinoxModelOp(state.model)
    final_carry = _carry_state(
        state, loop, last_metrics, model_override=trained_model_op
    )
    _dispatch(loop.callbacks, "on_train_end", loop, final_carry)

    # --- Close any open writer ---------------------------------------
    if loop.metric_writer is not None:
        loop.metric_writer.close()

    backend_info = {
        "backend": "equinox",
        "jax_version": jax.__version__,
        "equinox_version": eqx.__version__,
        "optax_version": optax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "total_seconds": time.time() - t0,
        "final_step": int(state.step),
        "final_metrics": last_metrics,
    }
    return trained_model_op, backend_info


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _carry_state(
    state: TrainState,
    loop: TrainingLoop,
    metrics: dict[str, float] | None = None,
    model_override: Operator | None = None,
) -> Any:
    """Build a TrainerCarryState snapshot from the JAX-side TrainState.

    Imported lazily so this module can be imported without instantiating
    the loop module (which imports back into us).
    """
    from pipekit_train.loop import TrainerCarryState

    step = int(state.step)
    epoch = step // loop.steps_per_epoch if loop.steps_per_epoch else 0
    return TrainerCarryState(
        model=model_override or _EquinoxModelOp(state.model),
        opt_state=state.opt_state,
        step=step,
        epoch=epoch,
        metrics=dict(metrics) if metrics is not None else {},
        rng=None,
    )
