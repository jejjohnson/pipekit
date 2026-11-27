"""The end-to-end training run, `run(loop)`."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import numpy as np
from pipekit_ops import Operator

from pipekit_train.backends import _common
from pipekit_train.backends.equinox._batches import _BatchSource
from pipekit_train.backends.equinox._evaluation import _evaluate
from pipekit_train.backends._equinox_ops import unwrap_module
from pipekit_train.backends.equinox._model import EquinoxModelOp, _wrap_trained_model
from pipekit_train.backends.equinox._sharding import ShardingSpec, _sharding_info
from pipekit_train.backends.equinox._state import (
    TrainState,
    TrainTask,
    _build_optimizer,
    _SynthesisedTask,
    restore_state,
    save_state,
    train_step,
)


# `optax` and `orbax.checkpoint` are lazy-imported inside the
# functions that need them so this module loads cleanly on systems
# that have equinox/jax but not the full [equinox] extra (e.g. the
# `uv sync --group dev` env that runs the rest of the test suite).
# All references below are wrapped in TYPE_CHECKING for static
# annotations.
if TYPE_CHECKING:
    import orbax.checkpoint as ocp  # ty: ignore[unresolved-import]

    from pipekit_train.loop._loop import TrainingLoop


# ---------------------------------------------------------------------
# Callback dispatch helpers
# ---------------------------------------------------------------------


# Callback dispatch + the early-stop check live in `adapters._common`,
# shared with the Bayesian backends.
_dispatch = _common.dispatch
_any_should_stop = _common.any_should_stop


def _resolve_checkpoint_dir(checkpoint_dir: str | None, callback: Any) -> str | None:
    """Return the absolute directory for checkpoints, or ``None`` for no saving.

    Args:
        checkpoint_dir: ``TrainingLoop.checkpoint_dir``.
        callback: The loop's `Checkpoint` callback, if any; its ``save_dir`` is
            used when the loop sets no directory.

    Returns:
        The absolute local path, or ``None`` if neither sets a directory.

    Raises:
        ValueError: If the two name different directories, or either is a URI,
            because the data-iterator side-car is written with local paths.
    """
    save_dir = getattr(callback, "save_dir", None)
    for candidate in (checkpoint_dir, save_dir):
        if candidate and "://" in candidate:
            raise ValueError(
                f"Checkpoint directories must be local paths; got {candidate!r}."
            )
    directory = checkpoint_dir or save_dir
    if not directory:
        return None
    resolved = os.path.abspath(directory)
    if checkpoint_dir and save_dir and os.path.abspath(save_dir) != resolved:
        raise ValueError(
            f"TrainingLoop.checkpoint_dir ({checkpoint_dir!r}) and "
            f"Checkpoint.save_dir ({save_dir!r}) name different directories. "
            "Set one of them."
        )
    return resolved


def run(loop: TrainingLoop) -> tuple[Operator, dict[str, Any]]:
    """Train ``loop`` end-to-end. The entry point for the Equinox backend.

    Returns:
        A pair ``(trained_model_op, backend_info)``. ``trained_model_op``
        is a `EquinoxModelOp` wrapping the trained ``eqx.Module``;
        ``backend_info`` is a dict with the keys ``backend``,
        ``jax_version``, ``equinox_version``, ``optax_version``,
        ``devices``, ``total_seconds``, ``final_step``, ``final_metrics``
        (the last training batch), and ``final_eval_metrics`` (the validation
        set on the final state, empty without a ``val_dataset``) for inclusion
        in the TrainingArtifact.
    """
    # Lazy imports — keep this module loadable without the full
    # [equinox] extra installed (see module docstring).
    try:
        import optax  # ty: ignore[unresolved-import]
        import orbax.checkpoint as ocp  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "backend='equinox' needs the full [equinox] extra (optax + "
            "orbax-checkpoint + grain). Install with "
            "`pip install pipekit-train[equinox]`."
        ) from exc

    t0 = time.time()
    # --- Unwrap the model ----------------------------------------------
    model_op = loop.model_op
    eqx_module = unwrap_module(model_op)
    if eqx_module is None:
        raise TypeError(
            "Equinox adapter expects loop.model_op to be a pipekit_train."
            "adapters.equinox.EquinoxModelOp (or any operator exposing "
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

    # --- Sharding (optional; issue #15) --------------------------------
    sharding: ShardingSpec | None = getattr(loop, "sharding", None)
    if sharding is not None and not isinstance(sharding, ShardingSpec):
        raise TypeError(
            "Equinox adapter: loop.sharding must be a pipekit_train.backends."
            f"equinox.ShardingSpec or None; got {type(sharding).__name__}."
        )

    # --- Optimiser, RNG -------------------------------------------------
    optimizer = _build_optimizer(loop.optimizer_config)

    state = TrainState.create(eqx_module, optimizer, sharding=sharding)
    rng = jax.random.key(loop.seed)

    # --- Checkpoint manager (if a directory is set) --------------------
    mngr: ocp.CheckpointManager | None = None
    # A Checkpoint callback sets cadence and retention, and may set the directory.
    checkpoint_cb = next(
        (
            cb
            for cb in loop.callbacks
            if hasattr(cb, "every_n_steps") and hasattr(cb, "keep_last")
        ),
        None,
    )
    checkpoint_dir = _resolve_checkpoint_dir(loop.checkpoint_dir, checkpoint_cb)
    if checkpoint_dir is not None:
        every_n = loop.eval_every_n_steps
        keep_last = 3
        if checkpoint_cb is not None:
            every_n = int(checkpoint_cb.every_n_steps)
            keep_last = int(checkpoint_cb.keep_last)
        mngr = ocp.CheckpointManager(
            directory=checkpoint_dir,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=keep_last,
                save_interval_steps=every_n,
            ),
        )
    # Resume from the latest checkpoint if one exists. Restores both
    # TrainState (weights + opt_state) AND the iterator position.
    # The data-iter state is saved aside for after we build the
    # iterator below.
    data_iter_state_to_restore: Any = None
    if mngr is not None:
        latest = mngr.latest_step()
        if latest is not None:
            state, data_iter_state_to_restore = restore_state(mngr, state, latest)

    # --- Data iterator -------------------------------------------------
    # Built only if we're actually going to consume batches. Skipping
    # this for finished resumes (state.step >= max_steps) means
    # users can reload a checkpointed model on a dataset smaller than
    # batch_size without tripping `_BatchSource`'s indexable-min-size
    # guard. See #17 / codex review on PR #24.
    train_iter: _BatchSource | None = None
    if int(state.step) < loop.max_steps:
        train_iter = _BatchSource(
            loop.dataset, loop.batch_size, loop.seed, sharding=sharding
        )
        if data_iter_state_to_restore is not None:
            train_iter.set_state(data_iter_state_to_restore)

    # --- Initial-state callback dispatch -------------------------------
    initial_state = _carry_state(state, loop)
    _dispatch(loop.callbacks, "on_train_begin", loop, initial_state)

    # --- Main loop -----------------------------------------------------
    # try/finally: a raising callback / loss / NaN guard must not leak
    # the metric writer's file handle (dropping buffered lines) or leave
    # an async checkpoint half-written.
    last_metrics: dict[str, float] = {}
    last_eval_metrics: dict[str, float] = {}
    try:
        # Inside the loop body, `train_iter` is guaranteed non-None because
        # the loop condition matches the guard that built it.
        while int(state.step) < loop.max_steps:
            assert train_iter is not None  # see iterator-build guard above
            rng, step_key = jax.random.split(rng)
            batch = train_iter.next_batch()
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
                rng, eval_key = jax.random.split(rng)
                last_eval_metrics = _evaluate(
                    state.model, loop.val_dataset, loop.batch_size, task, eval_key
                )
                _dispatch(loop.callbacks, "on_eval_end", loop, carry, last_eval_metrics)

            # Checkpoint — gated on the manager's configured cadence
            # (CheckpointManager.save_interval_steps, derived from the
            # Checkpoint callback's every_n_steps). Calling save() every
            # step would defeat the configured cadence and add I/O on
            # the fast path.
            if mngr is not None and step > 0 and mngr.should_save(step):
                save_state(mngr, state, step, data_iter_state=train_iter.get_state())

            # Early-stopping check — break before next step.
            if _any_should_stop(loop.callbacks):
                break

        # --- Save the final step, which is usually off the cadence ---------
        if (
            mngr is not None
            and train_iter is not None
            and int(state.step) != mngr.latest_step()
        ):
            save_state(
                mngr,
                state,
                int(state.step),
                data_iter_state=train_iter.get_state(),
                force=True,
            )

        # --- Final eval (if val_dataset and we didn't just eval) ------
        if (
            loop.val_dataset is not None
            and int(state.step) % loop.eval_every_n_steps != 0
        ):
            rng, eval_key = jax.random.split(rng)
            last_eval_metrics = _evaluate(
                state.model, loop.val_dataset, loop.batch_size, task, eval_key
            )
            carry = _carry_state(state, loop, last_metrics)
            _dispatch(loop.callbacks, "on_eval_end", loop, carry, last_eval_metrics)
    finally:
        # --- Wait for any async checkpoint + close the writer ---------
        if mngr is not None:
            mngr.wait_until_finished()
        if loop.metric_writer is not None:
            loop.metric_writer.close()

    # --- Wrap the trained module + dispatch on_train_end -------------
    trained_model_op = _wrap_trained_model(loop.model_op, state.model)
    final_carry = _carry_state(
        state, loop, last_metrics, model_override=trained_model_op
    )
    _dispatch(loop.callbacks, "on_train_end", loop, final_carry)

    backend_info = {
        "backend": "equinox",
        "jax_version": jax.__version__,
        "equinox_version": eqx.__version__,
        "optax_version": optax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "total_seconds": time.time() - t0,
        "final_step": int(state.step),
        "final_metrics": last_metrics,
        "final_eval_metrics": last_eval_metrics,
        "sharding": _sharding_info(sharding),
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
    from pipekit_train.loop._loop import TrainerCarryState

    step = int(state.step)
    epoch = step // loop.steps_per_epoch if loop.steps_per_epoch else 0
    return TrainerCarryState(
        model=model_override or EquinoxModelOp(state.model),
        opt_state=state.opt_state,
        step=step,
        epoch=epoch,
        metrics=dict(metrics) if metrics is not None else {},
        rng=None,
    )
