"""The training state, the training target, one train step, and Orbax checkpoints."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

from pipekit_train.backends import _common
from pipekit_train.backends._batches import (
    _jsonify_grain_state,
    _unjsonify_grain_state,
)
from pipekit_train.backends.equinox._sharding import ShardingSpec, _shard_pytree
from pipekit_train.losses import call_loss


# `optax` and `orbax.checkpoint` are lazy-imported inside the
# functions that need them so this module loads cleanly on systems
# that have equinox/jax but not the full [equinox] extra (e.g. the
# `uv sync --group dev` env that runs the rest of the test suite).
# All references below are wrapped in TYPE_CHECKING for static
# annotations.
if TYPE_CHECKING:
    import optax  # ty: ignore[unresolved-import]
    import orbax.checkpoint as ocp  # ty: ignore[unresolved-import]


# ---------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------


class TrainState(eqx.Module):
    """Equinox-side training state.

    A PyTree-clean container holding the bare ``eqx.Module``
    (without the ``EquinoxModelOp`` wrapper), the Optax optimizer
    state, and the global step counter.
    """

    model: eqx.Module
    opt_state: optax.OptState
    step: jax.Array

    @staticmethod
    def create(
        model: eqx.Module,
        optimizer: optax.GradientTransformation,
        sharding: ShardingSpec | None = None,
    ) -> TrainState:
        """Build the initial training state, optionally placed on a mesh.

        When ``sharding`` is given, the model and optimiser array leaves are
        placed on ``sharding.model_sharding`` and the step counter is
        replicated, so the very first ``train_step`` already runs sharded.
        """
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        state = TrainState(
            model=model,
            opt_state=opt_state,
            step=jnp.asarray(0, dtype=jnp.int32),
        )
        if sharding is not None:
            state = shard_train_state(state, sharding)
        return state


def shard_train_state(state: TrainState, spec: ShardingSpec) -> TrainState:
    """Return ``state`` with its model + optimiser leaves placed on ``spec``.

    The model and optimiser arrays go to ``spec.model_sharding``; the scalar
    step is replicated across the mesh.
    """
    return TrainState(
        model=_shard_pytree(state.model, spec.model_sharding, spec.replicated),
        opt_state=_shard_pytree(state.opt_state, spec.model_sharding, spec.replicated),
        step=jax.device_put(state.step, spec.replicated),
    )


# ---------------------------------------------------------------------
# TrainTask Protocol + synthesis from carrier-agnostic Loss
# ---------------------------------------------------------------------


@runtime_checkable
class TrainTask(Protocol):
    """User-defined Equinox training target.

    Provide ``loss_fn``. The adapter synthesises a `TrainTask` from
    ``loop.loss`` when no explicit task is supplied.

    By default evaluation also calls ``loss_fn``, compiled, on ragged batches,
    and averages its aux metrics per sample across the validation set.

    A task may also define ``eval_fn``, which switches evaluation to padded
    batches. Every batch then has ``batch_size`` rows, padded with copies of the
    last real row, and compiles once. ``valid`` is ``(batch_size,)`` bool, True
    for a real row, and the task must leave the other rows out of its metrics.
    Each batch's metrics are weighted by ``valid.sum()``, so a masked per-sample
    mean is exact across the set; a metric such as a quantile is not.

    Signatures:
        loss_fn(model, batch, key) -> (scalar_loss, aux_metrics_dict)
        eval_fn(model, batch, key, valid) -> metrics_dict  (optional)
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
        x, y = batch
        pred = jax.vmap(model)(x)
        result = call_loss(self.loss, pred, y, key)
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
    *,
    data_iter_state: Any = None,
    force: bool = False,
) -> None:
    """Persist the array leaves of ``state`` at ``step``.

    Static (non-array) leaves are reconstructed on restore from the
    in-memory template; see ``restore_state``.

    Args:
        mngr: Orbax checkpoint manager.
        state: Current ``TrainState``.
        step: Step number.
        data_iter_state: Optional state dict from
            ``grain_iterator.get_state()``. When provided, it's
            persisted as a JSON side-car next to the Orbax checkpoint
            (``<directory>/<step>/data_iter.json``) so resume-from-
            checkpoint restores byte-identical data-iter position
            alongside the weights. See #17.
        force: Save even if ``step`` is off the manager's cadence.
    """
    import json
    from pathlib import Path

    import orbax.checkpoint as ocp  # ty: ignore[unresolved-import]

    arrays, _ = eqx.partition(state, eqx.is_array)
    mngr.save(step, args=ocp.args.StandardSave(arrays), force=force)

    if data_iter_state is not None:
        # Orbax writes the checkpoint asynchronously; the per-step dir
        # exists after wait_until_finished. For correctness, write the
        # side-car eagerly into the same path. Orbax exposes
        # `mngr.directory / str(step)` as the per-step layout.
        mngr.wait_until_finished()
        step_dir = Path(os.fspath(mngr.directory)) / str(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        # Per-process side-car: under multi-host sharding each process owns a
        # distinct data shard and so a distinct Grain iterator state. A single
        # shared file would let the last writer win and every process would
        # then resume from the wrong position. Key it by process index.
        (step_dir / f"data_iter_{jax.process_index()}.json").write_text(
            json.dumps(_jsonify_grain_state(data_iter_state), sort_keys=True)
        )


def restore_state(
    mngr: ocp.CheckpointManager,
    template: TrainState,
    step: int,
) -> tuple[TrainState, Any | None]:
    """Restore array leaves into the static skeleton of ``template``.

    Returns:
        ``(restored_state, data_iter_state)``. ``data_iter_state`` is
        ``None`` when no side-car was written at the checkpoint step
        (e.g. a checkpoint written by an earlier pipekit-train
        without the iterator-state extension). When non-``None`` the
        caller passes it to ``grain_iterator.set_state(state)`` to
        resume the data stream byte-identically. See #17.
    """
    import json
    from pathlib import Path

    import orbax.checkpoint as ocp  # ty: ignore[unresolved-import]

    arrays, static = eqx.partition(template, eqx.is_array)
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, arrays)
    restored = mngr.restore(step, args=ocp.args.StandardRestore(abstract))

    # Read this process's own iterator-state side-car (see save_state). Fall
    # back to the legacy shared name for checkpoints written before the
    # per-process split.
    step_dir = Path(os.fspath(mngr.directory)) / str(step)
    side_car = step_dir / f"data_iter_{jax.process_index()}.json"
    if not side_car.exists():
        side_car = step_dir / "data_iter.json"
    data_iter_state: Any | None = None
    if side_car.exists():
        data_iter_state = _unjsonify_grain_state(json.loads(side_car.read_text()))

    return (
        cast(TrainState, eqx.combine(restored, static)),
        data_iter_state,
    )


# ---------------------------------------------------------------------
# Optimiser construction
# ---------------------------------------------------------------------


def _build_optimizer(config: dict[str, Any]) -> optax.GradientTransformation:
    """Build the optax optimiser via the shared `adapters._common` helper.

    Equinox's default optimiser is ``adamw``; supported names and the
    ``lr`` → ``learning_rate`` normalisation live in `_common` so the
    surface can't drift between backends.
    """
    return _common.build_optimizer(config, default_name="adamw")
