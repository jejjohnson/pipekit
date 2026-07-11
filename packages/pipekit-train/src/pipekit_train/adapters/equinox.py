"""Equinox + Optax + Orbax adapter — the v0.1 reference.

Concrete JAX-native stack as described in
``docs/design/api/adapters.md`` (Equinox section). The adapter
synthesises a ``TrainTask`` from a `pipekit_train.Loss` if the user
doesn't supply one, builds the optimiser from ``optimizer_config``,
runs an ``eqx.filter_jit``'d train step, and threads the user's
callbacks through the loop.

Module surface:

- `ShardingSpec` — optional multi-device / multi-host sharding config
  (`mesh`, `model_pspec`, `data_pspec`); threaded through `TrainState`,
  the data iterator, and the Orbax restore path. See issue #15, ADR D12.
- `TrainState(eqx.Module)` — model + opt_state + step.
- `TrainTask` (Protocol) — user-defined training target, ported from
  the eqx_trainer design.
- `train_step` — pure function, ``@eqx.filter_jit`` wrapped.
- `save_state` / `restore_state` — Orbax bridge via ``eqx.partition``.
- `EquinoxModelOp(Operator)` — wraps a trained ``eqx.Module`` as a
  `pipekit.Operator`. Deliberate in-package twin of
  `pipekit_jax.JaxModelOp` (kept separate so pipekit-train depends only
  on pipekit core); unlike `JaxModelOp` it has no weight-blob
  round-trip — wrap the trained module in a `JaxModelOp` when you need
  `serialize_weights` / `from_registry`.
- `run(loop)` — full end-to-end training entry point.

See ADRs D8, D9, D10 in ``docs/design/decisions.md``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit import Operator

from pipekit_train.adapters import _common


# `optax` and `orbax.checkpoint` are lazy-imported inside the
# functions that need them so this module loads cleanly on systems
# that have equinox/jax but not the full [equinox] extra (e.g. the
# `uv sync --group dev` env that runs the rest of the test suite).
# All references below are wrapped in TYPE_CHECKING for static
# annotations.
if TYPE_CHECKING:
    import optax  # ty: ignore[unresolved-import]
    import orbax.checkpoint as ocp  # ty: ignore[unresolved-import]

    from pipekit_train.loop import TrainingLoop


# ---------------------------------------------------------------------
# EquinoxModelOp — pipekit.Operator wrapper around an eqx.Module
# ---------------------------------------------------------------------


class EquinoxModelOp(Operator):
    """Wrap an `eqx.Module` as a `pipekit.Operator`.

    Public surface used in notebooks and downstream packages. Same
    constructor signature and ``_apply`` shape as
    ``pipekit_jax.JaxModelOp`` — swap the import when you need the
    registry weight round-trip (``serialize_weights`` /
    ``with_weights`` / ``from_registry``), which this in-package twin
    deliberately omits so pipekit-train depends only on pipekit core.

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
                "EquinoxModelOp.module must be an eqx.Module; got "
                f"{type(module).__name__}."
            )
        self.module = module

    def _apply(self, x: Any) -> Any:
        # eqx.Module is callable (instance method __call__), but ty
        # doesn't know that statically.
        return cast(Any, self.module)(x)

    def get_config(self) -> dict[str, Any]:
        # __qualname__ matches pipekit_jax.JaxModelOp so the two twins emit
        # identical configs for the same module.
        return {"module_class": type(self.module).__qualname__}


# ---------------------------------------------------------------------
# ShardingSpec — multi-device / multi-host configuration (issue #15)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ShardingSpec:
    """Sharding configuration for the Equinox adapter.

    Threaded through `TrainState` (model + optimiser leaves placed per
    ``model_pspec``), the data iterator (each batch placed per
    ``data_pspec``, with multi-host process sharding), and the Orbax
    restore path (sharding-aware). See ADR D12.

    The common case is **data parallelism**: ``model_pspec =
    PartitionSpec()`` (the model is replicated on every device) and
    ``data_pspec = PartitionSpec("data")`` (each batch is split along its
    leading axis across the mesh's ``"data"`` dimension). A uniform
    model-parallel ``model_pspec`` is also honoured when every sharded
    array leaf has a compatible rank.

    Attributes:
        mesh: The device mesh.
        model_pspec: PartitionSpec applied to every model / optimiser
            array leaf. ``PartitionSpec()`` replicates.
        data_pspec: PartitionSpec applied to each batch (the leading axis
            is the batch axis).
    """

    mesh: jax.sharding.Mesh
    model_pspec: jax.sharding.PartitionSpec
    data_pspec: jax.sharding.PartitionSpec

    @property
    def model_sharding(self) -> jax.sharding.NamedSharding:
        """`NamedSharding` for model / optimiser leaves."""
        return jax.sharding.NamedSharding(self.mesh, self.model_pspec)

    @property
    def data_sharding(self) -> jax.sharding.NamedSharding:
        """`NamedSharding` for data batches."""
        return jax.sharding.NamedSharding(self.mesh, self.data_pspec)

    @property
    def replicated(self) -> jax.sharding.NamedSharding:
        """`NamedSharding` that replicates a value on every device."""
        return jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())


def _shard_pytree(tree: Any, sharding: Any, replicated: Any) -> Any:
    """Place the array leaves of ``tree`` on ``sharding`` (static leaves untouched).

    Rank-0 (scalar) leaves are always replicated: a non-trivial
    ``PartitionSpec`` cannot shard a scalar, and optimiser state carries
    scalar bookkeeping (e.g. Adam's update count) that would otherwise make
    ``jax.device_put`` fail under a model-parallel ``model_pspec``.
    """
    arrays, static = eqx.partition(tree, eqx.is_array)
    shardings = jax.tree.map(
        lambda x: replicated if jnp.ndim(x) == 0 else sharding, arrays
    )
    placed = jax.device_put(arrays, shardings)
    return eqx.combine(placed, static)


def _place_batch(x: jax.Array, sharding: Any) -> jax.Array:
    """Place a batch array on ``sharding``.

    Single-process: a straight ``jax.device_put``. Multi-process: assemble
    the global array from each process's local shard via
    ``jax.make_array_from_process_local_data`` (the multi-host path).
    """
    if jax.process_count() > 1:
        return jax.make_array_from_process_local_data(sharding, np.asarray(x))
    return jax.device_put(x, sharding)


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
    *,
    data_iter_state: Any = None,
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
    """
    import json
    import os
    from pathlib import Path

    import orbax.checkpoint as ocp  # ty: ignore[unresolved-import]

    arrays, _ = eqx.partition(state, eqx.is_array)
    mngr.save(step, args=ocp.args.StandardSave(arrays))

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
    import os
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


def _jsonify_grain_state(state: Any) -> Any:
    """Make a Grain iterator state dict JSON-serialisable.

    Grain's `iterator.get_state()` may include bytes (np serialised
    integers, prefetch buffer markers). We base64-encode bytes leaves
    and tag them so `_unjsonify_grain_state` can round-trip back.
    """
    import base64

    if isinstance(state, dict):
        return {k: _jsonify_grain_state(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_jsonify_grain_state(v) for v in state]
    if isinstance(state, tuple):
        return {"__tuple__": [_jsonify_grain_state(v) for v in state]}
    if isinstance(state, bytes):
        return {"__bytes_b64__": base64.b64encode(state).decode("ascii")}
    return state


def _unjsonify_grain_state(state: Any) -> Any:
    """Inverse of `_jsonify_grain_state`."""
    import base64

    if isinstance(state, dict):
        if "__bytes_b64__" in state:
            return base64.b64decode(state["__bytes_b64__"])
        if "__tuple__" in state:
            return tuple(_unjsonify_grain_state(v) for v in state["__tuple__"])
        return {k: _unjsonify_grain_state(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_unjsonify_grain_state(v) for v in state]
    return state


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


# ---------------------------------------------------------------------
# Data iteration — Grain or direct fallback
# ---------------------------------------------------------------------


def _is_indexable(dataset: Any) -> bool:
    return hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")


def _stack_pair(samples: list[tuple[Any, Any]]) -> tuple[jax.Array, jax.Array]:
    xs = jnp.stack([jnp.asarray(s[0]) for s in samples])
    ys = jnp.stack([jnp.asarray(s[1]) for s in samples])
    return xs, ys


class _BatchSource:
    """Iterator-with-state wrapper used by the Equinox `run()` loop.

    Exposes `next_batch()` + optional `get_state()` / `set_state()`
    so the loop can checkpoint and resume iterator position.
    Indexable datasets use Grain's `MapDataset` chain (state methods
    work); streaming datasets fall back to direct Python iteration
    (state methods return / accept ``None``).
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        seed: int,
        sharding: ShardingSpec | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._seed = seed
        self._sharding = sharding
        # Process-sharding only kicks in for an *explicit* spec; without one
        # the documented single-device behaviour is preserved even if JAX was
        # distributed-initialised (process_count > 1).
        sharded = sharding is not None
        # Additive hook: a dataset may build its own Grain iterator (e.g.
        # `XarrayWindowDataset`'s block reader — coalesced reads, window
        # shuffle, multi-host sharding). It returns a checkpointable Grain
        # iterator, or None to decline (then we fall through to the generic
        # branches). Datasets without the method are unaffected.
        build = getattr(dataset, "build_batch_iter", None)
        custom = build(batch_size, seed) if build is not None else None
        if custom is not None:
            self._inner = custom
            self._kind = "grain"
        elif _is_indexable(dataset):
            self._inner = self._build_grain_iter(dataset, batch_size, seed, sharded)
            self._kind = "grain"
        else:
            if sharded and jax.process_count() > 1:
                raise NotImplementedError(
                    "Multi-host sharding needs an indexable dataset (the Grain "
                    "process-shard path); streaming IterableDatasets are "
                    "single-process. Materialise the dataset or wrap it as an "
                    "indexable TrainingDataset."
                )
            self._inner = self._build_streaming_iter(dataset, batch_size)
            self._kind = "streaming"

    @staticmethod
    def _build_grain_iter(
        dataset: Any, batch_size: int, seed: int, sharded: bool
    ) -> Any:
        """Build the Grain iterator.

        Pipeline: ``MapDataset.source(dataset) → .shuffle(seed) →
        .repeat(num_epochs=None) → .batch(local_batch, drop_remainder=True)
        → .to_iter_dataset() → iter(...)``. ``drop_remainder=True``
        matches the v0.1 numpy fallback's behaviour of dropping
        trailing partial batches so `train_step` doesn't recompile
        per epoch boundary.

        ``batch_size`` is the **global** per-step batch. With an explicit
        sharding spec on multiple processes, each process reads a disjoint
        shard (the MapDataset-API equivalent of
        ``grain.python.ShardByJaxProcess``) and batches the *local* size
        ``batch_size // process_count`` so the per-process batches assemble
        back into a global batch of exactly ``batch_size``.

        Raises:
            ValueError: when ``len(dataset) < batch_size``, or when a
                multi-host ``batch_size`` is not divisible by
                ``process_count``.
        """
        try:
            import grain  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise ImportError(
                "The Equinox adapter's data path needs grain (part of the "
                "[equinox] extra). Install with "
                "`pip install pipekit-train[equinox]`."
            ) from exc

        n = len(dataset)
        if n < batch_size:
            raise ValueError(
                f"Dataset has {n} samples but batch_size={batch_size}. "
                "Reduce batch_size, grow the dataset, or wrap as "
                "IterableDataset for streaming (which doesn't have this "
                "constraint)."
            )
        map_ds = grain.MapDataset.source(dataset)
        n_proc = jax.process_count()
        local_batch = batch_size
        if sharded and n_proc > 1:
            if batch_size % n_proc != 0:
                raise ValueError(
                    f"Multi-host batch_size={batch_size} must be divisible by "
                    f"process_count={n_proc} so the global batch stays "
                    f"{batch_size}."
                )
            map_ds = map_ds[jax.process_index() :: n_proc]
            local_batch = batch_size // n_proc
        map_ds = (
            map_ds.shuffle(seed=seed)
            .repeat(num_epochs=None)
            .batch(batch_size=local_batch, drop_remainder=True)
        )
        return iter(map_ds.to_iter_dataset())

    @staticmethod
    def _build_streaming_iter(
        dataset: Any, batch_size: int
    ) -> Iterator[tuple[jax.Array, jax.Array]]:
        return _iter_streaming(dataset, batch_size)

    def next_batch(self) -> tuple[jax.Array, jax.Array]:
        batch = next(self._inner)
        xs, ys = jnp.asarray(batch[0]), jnp.asarray(batch[1])
        if self._sharding is not None:
            data_sharding = self._sharding.data_sharding
            xs = _place_batch(xs, data_sharding)
            ys = _place_batch(ys, data_sharding)
        return xs, ys

    def get_state(self) -> Any | None:
        """Return the inner iterator's state, or ``None`` for streaming."""
        if self._kind == "grain":
            return cast(Any, self._inner).get_state()
        return None

    def set_state(self, state: Any) -> None:
        """Restore the inner iterator's state.

        No-op for streaming iterators (which have no notion of state).
        """
        if self._kind == "grain" and state is not None:
            cast(Any, self._inner).set_state(state)


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
    key: jax.Array,
) -> dict[str, float]:
    """Run model on val_dataset, return averaged metric dict.

    ``key`` seeds stochastic tasks (dropout, reparameterised sampling)
    and is split per batch so every batch — and every eval round —
    sees distinct randomness. Deterministic tasks ignore it.
    """
    sums: dict[str, float] = {}
    n = 0
    # We always do a single-pass (one epoch) over the val dataset.
    # Use direct iteration; the loss_fn we already have can score
    # mini-batches as if they were training batches.
    iterator = _eval_pass(val_dataset, batch_size)
    for batch in iterator:
        key, batch_key = jax.random.split(key)
        _, aux = task.loss_fn(model, batch, batch_key)
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


# Callback dispatch + the early-stop check live in `adapters._common`,
# shared with the Bayesian backends.
_dispatch = _common.dispatch
_any_should_stop = _common.any_should_stop


# ---------------------------------------------------------------------
# run — the orchestration entry point
# ---------------------------------------------------------------------


def run(loop: TrainingLoop) -> tuple[Operator, dict[str, Any]]:
    """Train ``loop`` end-to-end. The entry point for the Equinox backend.

    Returns:
        A pair ``(trained_model_op, backend_info)``. ``trained_model_op``
        is a `EquinoxModelOp` wrapping the trained ``eqx.Module``;
        ``backend_info`` is a dict with the keys ``backend``,
        ``jax_version``, ``equinox_version``, ``optax_version``,
        ``devices``, ``total_seconds``, ``final_step``, and
        ``final_metrics`` for inclusion in the TrainingArtifact.
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
    if isinstance(model_op, EquinoxModelOp):
        eqx_module = model_op.module
    elif isinstance(model_op, eqx.Module):
        eqx_module = model_op
    elif hasattr(model_op, "module") and isinstance(model_op.module, eqx.Module):
        # pipekit_jax.JaxModelOp (or anything module-shaped) lands here.
        eqx_module = model_op.module
    else:
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
            "Equinox adapter: loop.sharding must be a pipekit_train.adapters."
            f"equinox.ShardingSpec or None; got {type(sharding).__name__}."
        )

    # --- Optimiser, RNG -------------------------------------------------
    optimizer = _build_optimizer(loop.optimizer_config)

    state = TrainState.create(eqx_module, optimizer, sharding=sharding)
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
                eval_metrics = _evaluate(
                    state.model, loop.val_dataset, loop.batch_size, task, eval_key
                )
                _dispatch(loop.callbacks, "on_eval_end", loop, carry, eval_metrics)

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

        # --- Final eval (if val_dataset and we didn't just eval) ------
        if (
            loop.val_dataset is not None
            and int(state.step) % loop.eval_every_n_steps != 0
        ):
            rng, eval_key = jax.random.split(rng)
            final_eval = _evaluate(
                state.model, loop.val_dataset, loop.batch_size, task, eval_key
            )
            carry = _carry_state(state, loop, last_metrics)
            _dispatch(loop.callbacks, "on_eval_end", loop, carry, final_eval)
    finally:
        # --- Wait for any async checkpoint + close the writer ---------
        if mngr is not None:
            mngr.wait_until_finished()
        if loop.metric_writer is not None:
            loop.metric_writer.close()

    # --- Wrap the trained module + dispatch on_train_end -------------
    trained_model_op = EquinoxModelOp(state.model)
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
        "sharding": _sharding_info(sharding),
    }
    return trained_model_op, backend_info


def _sharding_info(spec: ShardingSpec | None) -> dict[str, Any] | None:
    """A JSON-safe summary of the sharding configuration for the artifact."""
    if spec is None:
        return None
    return {
        "mesh_shape": {str(k): int(v) for k, v in spec.mesh.shape.items()},
        "model_pspec": str(spec.model_pspec),
        "data_pspec": str(spec.data_pspec),
        "num_processes": jax.process_count(),
    }


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
        model=model_override or EquinoxModelOp(state.model),
        opt_state=state.opt_state,
        step=step,
        epoch=epoch,
        metrics=dict(metrics) if metrics is not None else {},
        rng=None,
    )
