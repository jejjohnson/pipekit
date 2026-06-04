"""Multi-device sharding tests for the Equinox adapter (issue #15).

Runs on a CPU-simulated multi-device mesh (see ``conftest.py``). Gated by
``importorskip`` so it only runs in the ``[equinox]`` extra environment, and
skipped if fewer than two devices are visible (JAX already initialised
single-device elsewhere in the session).
"""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np
import pytest


pytest.importorskip("equinox")
pytest.importorskip("optax")
pytest.importorskip("jax")
pytest.importorskip("orbax.checkpoint")
pytest.importorskip("grain")

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, PartitionSpec
from pipekit_train import MSE, IterableDataset, TrainingDataset, TrainingLoop
from pipekit_train.adapters.equinox import (
    EquinoxModelOp,
    ShardingSpec,
    TrainState,
    _BatchSource,
    restore_state,
    save_state,
    shard_train_state,
)


if len(jax.devices()) < 2:
    pytest.skip(
        "sharding tests need a >=2-device mesh; set "
        "XLA_FLAGS=--xla_force_host_platform_device_count=2 before JAX starts",
        allow_module_level=True,
    )


# --- Fixtures -------------------------------------------------------------


def _mesh() -> Mesh:
    """A 1-D, 2-device mesh named ``"data"``."""
    devices = np.array(jax.devices()[:2])
    return Mesh(devices, axis_names=("data",))


def _spec() -> ShardingSpec:
    """Data-parallel spec: model replicated, batch sharded along ``"data"``."""
    return ShardingSpec(
        mesh=_mesh(),
        model_pspec=PartitionSpec(),
        data_pspec=PartitionSpec("data"),
    )


def _toy_mlp(key: jax.Array) -> eqx.nn.MLP:
    return eqx.nn.MLP(in_size=1, out_size=1, width_size=16, depth=2, key=key)


def _pairs(n: int = 64, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    ys = (2.0 * xs + 1.0 + 0.01 * rng.standard_normal((n, 1))).astype(np.float32)
    return list(zip(xs, ys, strict=True))


class _IndexableDataset(TrainingDataset):
    """Minimal indexable dataset so the adapter takes the Grain path."""

    __config_mixin_auto__ = False

    def __init__(self, pairs: list[tuple[Any, Any]]) -> None:
        super().__init__()
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        return self._pairs[idx]

    def __iter__(self) -> Any:
        yield from self._pairs

    def content_hash(self) -> str:
        return f"indexable-{len(self._pairs)}"

    def get_config(self) -> dict[str, Any]:
        return {"n": len(self._pairs)}


def _mse(model: eqx.Module, pairs: list[tuple[Any, Any]]) -> float:
    xs = jnp.asarray(np.stack([p[0] for p in pairs]))
    ys = jnp.asarray(np.stack([p[1] for p in pairs]))
    preds = jax.vmap(model)(xs)
    return float(jnp.mean((preds - ys) ** 2))


# --- TrainState placement -------------------------------------------------


def test_shard_train_state_replicates_model() -> None:
    spec = _spec()
    model = _toy_mlp(jax.random.key(0))
    state = TrainState.create(model, optax.adam(1e-3))
    sharded = shard_train_state(state, spec)

    for leaf in jax.tree.leaves(eqx.filter(sharded.model, eqx.is_array)):
        assert leaf.sharding == spec.replicated
    assert sharded.step.sharding == spec.replicated


def test_model_parallel_replicates_optimizer_scalars() -> None:
    # A non-trivial model_pspec must not be applied to Adam's rank-0 update
    # count (jax.device_put with a named PartitionSpec can't shard a scalar).
    spec = ShardingSpec(
        mesh=_mesh(),
        model_pspec=PartitionSpec("data"),  # shard each leaf's leading axis
        data_pspec=PartitionSpec("data"),
    )
    model = eqx.nn.Linear(4, 4, key=jax.random.key(7))  # weight (4,4), bias (4,)
    state = TrainState.create(model, optax.adam(1e-3), sharding=spec)  # must not raise

    opt_leaves = jax.tree.leaves(eqx.filter(state.opt_state, eqx.is_array))
    scalars = [leaf for leaf in opt_leaves if leaf.ndim == 0]
    arrays = [leaf for leaf in opt_leaves if leaf.ndim > 0]
    assert scalars and arrays
    assert all(leaf.sharding.is_fully_replicated for leaf in scalars)
    # the non-scalar optimiser leaves are actually sharded along "data"
    assert any(not leaf.sharding.is_fully_replicated for leaf in arrays)


def test_create_with_sharding_places_leaves() -> None:
    spec = _spec()
    model = _toy_mlp(jax.random.key(1))
    state = TrainState.create(model, optax.adam(1e-3), sharding=spec)

    model_leaves = jax.tree.leaves(eqx.filter(state.model, eqx.is_array))
    opt_leaves = jax.tree.leaves(eqx.filter(state.opt_state, eqx.is_array))
    assert model_leaves and opt_leaves
    for leaf in (*model_leaves, *opt_leaves):
        assert leaf.sharding == spec.replicated


# --- Batch placement ------------------------------------------------------


def test_batch_source_shards_batches() -> None:
    spec = _spec()
    src = _BatchSource(_IndexableDataset(_pairs()), batch_size=8, seed=0, sharding=spec)
    xs, ys = src.next_batch()
    assert xs.sharding == spec.data_sharding
    assert ys.sharding == spec.data_sharding
    # batch_size 8 over a 2-device "data" axis => 4 rows per device.
    assert xs.shape == (8, 1)


def test_batch_source_unsharded_is_unchanged() -> None:
    src = _BatchSource(_IndexableDataset(_pairs()), batch_size=8, seed=0)
    xs, _ = src.next_batch()
    assert xs.shape == (8, 1)


# --- Checkpoint round-trip ------------------------------------------------


def test_sharded_checkpoint_roundtrip() -> None:
    pytest.importorskip("orbax.checkpoint")
    import orbax.checkpoint as ocp

    spec = _spec()
    model = _toy_mlp(jax.random.key(2))
    state = TrainState.create(model, optax.adam(1e-3), sharding=spec)

    with tempfile.TemporaryDirectory() as d:
        mngr = ocp.CheckpointManager(d)
        save_state(mngr, state, step=0)
        mngr.wait_until_finished()
        restored, _ = restore_state(mngr, state, step=0)

    orig = jax.tree.leaves(eqx.filter(state.model, eqx.is_array))
    back = jax.tree.leaves(eqx.filter(restored.model, eqx.is_array))
    assert len(orig) == len(back)
    for a, b in zip(orig, back, strict=True):
        # sharding preserved through the Orbax round-trip
        assert b.sharding == spec.replicated
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))


# --- End-to-end sharded training -----------------------------------------


@pytest.mark.parametrize("dataset_kind", ["grain", "streaming"])
def test_sharded_training_converges(dataset_kind: str) -> None:
    from pipekit_train.adapters.equinox import run as equinox_run

    spec = _spec()
    pairs = _pairs(n=64)
    if dataset_kind == "grain":
        dataset: TrainingDataset = _IndexableDataset(pairs)
    else:
        dataset = IterableDataset(source=pairs, content_hash="synth-64")

    model = _toy_mlp(jax.random.key(3))
    initial_mse = _mse(model, pairs)

    loop = TrainingLoop(
        model_op=EquinoxModelOp(model),
        dataset=dataset,
        loss=MSE(),
        optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=300,
        batch_size=8,
        backend="equinox",
        sharding=spec,
        seed=0,
    )
    trained_op, backend_info = equinox_run(loop)

    # converges
    final_mse = _mse(trained_op.module, pairs)
    assert final_mse < initial_mse * 0.5

    # trained weights stayed replicated on the mesh
    for leaf in jax.tree.leaves(eqx.filter(trained_op.module, eqx.is_array)):
        assert leaf.sharding.is_fully_replicated

    # the sharding config is reported in backend_info
    assert backend_info["sharding"] == {
        "mesh_shape": {"data": 2},
        "model_pspec": str(PartitionSpec()),
        "data_pspec": str(PartitionSpec("data")),
        "num_processes": 1,
    }
