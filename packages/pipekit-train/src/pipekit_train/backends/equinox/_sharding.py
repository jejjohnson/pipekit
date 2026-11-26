"""Multi-device and multi-host sharding of the model and the data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


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
