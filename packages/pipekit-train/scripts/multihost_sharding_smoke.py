"""Multi-host sharding smoke test for the Equinox adapter (issue #15).

Spawns ``--num-processes`` local processes wired together with
``jax.distributed.initialize`` (a single-machine stand-in for multi-host),
builds a small data-parallel `ShardingSpec`, and runs a few sharded
training steps. Each process reads a disjoint shard of the data (the
`grain` ``MapDataset`` process-slice) and the per-process batches are
assembled into a global ``jax.Array`` via
``jax.make_array_from_process_local_data`` inside the adapter.

This is **not** part of the pytest suite (it forks processes and needs a
free TCP port); it is the scripted multi-host check the issue's Definition
of Done calls for. Run it directly:

    python packages/pipekit-train/scripts/multihost_sharding_smoke.py
    python packages/pipekit-train/scripts/multihost_sharding_smoke.py --num-processes 4

Requires the ``[equinox]`` extra (``equinox``, ``optax``, ``jax``,
``grain``). Exits non-zero on failure so it can gate a manual/CI job.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from typing import Any


def _worker(process_id: int, num_processes: int, coordinator: str, q: Any) -> None:
    # One CPU device per process so the global mesh has `num_processes`
    # devices — the single-machine stand-in for one device per host.
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from jax.sharding import Mesh, PartitionSpec
        from pipekit_train.adapters.equinox import (
            ShardingSpec,
            TrainState,
            _place_batch,
            train_step,
        )

        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=num_processes,
            process_id=process_id,
        )

        # Global mesh spans every process's devices.
        mesh = Mesh(jax.devices(), axis_names=("data",))
        spec = ShardingSpec(
            mesh=mesh,
            model_pspec=PartitionSpec(),
            data_pspec=PartitionSpec("data"),
        )

        model = eqx.nn.MLP(
            in_size=1, out_size=1, width_size=8, depth=1, key=jax.random.key(0)
        )
        optimizer = optax.adam(1e-2)
        state = TrainState.create(model, optimizer, sharding=spec)

        class _Task:
            def loss_fn(self, model: Any, batch: Any, key: Any) -> Any:
                x, y = batch
                pred = jax.vmap(model)(x)
                return jnp.mean((pred - y) ** 2), {}

        task = _Task()

        # Per-process local batch (2 rows); the global batch is
        # 2 * num_processes, sharded along "data".
        rng = np.random.default_rng(process_id)
        local_x = rng.uniform(-1, 1, size=(2, 1)).astype("float32")
        local_y = (2 * local_x + 1).astype("float32")
        gx = _place_batch(jnp.asarray(local_x), spec.data_sharding)
        gy = _place_batch(jnp.asarray(local_y), spec.data_sharding)

        key = jax.random.key(0)
        for _ in range(20):
            state, _ = train_step(state, (gx, gy), key, task, optimizer)

        # Model stays replicated on every device across all processes.
        leaf = jax.tree.leaves(eqx.filter(state.model, eqx.is_array))[0]
        ok = bool(leaf.sharding.is_fully_replicated) and bool(
            jnp.isfinite(state.step).all()
        )
        q.put((process_id, ok, int(jax.process_count())))
    except Exception as exc:  # surface the failure to the parent
        q.put((process_id, False, repr(exc)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--coordinator", type=str, default="127.0.0.1:0")
    args = parser.parse_args()

    coordinator = args.coordinator
    if coordinator.endswith(":0"):
        import portpicker

        coordinator = f"127.0.0.1:{portpicker.pick_unused_port()}"

    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(i, args.num_processes, coordinator, q))
        for i in range(args.num_processes)
    ]
    for p in procs:
        p.start()
    results = [q.get() for _ in range(args.num_processes)]
    for p in procs:
        p.join()

    results.sort()
    all_ok = all(len(r) == 3 and r[1] is True for r in results)
    for r in results:
        print("process", r[0], "->", r[1:])
    if all_ok:
        print(f"OK: {args.num_processes}-process sharded training smoke passed.")
        return 0
    print("FAIL: see process results above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
