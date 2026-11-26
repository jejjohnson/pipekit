"""The Equinox backend: Equinox, Optax, Orbax and Grain (``[equinox]``).

`run(loop)` trains end to end. It synthesises a `TrainTask` from a `pipekit_train.Loss`
when the loop has none, builds the optimiser from ``optimizer_config``, runs
`train_step` under ``eqx.filter_jit``, and threads the loop's callbacks through.
`TrainState` holds the model, the optimiser state and the step; `save_state` and
`restore_state` checkpoint it with Orbax; `ShardingSpec` and `shard_train_state` spread
it over devices and hosts (ADR D12). `EquinoxModelOp` wraps a trained ``eqx.Module`` as
an operator. It is a deliberate twin of `pipekit_jax.JaxModelOp`, so that pipekit-train
depends only on pipekit-ops; it writes its weights with `serialize_weights` but cannot
reload them, so pass a `JaxModelOp` to `TrainingLoop` for that, and `run` returns the
trained model in the same wrapper.

See ADRs D8, D9 and D10 in ``docs/design/decisions.md``.
"""

from __future__ import annotations

from pipekit_ops.core import record_moves, record_under

from pipekit_train.backends.equinox._model import EquinoxModelOp
from pipekit_train.backends.equinox._run import run
from pipekit_train.backends.equinox._sharding import ShardingSpec
from pipekit_train.backends.equinox._state import (
    TrainState,
    TrainTask,
    restore_state,
    save_state,
    shard_train_state,
    train_step,
)


__all__ = [
    "EquinoxModelOp",
    "ShardingSpec",
    "TrainState",
    "TrainTask",
    "restore_state",
    "run",
    "save_state",
    "shard_train_state",
    "train_step",
]


record_under(__name__, EquinoxModelOp)
record_moves(__name__, "pipekit_train.adapters.equinox")
