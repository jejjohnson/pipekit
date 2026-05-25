"""Equinox + Optax + Grain + Orbax adapter — v0.1 reference.

This is the **reference** adapter for the v0.1 release — the
concrete JAX-native stack described in ``docs/design/api/adapters.md``
(Equinox section) and synthesised from the upstream eqx_trainer
design (`jej_vc_snippets/design_docs/eqx_trainer/`).

The scaffold ships this module with a stub ``run`` that raises
``NotImplementedError``; the body lands when v0.1 implementation
begins. Once implemented, the adapter will use:

- ``equinox.filter_jit`` + ``equinox.filter_grad`` for the
  per-step gradient update (`train_step`).
- ``optax`` for the optimiser transformation.
- ``grain.DataLoader`` for the data iterator, with
  ``grain.PyGrainCheckpointHandler`` so preemption restores both the
  weights *and* the data-iter position.
- ``orbax-checkpoint.CheckpointManager`` for serialisation, with
  ``eqx.partition`` bridging Equinox PyTrees into Orbax's
  ``StandardSave`` / ``StandardRestore``.

See ``docs/design/api/adapters.md`` for the full planned design and
ADRs D8 / D9 / D10 in ``docs/design/decisions.md`` for the locking
decisions.
"""

from __future__ import annotations

from typing import Any


def run(loop: Any) -> tuple[Any, dict[str, Any]]:
    """Train ``loop`` with the Equinox + Optax + Grain + Orbax stack.

    Args:
        loop: The `TrainingLoop` operator.

    Returns:
        ``(trained_model_op, backend_info)`` once implemented.

    Raises:
        NotImplementedError: Always in the v0.0 scaffold. The body
            lands when v0.1 implementation begins. See
            ``packages/pipekit-train/docs/design/api/adapters.md``.
    """
    raise NotImplementedError(
        "The Equinox adapter body lands with the v0.1 implementation. "
        "See packages/pipekit-train/docs/design/api/adapters.md "
        "(Equinox section) for the reference design."
    )
