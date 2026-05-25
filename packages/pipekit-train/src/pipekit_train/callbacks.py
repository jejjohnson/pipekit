"""Callback protocol.

The five-hook callback surface from the v0.1 design. Concrete
callbacks (`Checkpoint`, `EarlyStopping`, `LogToExperiment`) are
not shipped in this scaffold; they land alongside the v0.1
implementation of the `TrainingLoop`. See
``docs/design/api/components.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from pipekit import CarryState


@runtime_checkable
class Callback(Protocol):
    """Per-step / per-epoch / per-eval hooks.

    Each hook is optional — adapter dispatch uses
    ``getattr(cb, hook, None)``. Adapters translate these calls into
    their backend's native callback API (Lightning's ``pl.Callback``;
    Equinox: explicit hook points in the outer loop; Keras's
    ``keras.callbacks.Callback``).

    See ``docs/design/api/components.md`` and ADR D9 in
    ``docs/design/decisions.md``.
    """

    def on_train_begin(self, loop: Any, state: CarryState) -> None: ...

    def on_step_end(
        self,
        loop: Any,
        state: CarryState,
        metrics: dict[str, float],
    ) -> None: ...

    def on_epoch_end(
        self,
        loop: Any,
        state: CarryState,
        metrics: dict[str, float],
    ) -> None: ...

    def on_eval_end(
        self,
        loop: Any,
        state: CarryState,
        eval_metrics: dict[str, float],
    ) -> None: ...

    def on_train_end(self, loop: Any, state: CarryState) -> None: ...
