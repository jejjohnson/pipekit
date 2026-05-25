"""Callback protocol.

The five-hook callback surface from the v0.1 design. Concrete
callbacks (`Checkpoint`, `EarlyStopping`, `LogToExperiment`) are
not shipped in this scaffold; they land alongside the v0.1
implementation of the `TrainingLoop`. See
``docs/design/api/components.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from pipekit import CarryState


class Callback(Protocol):
    """Per-step / per-epoch / per-eval hooks.

    Each of the five hooks is **optional** at runtime. Adapters
    dispatch via ``getattr(cb, hook_name, None)`` so a callback that
    only cares about one hook (say, ``on_step_end`` for a metric
    logger) implements just that method.

    This Protocol is intentionally **not** ``@runtime_checkable``:
    structural ``isinstance`` would require every callback to
    implement all five methods, which contradicts the
    partial-implementation model documented in
    ``docs/design/api/components.md``. The Protocol exists as the
    type-checker contract describing the *full* hook surface; the
    runtime contract is "ducks with the methods you actually use".

    See ADR D9 in ``docs/design/decisions.md``.
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
