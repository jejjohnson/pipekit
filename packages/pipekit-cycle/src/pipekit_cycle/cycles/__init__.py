"""Time stepping: the cycles, their states, and how gradients flow through them.

`Cycle`, `EnsembleCycle`, `WindowedCycle` and `Recurrence` step a model through time;
`DAState`, `IterationState` and `WindowState` are the states they carry;
`DirectAdjoint`, `TruncatedAdjoint`, `RecursiveCheckpointAdjoint`, `ImplicitAdjoint` and
`BacksolveAdjoint` say how a gradient flows back through the steps.
"""

from __future__ import annotations

from pipekit_cycle.cycles._adjoints import (
    AdjointSpec,
    BacksolveAdjoint,
    DirectAdjoint,
    ImplicitAdjoint,
    RecursiveCheckpointAdjoint,
    TruncatedAdjoint,
)
from pipekit_cycle.cycles._cycle import (
    Cycle,
    EnsembleCycle,
    Recurrence,
    WindowedCycle,
)
from pipekit_cycle.cycles._state import (
    DAState,
    IterationState,
    WindowState,
)


__all__ = [
    "AdjointSpec",
    "BacksolveAdjoint",
    "Cycle",
    "DAState",
    "DirectAdjoint",
    "EnsembleCycle",
    "ImplicitAdjoint",
    "IterationState",
    "Recurrence",
    "RecursiveCheckpointAdjoint",
    "TruncatedAdjoint",
    "WindowState",
    "WindowedCycle",
]
