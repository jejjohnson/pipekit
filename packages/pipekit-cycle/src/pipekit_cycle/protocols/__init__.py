"""The structural seams a library satisfies without importing pipekit-cycle.

`ForwardModel`, `ObservationOperator` and `AnalysisStep` are what a cycle steps,
observes and updates with; the rest (`TangentLinearModel`, `ObservationNoise`,
`CovarianceLocalizer`, `EnsembleInflator`, `ReducedBasis`, `LatentMap`,
`SupportsAdjoint`, …) are the seams of ensemble, inversion and reduced-order methods.
All are runtime-checkable.
"""

from __future__ import annotations

from pipekit_cycle.protocols._protocols import (
    AnalysisStep,
    CovarianceLocalizer,
    EnsembleInflator,
    ErrorSubspace,
    ForwardModel,
    IterativeProcess,
    LatentMap,
    ObservationNoise,
    ObservationOperator,
    ReducedBasis,
    ReducedOrderModel,
    StepScheduler,
    SupportsAdjoint,
    TangentLinearModel,
)


__all__ = [
    "AnalysisStep",
    "CovarianceLocalizer",
    "EnsembleInflator",
    "ErrorSubspace",
    "ForwardModel",
    "IterativeProcess",
    "LatentMap",
    "ObservationNoise",
    "ObservationOperator",
    "ReducedBasis",
    "ReducedOrderModel",
    "StepScheduler",
    "SupportsAdjoint",
    "TangentLinearModel",
]
