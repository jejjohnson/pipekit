"""pipekit-cycle — time stepping, data assimilation, observation operators.

Four areas, each with one job. An area imports only from the ones before it.

- `pipekit_cycle.protocols`: the runtime-checkable seams, `ForwardModel`,
  `ObservationOperator`, `AnalysisStep` and the ensemble and inversion seams.
- `pipekit_cycle.cycles`: `Cycle`, `EnsembleCycle`, `WindowedCycle`,
  `Recurrence`, the carry states and the adjoint specs.
- `pipekit_cycle.components`: the observation operators and the forward-model
  adapters.
- `pipekit_cycle.assimilation`: `DACycle`, `EnsembleDACycle`, `SmootherCycle`
  and `BFNCycle`.

Built on `pipekit_ops.core` (`StatefulOperator` and `CarryState`). This package
re-exports the names of every area. A saved pipeline records each operator under
``pipekit_cycle``, wherever its file is, and a state saved under a module of the
flat layout (``pipekit_cycle.da``, ``pipekit_cycle.cycle``, …) still loads.

See master plan Report 10.
"""

from pipekit_ops import Operator
from pipekit_ops.core import record_moves, record_under

from pipekit_cycle.assimilation import (
    BFNCycle,
    DACycle,
    EnsembleDACycle,
    SmootherCycle,
)
from pipekit_cycle.components import (
    CallableForward,
    CallableObs,
    CompositeForward,
    CompositeObs,
    IdentityObs,
    LinearObs,
    NeuralForward,
)
from pipekit_cycle.cycles import (
    AdjointSpec,
    BacksolveAdjoint,
    Cycle,
    DAState,
    DirectAdjoint,
    EnsembleCycle,
    ImplicitAdjoint,
    IterationState,
    Recurrence,
    RecursiveCheckpointAdjoint,
    TruncatedAdjoint,
    WindowedCycle,
    WindowState,
)
from pipekit_cycle.cycles._cycle import _ForwardModelAdapter
from pipekit_cycle.protocols import (
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
    "AdjointSpec",
    "AnalysisStep",
    "BFNCycle",
    "BacksolveAdjoint",
    "CallableForward",
    "CallableObs",
    "CompositeForward",
    "CompositeObs",
    "CovarianceLocalizer",
    "Cycle",
    "DACycle",
    "DAState",
    "DirectAdjoint",
    "EnsembleCycle",
    "EnsembleDACycle",
    "EnsembleInflator",
    "ErrorSubspace",
    "ForwardModel",
    "IdentityObs",
    "ImplicitAdjoint",
    "IterationState",
    "IterativeProcess",
    "LatentMap",
    "LinearObs",
    "NeuralForward",
    "ObservationNoise",
    "ObservationOperator",
    "Recurrence",
    "RecursiveCheckpointAdjoint",
    "ReducedBasis",
    "ReducedOrderModel",
    "SmootherCycle",
    "StepScheduler",
    "SupportsAdjoint",
    "TangentLinearModel",
    "TruncatedAdjoint",
    "WindowState",
    "WindowedCycle",
]

__version__ = "0.2.0"  # x-release-please-version


record_under(
    __name__,
    _ForwardModelAdapter,
    *(
        value
        for name in __all__
        if isinstance(value := globals()[name], type) and issubclass(value, Operator)
    ),
)
record_moves(
    __name__,
    "pipekit_cycle.bfn",
    "pipekit_cycle.cycle",
    "pipekit_cycle.da",
    "pipekit_cycle.forward",
    "pipekit_cycle.obs",
)
