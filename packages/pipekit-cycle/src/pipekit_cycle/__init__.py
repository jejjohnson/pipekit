"""pipekit-cycle — time-stepping, data assimilation, observation operators.

Built on `pipekit.state` (the `StatefulOperator` + `CarryState`
primitives from pipekit core, Report 2 Group M). Ships:

- **Cycle wrappers** (`cycle`): `Cycle`, `EnsembleCycle`,
  `WindowedCycle`, `Recurrence`.
- **Protocols** (`protocols`): `ForwardModel`, `ObservationOperator`,
  `AnalysisStep`, `ReducedBasis`, `TangentLinearModel`, `ErrorSubspace`
  — runtime-checkable.
- **DA cycles** (`da`, `bfn`): `DACycle`, `EnsembleDACycle`,
  `SmootherCycle`, `BFNCycle` (adjoint-free back-and-forth nudging).
- **Observation operators** (`obs`): `IdentityObs`, `LinearObs`,
  `CallableObs`, `CompositeObs`.
- **Forward-model adapters** (`forward`): `CallableForward`,
  `CompositeForward`, `NeuralForward`.
- **Canonical carry-states** (`state`): `DAState`, `IterationState`,
  `WindowState`.

See master plan Report 10.
"""

from pipekit_cycle.bfn import BFNCycle
from pipekit_cycle.cycle import (
    Cycle,
    EnsembleCycle,
    Recurrence,
    WindowedCycle,
)
from pipekit_cycle.da import DACycle, EnsembleDACycle, SmootherCycle
from pipekit_cycle.forward import (
    CallableForward,
    CompositeForward,
    NeuralForward,
)
from pipekit_cycle.obs import (
    CallableObs,
    CompositeObs,
    IdentityObs,
    LinearObs,
)
from pipekit_cycle.protocols import (
    AnalysisStep,
    ErrorSubspace,
    ForwardModel,
    ObservationOperator,
    ReducedBasis,
    TangentLinearModel,
)
from pipekit_cycle.state import DAState, IterationState, WindowState


__all__ = [
    "AnalysisStep",
    "BFNCycle",
    "CallableForward",
    "CallableObs",
    "CompositeForward",
    "CompositeObs",
    "Cycle",
    "DACycle",
    "DAState",
    "EnsembleCycle",
    "EnsembleDACycle",
    "ErrorSubspace",
    "ForwardModel",
    "IdentityObs",
    "IterationState",
    "LinearObs",
    "NeuralForward",
    "ObservationOperator",
    "Recurrence",
    "ReducedBasis",
    "SmootherCycle",
    "TangentLinearModel",
    "WindowState",
    "WindowedCycle",
]

__version__ = "0.0.1"  # x-release-please-version
