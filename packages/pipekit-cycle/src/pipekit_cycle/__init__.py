"""pipekit-cycle — time-stepping, data assimilation, observation operators.

Built on `pipekit.state` (the `StatefulOperator` + `CarryState`
primitives from pipekit core, Report 2 Group M). Ships:

- **Cycle wrappers** (`cycle`): `Cycle`, `EnsembleCycle`,
  `WindowedCycle`, `Recurrence`.
- **Protocols** (`protocols`): `ForwardModel`, `ObservationOperator`,
  `AnalysisStep` — runtime-checkable.
- **DA cycles** (`da`): `DACycle`, `EnsembleDACycle`, `SmootherCycle`.
- **Observation operators** (`obs`): `IdentityObs`, `LinearObs`,
  `CallableObs`, `CompositeObs`.
- **Forward-model adapters** (`forward`): `CallableForward`,
  `CompositeForward`, `NeuralForward`.
- **Canonical carry-states** (`state`): `DAState`, `IterationState`,
  `WindowState`.

See master plan Report 10.
"""

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
    ForwardModel,
    ObservationOperator,
)
from pipekit_cycle.state import DAState, IterationState, WindowState


__all__ = [
    "AnalysisStep",
    "CallableForward",
    "CallableObs",
    "CompositeForward",
    "CompositeObs",
    "Cycle",
    "DACycle",
    "DAState",
    "EnsembleCycle",
    "EnsembleDACycle",
    "ForwardModel",
    "IdentityObs",
    "IterationState",
    "LinearObs",
    "NeuralForward",
    "ObservationOperator",
    "Recurrence",
    "SmootherCycle",
    "WindowState",
    "WindowedCycle",
]

__version__ = "0.1.0"  # x-release-please-version
