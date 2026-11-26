"""pipekit-train — training pipelines on top of pipekit.

Five areas, each with one job. An area imports only from the ones before it.

- `pipekit_train.data`: `TrainingDataset`, `IterableDataset`,
  `SimulationDataset`, `CachedDataset`, `Split` and `hash_config`.
- `pipekit_train.losses`: `Loss`, `MSE`, `NLL`, `KL`, `Composite` and
  `call_loss`.
- `pipekit_train.loop`: `TrainingLoop`, `ValidationStep`, `TrainerCarryState`,
  the callbacks and the metric writers.
- `pipekit_train.sweep`: `HyperSweep` and `ParameterGrid`.
- `pipekit_train.backends`: one area per backend, each behind its extra:
  ``equinox``, ``numpyro``, ``blackjax`` and ``keras``. `TrainingLoop` loads its
  backend by name when it runs, so importing pipekit-train imports none of them.

This package re-exports the names of the first four areas. A saved pipeline
records each of their operators under ``pipekit_train``, and each backend's under
its area, wherever the file is; a state saved under a module of the flat layout
(``pipekit_train.loop``, ``pipekit_train.adapters.equinox``, …) still loads.

See ``docs/design/`` for the design.
"""

from __future__ import annotations

from pipekit_ops import Operator
from pipekit_ops.core import record_moves, record_under

from pipekit_train.data import (
    CachedDataset,
    IterableDataset,
    SimulationDataset,
    TrainingDataset,
)
from pipekit_train.loop import (
    Callback,
    Checkpoint,
    EarlyStopping,
    JSONLWriter,
    LogToExperiment,
    MetricWriter,
    TrainerCarryState,
    TrainingLoop,
    ValidationStep,
)
from pipekit_train.losses import KL, MSE, NLL, Composite, Loss
from pipekit_train.sweep import HyperSweep, ParameterGrid, SweepCarryState, SweepResult


def __getattr__(name: str):
    # `XarrayWindowDataset` moved to geotoolz-patcher, which owns the temporal
    # stencils it needs: keeping it here made pipekit import geopatcher, a
    # cycle with geotoolz's dependency on pipekit. The old name stays as a
    # pointer, not a re-export.
    if name == "XarrayWindowDataset":
        raise ImportError(
            "XarrayWindowDataset moved to geopatcher: use"
            " `from geopatcher.integrations.train import XarrayWindowDataset`"
            " (install 'geotoolz-patcher[train]')."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KL",
    "MSE",
    "NLL",
    "CachedDataset",
    "Callback",
    "Checkpoint",
    "Composite",
    "EarlyStopping",
    "HyperSweep",
    "IterableDataset",
    "JSONLWriter",
    "LogToExperiment",
    "Loss",
    "MetricWriter",
    "ParameterGrid",
    "SimulationDataset",
    "SweepCarryState",
    "SweepResult",
    "TrainerCarryState",
    "TrainingDataset",
    "TrainingLoop",
    "ValidationStep",
]


record_under(
    __name__,
    *(
        value
        for name in __all__
        if isinstance(value := globals()[name], type) and issubclass(value, Operator)
    ),
)
record_moves(
    __name__,
    "pipekit_train.callbacks",
    "pipekit_train.dataset",
    "pipekit_train.loop",
    "pipekit_train.loss",
    "pipekit_train.sweep",
    "pipekit_train.writer",
)

__version__ = "0.2.0"  # x-release-please-version
