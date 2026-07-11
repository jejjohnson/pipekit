"""pipekit-train — training pipelines on top of pipekit.

v0.0 ships the carrier-agnostic Protocols (`Loss`, `Callback`,
`MetricWriter`), the default `JSONLWriter`, and stub adapter modules
(Lightning, Keras) that raise ``NotImplementedError`` per the v0.1
design. The full surface (datasets, training loop, Equinox adapter)
is documented under ``docs/design/`` and lands in v0.1.

See ``docs/design/`` for the design.
"""

from pipekit_train.callbacks import (
    Callback,
    Checkpoint,
    EarlyStopping,
    LogToExperiment,
)
from pipekit_train.dataset import (
    CachedDataset,
    IterableDataset,
    SimulationDataset,
    TrainingDataset,
)
from pipekit_train.loop import TrainerCarryState, TrainingLoop, ValidationStep
from pipekit_train.loss import KL, MSE, NLL, Composite, Loss
from pipekit_train.sweep import HyperSweep, ParameterGrid, SweepCarryState, SweepResult
from pipekit_train.writer import JSONLWriter, MetricWriter


def __getattr__(name: str):
    # `XarrayWindowDataset` is re-exported lazily: importing it eagerly would
    # pull `xarray` + `geopatcher` (the `[xreader]` extra) into every
    # `import pipekit_train`, breaking the base install. PEP 562 defers it
    # until first access.
    if name == "XarrayWindowDataset":
        from pipekit_train.xarray_window import XarrayWindowDataset

        return XarrayWindowDataset
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
    "XarrayWindowDataset",
]

__version__ = "0.0.2"  # x-release-please-version
