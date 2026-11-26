"""The training loop and what it reports to.

`TrainingLoop` runs a backend over a dataset, with `ValidationStep` and
`TrainerCarryState`; the callbacks `EarlyStopping`, `Checkpoint` and `LogToExperiment`
follow the `Callback` protocol, and `JSONLWriter` follows `MetricWriter`. The loop loads
its backend by name when it runs, so this area imports none of them.
"""

from __future__ import annotations

from pipekit_train.loop._callbacks import (
    Callback,
    Checkpoint,
    EarlyStopping,
    LogToExperiment,
)
from pipekit_train.loop._loop import TrainerCarryState, TrainingLoop, ValidationStep
from pipekit_train.loop._writer import JSONLWriter, MetricWriter


__all__ = [
    "Callback",
    "Checkpoint",
    "EarlyStopping",
    "JSONLWriter",
    "LogToExperiment",
    "MetricWriter",
    "TrainerCarryState",
    "TrainingLoop",
    "ValidationStep",
]
