"""The datasets a training run reads.

`TrainingDataset` is the base, `IterableDataset` streams, `SimulationDataset` draws from
a simulator and `CachedDataset` keeps what another dataset produced. `Split` names a
split, and `hash_config` is the content hash of a dataset's configuration.
"""

from __future__ import annotations

from pipekit_train.data._datasets import (
    CachedDataset,
    IterableDataset,
    SimulationDataset,
    Split,
    TrainingDataset,
    hash_config,
)


__all__ = [
    "CachedDataset",
    "IterableDataset",
    "SimulationDataset",
    "Split",
    "TrainingDataset",
    "hash_config",
]
