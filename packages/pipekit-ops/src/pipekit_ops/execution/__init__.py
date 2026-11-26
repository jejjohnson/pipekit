"""Running an operator over many inputs.

`Map` is one engine over serial, thread, process and async execution and any borrowed
`concurrent.futures.Executor`, with `executor_scope`, `Failed` and `MapError`;
`ThreadMap`, `ProcessMap`, `AsyncMap` and `BatchedMap` are its presets; `Prefetch` and
`Fold` stream and reduce; `check_pickleable` finds what a process pool cannot send.
"""

from __future__ import annotations

from pipekit_ops.execution._map import (
    Failed,
    Fold,
    Map,
    MapError,
    Prefetch,
    executor_scope,
)
from pipekit_ops.execution._presets import (
    AsyncMap,
    BatchedMap,
    ProcessMap,
    ThreadMap,
    check_pickleable,
)


__all__ = [
    "AsyncMap",
    "BatchedMap",
    "Failed",
    "Fold",
    "Map",
    "MapError",
    "Prefetch",
    "ProcessMap",
    "ThreadMap",
    "check_pickleable",
    "executor_scope",
]
