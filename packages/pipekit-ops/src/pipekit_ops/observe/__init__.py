"""Looking inside a running pipeline.

The observers `Tap`, `Snapshot`, `ShapeTrace`, `Profile`, `Histogram` and `Log` pass
their input through and record something about it; `jit_report` says which steps of a
pipeline run under ``jax.jit`` and why the others do not.
"""

from __future__ import annotations

from pipekit_ops.observe._jit import (
    JitReport,
    JitStep,
    jit_report,
)
from pipekit_ops.observe._observers import (
    Histogram,
    Log,
    Profile,
    ShapeTrace,
    Snapshot,
    Tap,
)


__all__ = [
    "Histogram",
    "JitReport",
    "JitStep",
    "Log",
    "Profile",
    "ShapeTrace",
    "Snapshot",
    "Tap",
    "jit_report",
]
