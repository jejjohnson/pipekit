"""pipekit-array — Array API operators on top of pipekit.

Phase A surface: namespace dispatch + reductions + stack/concat
combinators. The full v0.1 catalogue (12 operators) lands across
phases A-D per `docs/design/boundaries.md` §4.
"""

from __future__ import annotations

from pipekit_array._namespace import array_namespace
from pipekit_array.combinators import ConcatenateAlong, StackAlong
from pipekit_array.reduce import MeanScalar


__version__ = "0.0.2"  # x-release-please-version

__all__ = [
    "ConcatenateAlong",
    "MeanScalar",
    "StackAlong",
    "array_namespace",
]
