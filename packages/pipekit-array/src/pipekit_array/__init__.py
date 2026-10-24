"""pipekit-array — Array API operators on top of pipekit.

Phase A surface: namespace dispatch + reductions + stack/concat
combinators. The full v0.1 catalogue (12 operators) lands across
phases A-D per `docs/design/boundaries.md` §4.

Named arrays: `DimArray` (data, dims, labels) and its reference
adapter `DimArrayAdapter`, registered with `pipekit.register_adapter`
on import.
"""

from __future__ import annotations

from pipekit import register_adapter

from pipekit_array._namespace import array_namespace
from pipekit_array.combinators import ConcatenateAlong, StackAlong
from pipekit_array.dimarray import DimArray, DimArrayAdapter
from pipekit_array.reduce import MeanScalar


__version__ = "0.0.3"  # x-release-please-version

register_adapter(DimArray, DimArrayAdapter())

__all__ = [
    "ConcatenateAlong",
    "DimArray",
    "DimArrayAdapter",
    "MeanScalar",
    "StackAlong",
    "array_namespace",
]
