"""The config of a component that a cycle holds, which need not be an operator."""

from __future__ import annotations

from typing import Any

from pipekit_ops import Operator
from pipekit_ops.core import nested_config


def _component_config(obj: Any) -> Any:
    """Debug-config payload for a DA component.

    Components are protocol objects (`ForwardModel`, `AnalysisStep`, …)
    that need not be pipekit Operators; when one is, emit core's
    canonical ``{"class", "config"}`` payload, otherwise fall back to
    the bare class name.
    """
    if obj is None:
        return None
    if isinstance(obj, Operator):
        return nested_config(obj)
    return type(obj).__name__
