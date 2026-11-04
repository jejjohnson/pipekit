"""Detect JAX tracers without importing JAX.

pipekit core has no JAX dependency, and it stays that way: if JAX was never
imported, nothing can be traced, so there is nothing to check.
"""

from __future__ import annotations

import sys
from typing import Any


def is_traced(x: Any) -> bool:
    """Return whether ``x`` is, or is a pytree containing, a JAX tracer.

    Args:
        x: Any carrier. Containers JAX doesn't treat as pytrees count as one
            untraced leaf.

    Returns:
        ``True`` if a leaf of ``x`` is a tracer. Always ``False`` when JAX has
        not been imported.

    Examples:
        >>> is_traced(1.0)
        False
        >>> is_traced({"a": [1.0, 2.0]})
        False
    """
    jax = sys.modules.get("jax")
    if jax is None:
        return False
    leaves = jax.tree_util.tree_leaves(x)
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)
