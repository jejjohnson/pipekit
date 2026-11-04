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
        x: Any carrier. An object JAX doesn't treat as a pytree counts as one
            leaf, and one it can't flatten, such as a dict with mixed key types,
            counts as untraced.

    Returns:
        ``True`` if a leaf of ``x`` is a tracer. Always ``False`` when JAX has
        not been imported.

    Examples:
        >>> is_traced(1.0)
        False
        >>> is_traced({"a": [1.0, 2.0]})
        False
        >>> is_traced({0: "zero", "a": "letter"})
        False
    """
    jax = sys.modules.get("jax")
    if jax is None:
        return False
    # JAX can't flatten every carrier: a dict with mixed key types isn't sortable.
    # This check must never break an eager call that worked without it.
    try:
        leaves = jax.tree_util.tree_leaves(x)
    except Exception:
        return False
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)
