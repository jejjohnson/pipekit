"""JAX interpreters for the pipekit adjoint-strategy specs.

Two pieces:

- `truncated_scan` — a `jax.lax.scan` whose carry runs under
  ``stop_gradient`` except within the trailing ``k`` steps. The
  execution engine for ``TruncatedAdjoint`` at the cycle layer.
- `to_diffrax_adjoint` — map a spec from `pipekit_cycle.adjoints` onto
  the corresponding ``diffrax.AbstractAdjoint`` (requires the
  ``pipekit-jax[diffrax]`` extra).

Specs are matched by class name and fields rather than identity, so
structurally identical spec objects defined by algorithm libraries
(which may not depend on pipekit) are accepted interchangeably.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def truncated_scan[Carry, X, Y](
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    *,
    k: int = 1,
) -> tuple[Carry, Y]:
    """``jax.lax.scan`` with gradients truncated to the trailing ``k`` steps.

    The carry leaving every step before ``T - k`` is wrapped in
    ``stop_gradient``, so reverse-mode differentiation sees only the
    last ``k`` transitions *through the carry*. Each step's own output
    ``y_t`` still receives gradients with respect to that step's
    inputs — only the flow through history is cut, which is exactly
    the ROAD-EnKF / truncated-BPTT training regime. ``k >= T`` is the
    identity transformation (a plain scan).

    Memory for the backward pass is O(k) in the rollout length instead
    of O(T), and for chaotic step functions the truncation acts as
    gradient regularisation (the exact adjoint norm grows like
    ``exp(lambda_max * T)``).

    Args:
        f: Scan body ``(carry, x) -> (carry, y)``.
        init: Initial carry.
        xs: Stacked per-step inputs with leading axis ``T``.
        k: Number of trailing steps that propagate gradients
            through the carry. Must be at least 1.

    Returns:
        ``(final_carry, ys)`` exactly as ``jax.lax.scan`` would —
        the forward values are identical; only gradients differ.

    Raises:
        ValueError: if ``k < 1`` or ``xs`` has no leading axis.
    """
    if k < 1:
        raise ValueError(f"truncated_scan needs k >= 1; got k={k}.")
    leaves = jax.tree.leaves(xs)
    if not leaves:
        raise ValueError("truncated_scan needs at least one stacked input in xs.")
    length = leaves[0].shape[0]
    cutoff = length - k

    def body(carry: Carry, indexed: tuple[Any, X]) -> tuple[Carry, Y]:
        i, x = indexed
        new_carry, y = f(carry, x)
        # jnp.where keeps the graph differentiable in the trailing
        # window and routes earlier steps through stop_gradient.
        cut = i < cutoff
        new_carry = jax.tree.map(
            lambda c: jnp.where(cut, jax.lax.stop_gradient(c), c), new_carry
        )
        return new_carry, y

    return jax.lax.scan(body, init, (jnp.arange(length), xs))


def to_diffrax_adjoint(spec: Any) -> Any:
    """Map a pipekit adjoint spec onto a ``diffrax.AbstractAdjoint``.

    Mapping (specs matched structurally by class name):

    - ``RecursiveCheckpointAdjoint(checkpoints)`` →
      ``diffrax.RecursiveCheckpointAdjoint(checkpoints)`` — the
      recommended default.
    - ``DirectAdjoint()`` → ``diffrax.DirectAdjoint()``.
    - ``BacksolveAdjoint()`` → ``diffrax.BacksolveAdjoint()``.
      **Warning:** continuous adjoints diverge for chaotic or stiff
      dynamics; never use as a default.
    - ``ImplicitAdjoint()`` → ``diffrax.ImplicitAdjoint()`` —
      steady-state solves only.
    - ``TruncatedAdjoint`` → ``ValueError``: truncation is a property
      of an outer rollout (cycle or solver iteration), not of a single
      ODE/SDE solve; use `truncated_scan` at that layer instead.

    Args:
        spec: A spec instance from `pipekit_cycle.adjoints` or any
            structurally identical object. A ready-made
            ``diffrax.AbstractAdjoint`` passes through unchanged.

    Returns:
        The corresponding diffrax adjoint instance.

    Raises:
        ValueError: for ``TruncatedAdjoint`` or an unrecognised spec.
        ModuleNotFoundError: if diffrax is not installed (install the
            ``pipekit-jax[diffrax]`` extra).
    """
    import diffrax  # deferred: optional extra

    if isinstance(spec, diffrax.AbstractAdjoint):
        return spec
    name = type(spec).__name__
    if name == "RecursiveCheckpointAdjoint":
        return diffrax.RecursiveCheckpointAdjoint(
            checkpoints=getattr(spec, "checkpoints", None)
        )
    if name == "DirectAdjoint":
        return diffrax.DirectAdjoint()
    if name == "BacksolveAdjoint":
        return diffrax.BacksolveAdjoint()
    if name == "ImplicitAdjoint":
        return diffrax.ImplicitAdjoint()
    if name == "TruncatedAdjoint":
        raise ValueError(
            "TruncatedAdjoint applies to an outer rollout (assimilation cycle "
            "or solver iteration), not to a single ODE/SDE solve — use "
            "pipekit_jax.truncated_scan at that layer."
        )
    raise ValueError(f"Unrecognised adjoint spec: {spec!r}")
