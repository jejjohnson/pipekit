"""Deprecated forwarders for the adjoint maths that moved to integrix.

The backward sweeps now live in ``integrix.adjoint``; pipekit-jax keeps
composition (`ScanCycle`, `Rollout`, `DiffraxForwardModel`) and the specs
stay in `pipekit_cycle.adjoints`. Both names below emit a
``DeprecationWarning`` and will be removed in a later release:

- `truncated_scan` — use ``integrix.adjoint.truncated_scan``. It keeps
  its own implementation here, so the base install, which does not
  depend on integrix, still runs it.
- `to_diffrax_adjoint` — use ``integrix.adjoint.to_diffrax_adjoint``.
  It forwards there, so it needs the ``pipekit-jax[diffrax]`` extra,
  which installs integrix.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def _warn_moved(name: str) -> None:
    warnings.warn(
        f"pipekit_jax.{name} is deprecated and will be removed in a later "
        f"release; use integrix.adjoint.{name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def truncated_scan[Carry, X, Y](
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    *,
    k: int = 1,
) -> tuple[Carry, Y]:
    """``jax.lax.scan`` with gradients truncated to the trailing ``k`` steps.

    Deprecated: use ``integrix.adjoint.truncated_scan``, which has the
    same signature and forward values. This copy stays for one release
    and emits a ``DeprecationWarning`` on every call.

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
    _warn_moved("truncated_scan")
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

    Deprecated: forwards to ``integrix.adjoint.to_diffrax_adjoint`` and
    emits a ``DeprecationWarning``. The mapping is unchanged. Specs are
    matched by class name: ``RecursiveCheckpointAdjoint``,
    ``DirectAdjoint``, ``BacksolveAdjoint`` and ``ImplicitAdjoint`` map
    onto the diffrax classes of the same name, a ready-made
    ``diffrax.AbstractAdjoint`` passes through, and ``TruncatedAdjoint``
    is refused because truncation belongs to an outer rollout.

    Args:
        spec: A spec instance from `pipekit_cycle.adjoints` or any
            structurally identical object. A ready-made
            ``diffrax.AbstractAdjoint`` passes through unchanged.

    Returns:
        The corresponding diffrax adjoint instance.

    Raises:
        ValueError: for ``TruncatedAdjoint`` or an unrecognised spec.
        ImportError: if integrix is not installed (install the
            ``pipekit-jax[diffrax]`` extra).
    """
    _warn_moved("to_diffrax_adjoint")
    try:
        # deferred: integrix comes with the [diffrax] extra
        from integrix.adjoint import to_diffrax_adjoint as _to_diffrax_adjoint
    except ImportError as err:
        raise ImportError(
            "pipekit_jax.to_diffrax_adjoint forwards to "
            "integrix.adjoint.to_diffrax_adjoint; install the "
            "pipekit-jax[diffrax] extra."
        ) from err
    return _to_diffrax_adjoint(spec)
