"""`DiffraxForwardModel` — a pipekit `ForwardModel` backed by diffrax.

Wraps ``diffrax.diffeqsolve`` so a vector field (ODE) or drift +
diffusion pair (SDE) becomes a component that satisfies both
``pipekit_cycle.ForwardModel`` (``step(state, dt)`` plus ``dt`` /
``state_signature``) and ``pipekit_cycle.SupportsAdjoint``
structurally. The gradient strategy for the solve is declared with a
spec from `pipekit_cycle.adjoints` and interpreted by
`pipekit_jax.to_diffrax_adjoint` — so swapping, say, checkpointed for
backsolve adjoints is configuration, not code.

Requires the ``pipekit-jax[diffrax]`` extra.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from pipekit_cycle.adjoints import RecursiveCheckpointAdjoint

from pipekit_jax.adjoints import to_diffrax_adjoint


class DiffraxForwardModel(eqx.Module):
    """Advance a state with ``diffrax.diffeqsolve``.

    ODE by default; supply ``diffusion`` (and ``sde_key``) for an SDE,
    in which case the Brownian path is a ``VirtualBrownianTree`` over
    each integration span. The adjoint spec selects how reverse-mode
    gradients flow through the solve (default: treeverse
    checkpointing, the robust choice — see
    ``pipekit_cycle.adjoints`` for the menu and the warning on
    ``BacksolveAdjoint``).

    pipekit cycles call ``step(state, dt)`` with no absolute time, so
    integration always runs over ``[0, dt]`` (autonomous dynamics);
    ``as_dynamics()`` exposes the explicit ``(state, t0, t1)`` form
    used by algorithm libraries (e.g. filterax ``AbstractDynamics``).

    Attributes:
        vector_field: Drift ``f(t, y, args) -> dy/dt`` in diffrax
            convention.
        solver: Any diffrax solver (``Tsit5()`` for non-stiff ODEs,
            ``Kvaerno5()`` for stiff, ``Euler()``/``Heun()`` for
            SDEs, ...).
        dt0: Initial/fixed integrator step.
        dt: Default window length advertised to pipekit cycles.
        adjoint: Adjoint spec (or a raw ``diffrax.AbstractAdjoint``).
        stepsize_controller: diffrax step-size controller; defaults to
            constant steps.
        diffusion: Optional ``g(t, y, args)`` turning the solve into
            an SDE ``dy = f dt + g dW``. Per diffrax >= 0.7 semantics
            the return value must be a *linear map* from the Brownian
            increment into state space — e.g.
            ``lineax.DiagonalLinearOperator(sigma)`` for diagonal
            noise, or a ``(N_x, N_w)`` matrix.
        brownian_shape: Shape of the Brownian increment (SDE only).
            ``None`` derives it from the state — correct for diagonal
            noise. For non-square noise (a diffusion returning an
            ``(N_x, N_w)`` linear map with ``N_w != N_x``) pass
            ``(N_w,)`` explicitly.
        sde_key: PRNG key for the Brownian path (SDE only). The key is
            fixed at construction; refresh it per cycle through your
            DA state (e.g. ``DAState.extras["rng_key"]``) by rebuilding
            with ``dataclasses.replace`` if independent paths per
            window are required.
        max_steps: Bound on integrator steps per solve.
    """

    vector_field: Callable[[Any, Any, Any], Any]
    solver: diffrax.AbstractSolver = diffrax.Tsit5()
    dt0: float = 0.01
    dt: float = 1.0
    adjoint: Any = eqx.field(static=True, default_factory=RecursiveCheckpointAdjoint)
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()
    diffusion: Callable[[Any, Any, Any], Any] | None = None
    sde_key: jax.Array | None = None
    brownian_shape: tuple[int, ...] | None = eqx.field(static=True, default=None)
    max_steps: int = eqx.field(static=True, default=4096)

    def __check_init__(self) -> None:
        if self.diffusion is not None and self.sde_key is None:
            raise ValueError(
                "SDE dynamics (diffusion is not None) need an sde_key for the "
                "VirtualBrownianTree."
            )

    def _terms(self, t0: Any, t1: Any, state: Any) -> Any:
        drift = diffrax.ODETerm(self.vector_field)
        if self.diffusion is None:
            return drift
        bm_shape = (
            self.brownian_shape if self.brownian_shape is not None else jnp.shape(state)
        )
        brownian = diffrax.VirtualBrownianTree(
            t0,
            t1,
            tol=self.dt0 / 2,
            shape=jax.ShapeDtypeStruct(bm_shape, jnp.result_type(state)),
            key=self.sde_key,
        )
        return diffrax.MultiTerm(drift, diffrax.ControlTerm(self.diffusion, brownian))

    def _solve(self, state: Any, t0: Any, t1: Any) -> Any:
        sol = diffrax.diffeqsolve(
            self._terms(t0, t1, state),
            self.solver,
            t0=t0,
            t1=t1,
            dt0=self.dt0,
            y0=state,
            stepsize_controller=self.stepsize_controller,
            adjoint=to_diffrax_adjoint(self.adjoint),
            max_steps=self.max_steps,
            saveat=diffrax.SaveAt(t1=True),
        )
        return jax.tree.map(lambda y: y[-1], sol.ys)

    def step(self, state: Any, dt: Any) -> Any:
        """Advance ``state`` by ``dt`` (autonomous: integrates ``[0, dt]``)."""
        return self._solve(state, jnp.zeros_like(jnp.asarray(dt)), dt)

    @property
    def state_signature(self) -> None:
        """No shape contract advertised; pipekit treats ``None`` as opt-out."""
        return None

    def with_adjoint(self, spec: Any) -> DiffraxForwardModel:
        """Return a copy configured with another adjoint spec."""
        return dataclasses.replace(self, adjoint=spec)

    def as_dynamics(self) -> Callable[[Any, Any, Any], Any]:
        """Return the ``(state, t0, t1) -> state`` callable form.

        Plugs directly into APIs that take explicit integration spans,
        e.g. filterax ``AbstractDynamics`` slots or vardax 4D-Var
        ``forward`` arguments.
        """
        return self._solve
