"""Adjoint strategy specs — a shared vocabulary for gradient flow.

Differentiating a data-assimilation system involves up to three nested
rollouts: the dynamics integration (one forecast window), the inner
variational solve (4D-Var minimisation), and the outer assimilation
cycle. Each layer realises reverse-mode differentiation through a
different mechanism (diffrax adjoints, optimistix adjoints, scan
transforms), but the *strategy* — how gradients should flow — is the
same small set of concepts everywhere.

The classes here are that vocabulary: plain frozen dataclasses, pure
Python, serialisable like any other pipekit config knob. They carry no
behaviour; each execution layer interprets a spec into its native
mechanism (see ``pipekit_jax.to_diffrax_adjoint`` for the diffrax
mapping, and the algorithm libraries — filterax, vardax — for the
cycle- and solver-layer interpretations). Names mirror diffrax's
adjoint classes where the concepts coincide, so the vocabulary is
familiar; ``TruncatedAdjoint`` is the one concept diffrax does not
ship.

Strategy-by-layer summary:

| Spec                  | dynamics       | inner solve    | cycle       |
|-----------------------|----------------|----------------|-------------|
| ``DirectAdjoint``     | store all      | unrolled       | plain scan  |
| ``RecursiveCheckpointAdjoint`` | treeverse | checkpointed | ckpt. scan |
| ``TruncatedAdjoint``  | —              | k-step         | trunc. scan |
| ``ImplicitAdjoint``   | steady-state   | IFT            | —           |
| ``BacksolveAdjoint``  | continuous     | —              | —           |
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DirectAdjoint:
    """Exact gradients by storing every step of the rollout.

    Reverse-mode memory grows linearly with the rollout length; the
    right choice for short rollouts where recomputation overhead is
    not worth paying.
    """


@dataclass(frozen=True)
class RecursiveCheckpointAdjoint:
    """Exact gradients with treeverse checkpointing (the recommended default).

    Stores only ``checkpoints`` snapshots and recomputes segments
    during the backward pass (Griewank 1992): memory is logarithmic to
    square-root in the rollout length at the cost of extra forward
    work. ``checkpoints=None`` lets the interpreter pick its default
    schedule.

    Attributes:
        checkpoints: Number of checkpoints to store, or ``None`` for
            the interpreting layer's default schedule.
    """

    checkpoints: int | None = None


@dataclass(frozen=True)
class TruncatedAdjoint:
    """Biased gradients: flow stops more than ``k`` steps back.

    Steps before the trailing-``k`` window run under ``stop_gradient``,
    giving O(1) memory in the rollout length. ``k=1`` is the one-step
    adjoint (Bolte et al. 2023) — *exact* when the rollout has
    converged to a fixed point, and the ROAD-EnKF training regime when
    applied at the cycle layer. For chaotic dynamics, truncation is
    gradient *regularisation*: the exact adjoint norm grows
    exponentially with the horizon, so the unbiased gradient is
    useless at long horizons regardless of how it is computed.

    Attributes:
        k: Number of trailing steps that propagate gradients. Must be
            at least 1.
    """

    k: int = 1


@dataclass(frozen=True)
class ImplicitAdjoint:
    """Differentiate a converged fixed point via the implicit function theorem.

    O(1) memory and independent of the solve path, at the price of one
    linear solve with the (Gauss-Newton) Hessian. Only sound when the
    inner solve actually converged — an unconverged iterate gets the
    gradient of a point it never reached. Applies to the inner-solve
    layer (and steady-state dynamics solves); meaningless for a
    non-fixed-point cycle.
    """


@dataclass(frozen=True)
class BacksolveAdjoint:
    """Continuous adjoint: solve the adjoint ODE backwards (O(1) memory).

    Warning:
        Optimise-then-discretise — the result is *not* the exact
        gradient of the discrete forward solve, and reconstructing the
        forward trajectory by integrating backwards diverges for
        chaotic or stiff dynamics (the usual case in DA). Provided for
        completeness at the dynamics layer; never a default.
    """


AdjointSpec = (
    DirectAdjoint
    | RecursiveCheckpointAdjoint
    | TruncatedAdjoint
    | ImplicitAdjoint
    | BacksolveAdjoint
)
"""Union of the adjoint strategy specs."""
