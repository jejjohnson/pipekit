"""Protocols decomposing data assimilation into its standard form.

The core triple — `ForwardModel`, `ObservationOperator`, `AnalysisStep`
(predict, compare, update) — plus the reduced-order seams for
variational and reduced-rank methods: `ReducedBasis` (a reduced control
basis with a prior), `TangentLinearModel` (M' / M* of the dynamics),
`ErrorSubspace` (a propagating low-rank covariance factor), and
`ReducedOrderModel` (a Galerkin ROM — encode/decode + latent dynamics).
Each algorithm
library (filterx, vardax, plumax, …) ships adapter classes that satisfy
these protocols **without importing pipekit-cycle**. The protocols are
runtime-checkable so ``isinstance(obj, ForwardModel)`` succeeds on any
structurally compatible class.

See master plan Report 10, section 2.3.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ForwardModel(Protocol):
    """Advance a model state forward in time by ``dt``.

    Implementations: domain forward models (chemistry transport,
    ocean state, plume dispersion), neural emulators, hybrid
    physics + ML hybrids. Algorithm libraries provide adapters
    that satisfy this protocol structurally.

    Members:
        step(state, dt): Return the state advanced by ``dt``.
        dt: Default integration step.
        state_signature: Optional `pipekit.Signature` describing the
            shape / dtype of the state carrier. ``None`` if the model
            doesn't track named dimensions.
    """

    def step(self, state: Any, dt: float) -> Any: ...

    @property
    def dt(self) -> float: ...

    @property
    def state_signature(self) -> Any: ...


@runtime_checkable
class ObservationOperator(Protocol):
    """Map model state → predicted observations.

    The H operator in classical data-assimilation notation:
    ``H(x)`` produces "what would the observations look like if the
    state were ``x``?". The innovation in DA is then ``obs - H(forecast)``.

    Members:
        __call__(state): Return predicted observations for ``state``.
        linearize(state): Optional tangent-linear operator at
            ``state`` (returns a callable / matrix). Implementations
            that don't expose a linearisation may raise
            ``NotImplementedError``.
    """

    def __call__(self, state: Any) -> Any: ...

    def linearize(self, state: Any) -> Any: ...


@runtime_checkable
class AnalysisStep(Protocol):
    """Combine forecast state with observations to produce analysis state.

    Implementations: ensemble Kalman analyses (EnKF, ETKF, LETKF),
    variational solvers (3D/4D-Var), particle filters, smoothers.
    Algorithm libraries supply concrete classes.

    Members:
        __call__(forecast, obs, *, obs_op, obs_err_cov): Return the
            analysis state given the forecast, the observations, the
            observation operator, and the observation-error covariance.
    """

    def __call__(
        self,
        forecast: Any,
        obs: Any,
        *,
        obs_op: ObservationOperator,
        obs_err_cov: Any,
    ) -> Any: ...


@runtime_checkable
class ReducedBasis(Protocol):
    """Map a reduced control vector to a state increment (with a prior).

    Variational inverse problems (4D-Var on SSH, …) parameterise the
    control vector ``X`` not in full state space but in a reduced basis
    ``Φ`` (Gaussian RBF, wavelet, MIOST), each coefficient carrying a
    diagonal prior ``Q``. The basis maps coefficients to a state
    increment, and its prior supplies the background term ``½ Xᵀ Q⁻¹ X``.

    Concrete implementations live in algorithm libraries (or
    `pipekit_cycle.basis`) and satisfy this protocol structurally.

    Members:
        operg(t, X, state): Apply the basis — coefficients ``X`` to a
            state increment ``Φ(t) X`` at time ``t``.
        prior_inv(X): Apply the diagonal prior inverse ``Q⁻¹ X`` (the
            4D-Var background term operand).
        nbasis: Length of the control vector ``X``.
    """

    def operg(self, t: float, X: Any, state: Any = None) -> Any: ...

    def prior_inv(self, X: Any) -> Any: ...

    @property
    def nbasis(self) -> int: ...


@runtime_checkable
class TangentLinearModel(Protocol):
    """Tangent-linear and adjoint of a `ForwardModel`'s dynamics.

    The ``M'`` / ``M*`` seam for methods that need the linearised model
    explicitly — incremental 4D-Var inner loops and reduced-rank Kalman
    error-subspace propagation — rather than relying on automatic
    differentiation of `ForwardModel.step`. Implementations are supplied by
    algorithm libraries (or wrap ``jax.jvp`` / ``jax.vjp`` of a
    differentiable model).

    Members:
        tangent(state, dx, dt): Apply the tangent-linear ``M'(state)`` to a
            state perturbation ``dx``, advancing it by ``dt``.
        adjoint(state, dz, dt): Apply the adjoint ``M*(state)`` to an
            adjoint (dual) variable ``dz`` over ``dt``.
    """

    def tangent(self, state: Any, dx: Any, dt: float) -> Any: ...

    def adjoint(self, state: Any, dz: Any, dt: float) -> Any: ...


@runtime_checkable
class ErrorSubspace(Protocol):
    """A propagating low-rank error-covariance factor ``L`` (``B ≈ L Lᵀ``).

    The reduced-rank representation behind SEEK / SEIK / RRSQRT and other
    reduced-order Kalman methods: instead of a full covariance, the error is
    carried as a small set of modes that are propagated through the
    linearised dynamics each forecast and updated by the analysis. Thread it
    alongside the model state (e.g. on ``DAState.extras``) so a reduced-rank
    `AnalysisStep` can read and refresh it.

    Members:
        propagate(model, state, dt): Advance the modes by ``dt`` about the
            linearisation point ``state`` — typically ``M' L`` via a
            `TangentLinearModel`, or finite differences of a `ForwardModel`.
            Returns the forecast error subspace.
        modes: The low-rank factor (its columns are the error modes).
        rank: Number of modes retained.
    """

    def propagate(self, model: Any, state: Any, dt: float) -> ErrorSubspace: ...

    @property
    def modes(self) -> Any: ...

    @property
    def rank(self) -> int: ...


@runtime_checkable
class ReducedOrderModel(Protocol):
    """A Galerkin / POD reduced-order model: encode/decode + latent dynamics.

    A Galerkin ROM projects the full state onto a low-dimensional trial
    subspace and advances the *reduced* coordinates with the
    subspace-projected dynamics. Bundling the projection maps with the
    reduced step lets one object serve both uses:

    - **As a fast `ForwardModel`** on full states — wrap
      ``x -> decode(step(encode(x), dt))``.
    - **For data assimilation in reduced space** — ``encode`` the state (and
      observations), assimilate the small latent vector, then ``decode``.

    Concrete ROMs (POD/EOF, balanced truncation, autoencoder latents) live
    in algorithm libraries and satisfy this structurally; the latent
    dynamics may itself be a `ForwardModel` or a learned emulator.

    Members:
        encode(state): Full state -> reduced coordinates ``z`` (e.g.
            ``z = Psi^T (x - x_ref)`` for a (Petrov-)Galerkin trial/test
            basis).
        decode(coords): Reduced coordinates -> full state
            (``x = x_ref + Phi z``).
        step(coords, dt): Advance the reduced coordinates by ``dt`` with the
            projected reduced dynamics.
        latent_dim: Dimension of the reduced coordinate vector.
    """

    def encode(self, state: Any) -> Any: ...

    def decode(self, coords: Any) -> Any: ...

    def step(self, coords: Any, dt: float) -> Any: ...

    @property
    def latent_dim(self) -> int: ...
