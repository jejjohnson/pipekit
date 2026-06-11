"""Protocols decomposing data assimilation into its standard form.

The core triple — `ForwardModel`, `ObservationOperator`, `AnalysisStep`
(predict, compare, update) — plus the reduced-order seams for
variational and reduced-rank methods: `ReducedBasis` (a reduced control
basis with a prior), `TangentLinearModel` (M' / M* of the dynamics),
`ErrorSubspace` (a propagating low-rank covariance factor), and
`ReducedOrderModel` (a Galerkin ROM — encode/decode + latent dynamics).

Ensemble-method seams: `ObservationNoise` (a noise model that can both
report its covariance and draw samples), `CovarianceLocalizer`
(distance-based covariance tapering), `EnsembleInflator` (post-analysis
ensemble inflation), and `LatentMap` (an encode/decode pair for
latent-space assimilation — the stateless subset of
`ReducedOrderModel`).

Iterative-inversion seams: `IterativeProcess` (ensemble Kalman
inversion / sampling — EKI, EKS, UKI) and `StepScheduler` (its
step-size policy).

Each algorithm
library (filterax, vardax, plumax, …) ships adapter classes that satisfy
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


@runtime_checkable
class LatentMap(Protocol):
    """An encode/decode pair between full state space and a latent space.

    The stateless subset of `ReducedOrderModel` — no latent dynamics,
    just the projection maps. This is the seam latent-space assimilation
    methods need (encode the ensemble, assimilate in latent space,
    decode the analysis) and what autoencoder priors naturally expose.
    Implementations: POD/EOF projections, autoencoders, or an identity
    map for testing.

    Members:
        encode(state): Full state -> latent coordinates ``z``.
        decode(coords): Latent coordinates -> full state.
    """

    def encode(self, state: Any) -> Any: ...

    def decode(self, coords: Any) -> Any: ...


@runtime_checkable
class ObservationNoise(Protocol):
    """An observation-error model that exposes covariance *and* sampling.

    Richer than the opaque ``obs_err_cov`` operand of `AnalysisStep`:
    deterministic analyses (ETKF, 3D-Var) need only the covariance,
    while stochastic ones (perturbed-observation EnKF) and synthetic-data
    generation also need draws from the error distribution. Bundling
    both keeps the two views consistent by construction.

    Members:
        covariance(): The error covariance — a matrix, a linear
            operator, or any object the consuming `AnalysisStep`
            understands.
        sample(key, shape): Draw error samples of the given batch
            shape. ``key`` is the RNG seed/key (e.g. a JAX PRNG key);
            implementations on a global RNG may ignore it.
    """

    def covariance(self) -> Any: ...

    def sample(self, key: Any, shape: tuple[int, ...]) -> Any: ...


@runtime_checkable
class CovarianceLocalizer(Protocol):
    """Taper spurious long-range covariances in ensemble methods.

    Localisation suppresses sampling noise in low-rank ensemble
    covariances by damping entries between distant locations
    (Gaspari-Cohn, Gaussian taper, hard cutoff, …). Cycles thread the
    localizer through to the `AnalysisStep`; supplying one is optional.

    Members:
        __call__(covariance, coords): Return the localized covariance
            given per-element coordinates used to compute distances.
    """

    def __call__(self, covariance: Any, coords: Any) -> Any: ...


@runtime_checkable
class EnsembleInflator(Protocol):
    """Inflate an analysis ensemble to counter variance underestimation.

    Ensemble filters systematically underestimate spread; inflation
    (multiplicative, additive, relaxation-to-prior) corrects for it
    after each analysis. Relaxation methods blend in the forecast
    ensemble, hence the optional second argument.

    Members:
        __call__(particles, forecast_particles=None, **kwargs): Return
            the inflated ensemble. ``forecast_particles`` is the prior
            ensemble for relaxation-to-prior schemes (RTPS / RTPP);
            pure multiplicative or additive schemes ignore it.
    """

    def __call__(
        self,
        particles: Any,
        forecast_particles: Any = None,
        **kwargs: Any,
    ) -> Any: ...


@runtime_checkable
class StepScheduler(Protocol):
    """Step-size policy for an `IterativeProcess`.

    Iterative ensemble methods (EKI, EKS, UKI) advance in pseudo-time;
    the scheduler chooses each increment — fixed, data-misfit
    controlled, or stability-bounded — from the current iteration
    state.

    Members:
        get_dt(state): Return the scalar pseudo-time step for the next
            iteration given the process state (particles, forward
            evaluations, misfit, …).
    """

    def get_dt(self, state: Any) -> Any: ...


@runtime_checkable
class IterativeProcess(Protocol):
    """An iterative ensemble inversion / sampling process (EKI, EKS, UKI).

    The parameter-estimation counterpart of the sequential-DA triple:
    instead of cycling through time, the ensemble iterates in
    pseudo-time toward the posterior over parameters. pipekit owns the
    loop (initialise, evaluate the forward model on each particle,
    update, repeat per the `StepScheduler`); algorithm libraries supply
    the process.

    Members:
        init(particles, obs, noise_cov): Build the initial iteration
            state from the prior ensemble, the observations, and the
            observation-noise covariance.
        update(state, forward_evals, **kwargs): Apply one iteration
            given the forward-model evaluations of the current
            particles; return the updated state.
    """

    def init(self, particles: Any, obs: Any, noise_cov: Any) -> Any: ...

    def update(self, state: Any, forward_evals: Any, **kwargs: Any) -> Any: ...
