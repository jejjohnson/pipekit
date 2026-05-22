"""Three protocols decomposing data assimilation into its standard form.

`ForwardModel`, `ObservationOperator`, `AnalysisStep` — predict, compare,
update. Each algorithm library (filterx, vardax, plumax, …) ships
adapter classes that satisfy these protocols **without importing
pipekit-cycle**. The protocols are runtime-checkable so
``isinstance(obj, ForwardModel)`` succeeds on any structurally compatible
class.

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
