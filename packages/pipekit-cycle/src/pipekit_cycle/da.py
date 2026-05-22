"""Data-assimilation cycles.

Three composable shapes:

- `DACycle` — classic forecast → observe → analyse loop.
- `EnsembleDACycle` — same shape, but the carrier is an ensemble of
  members; the analysis step is an ensemble update (EnKF, ETKF, …).
- `SmootherCycle` — windowed 4D-Var-shaped variant; the analysis step
  receives the full window of observations at once.

Each `DACycle` returns ``(analysis_carrier, state)``. When
``save_history=True`` the per-cycle ``(forecast, analysis)`` pairs
accumulate on ``self.history``.

See master plan Report 10, section 2.4.
"""

from __future__ import annotations

from typing import Any

from pipekit import Operator, StatefulOperator

from pipekit_cycle.protocols import AnalysisStep, ForwardModel, ObservationOperator


class DACycle(StatefulOperator):
    """Classic forecast-analysis cycle.

    Each call advances ``n_steps`` cycles. Within each cycle:

    1. **Forecast**: ``forecast = forward_model.step(state_carrier, dt)``.
    2. **Observation**: ``obs = obs_source(step_index)`` if
       ``obs_source`` is set, else skip the update.
    3. **Analyse**: ``carrier = analysis_step(forecast, obs,
       obs_op=obs_op, obs_err_cov=state.obs_err_cov)``.

    The CarryState (typically `DAState`) carries
    ``obs_err_cov`` and bookkeeping (``t``, ``cycle_count``). At
    least: any object with attributes ``t``, ``cycle_count``,
    ``obs_err_cov`` works — duck-typed.

    Args:
        forward_model: Object satisfying `ForwardModel`.
        obs_op: Object satisfying `ObservationOperator`.
        analysis_step: Object satisfying `AnalysisStep`.
        obs_source: Optional `Operator` invoked as
            ``obs_source(step_index)`` to obtain observations for the
            current cycle. If ``None``, the analysis step is skipped
            and the forecast is propagated forward.
        n_steps: Number of cycles per call.
        save_history: Append ``(forecast, analysis)`` pairs to
            ``self.history`` each cycle.

    Returns from ``_apply(carrier, state)``:
        ``(analysis_carrier, updated_state)``.
    """

    __config_mixin_auto__ = False

    def __init__(
        self,
        forward_model: ForwardModel,
        obs_op: ObservationOperator,
        analysis_step: AnalysisStep,
        obs_source: Operator | None = None,
        n_steps: int = 1,
        save_history: bool = True,
    ) -> None:
        if not isinstance(forward_model, ForwardModel):
            raise TypeError(
                "DACycle.forward_model must satisfy ForwardModel, "
                f"got {type(forward_model).__name__}."
            )
        if not isinstance(obs_op, ObservationOperator):
            raise TypeError(
                "DACycle.obs_op must satisfy ObservationOperator, "
                f"got {type(obs_op).__name__}."
            )
        if not isinstance(analysis_step, AnalysisStep):
            raise TypeError(
                "DACycle.analysis_step must satisfy AnalysisStep, "
                f"got {type(analysis_step).__name__}."
            )
        if obs_source is not None and not isinstance(obs_source, Operator):
            raise TypeError(
                "DACycle.obs_source must be an Operator or None, "
                f"got {type(obs_source).__name__}."
            )
        if n_steps < 0:
            raise ValueError(f"DACycle.n_steps must be >= 0, got {n_steps}.")
        self.forward_model = forward_model
        self.obs_op = obs_op
        self.analysis_step = analysis_step
        self.obs_source = obs_source
        self.n_steps = n_steps
        self.save_history = save_history
        self.history: list[tuple[Any, Any]] = []

    def _apply(self, carrier: Any, state: Any) -> tuple[Any, Any]:
        self.history = []
        for i in range(self.n_steps):
            forecast = self.forward_model.step(carrier, self.forward_model.dt)
            if self.obs_source is not None:
                obs = self.obs_source(i)
                carrier = self.analysis_step(
                    forecast,
                    obs,
                    obs_op=self.obs_op,
                    obs_err_cov=getattr(state, "obs_err_cov", None),
                )
            else:
                carrier = forecast
            if self.save_history:
                self.history.append((forecast, carrier))
            state = _advance_da_state(state, dt=self.forward_model.dt)
        return carrier, state

    def get_config(self) -> dict[str, Any]:
        return {
            "forward_model": type(self.forward_model).__name__,
            "obs_op": type(self.obs_op).__name__,
            "analysis_step": type(self.analysis_step).__name__,
            "obs_source": (
                type(self.obs_source).__name__ if self.obs_source is not None else None
            ),
            "n_steps": self.n_steps,
            "save_history": self.save_history,
        }


class EnsembleDACycle(StatefulOperator):
    """Ensemble-based DA — EnKF, ETKF, LETKF, particle filters, …

    Differences from `DACycle`:

    - The carrier is a list (or tuple) of ``n_members`` ensemble
      members. Each member is propagated through the forward model
      independently.
    - The analysis step receives the full ensemble (``forecast_members``)
      plus observations; it returns the updated ensemble.

    Args:
        forward_model: Same protocol as `DACycle`. Applied per-member.
        obs_op: Same protocol as `DACycle`.
        analysis_step: An ensemble analysis (e.g. filterx.EnKFAnalysis).
            Receives ``(forecast_members, obs, obs_op=..., obs_err_cov=...)``.
        obs_source: Optional observation source.
        n_steps: Number of cycles per call.
        n_members: Expected ensemble size (length-checked at call time).
    """

    __config_mixin_auto__ = False

    def __init__(
        self,
        forward_model: ForwardModel,
        obs_op: ObservationOperator,
        analysis_step: AnalysisStep,
        obs_source: Operator | None,
        n_steps: int,
        n_members: int,
    ) -> None:
        if not isinstance(forward_model, ForwardModel):
            raise TypeError("EnsembleDACycle.forward_model must satisfy ForwardModel.")
        if not isinstance(obs_op, ObservationOperator):
            raise TypeError("EnsembleDACycle.obs_op must satisfy ObservationOperator.")
        if not isinstance(analysis_step, AnalysisStep):
            raise TypeError("EnsembleDACycle.analysis_step must satisfy AnalysisStep.")
        if obs_source is not None and not isinstance(obs_source, Operator):
            raise TypeError("EnsembleDACycle.obs_source must be Operator or None.")
        if n_members < 1:
            raise ValueError(
                f"EnsembleDACycle.n_members must be >= 1, got {n_members}."
            )
        if n_steps < 0:
            raise ValueError(f"EnsembleDACycle.n_steps must be >= 0, got {n_steps}.")
        self.forward_model = forward_model
        self.obs_op = obs_op
        self.analysis_step = analysis_step
        self.obs_source = obs_source
        self.n_steps = n_steps
        self.n_members = n_members

    def _apply(  # ty: ignore[invalid-method-override]
        self,
        members: list[Any],
        state: Any,
    ) -> tuple[list[Any], Any]:
        if len(members) != self.n_members:
            raise ValueError(
                f"EnsembleDACycle expected {self.n_members} members, "
                f"got {len(members)}."
            )
        for i in range(self.n_steps):
            forecasts = [
                self.forward_model.step(m, self.forward_model.dt) for m in members
            ]
            if self.obs_source is not None:
                obs = self.obs_source(i)
                members = list(
                    self.analysis_step(
                        forecasts,
                        obs,
                        obs_op=self.obs_op,
                        obs_err_cov=getattr(state, "obs_err_cov", None),
                    )
                )
            else:
                members = forecasts
            state = _advance_da_state(state, dt=self.forward_model.dt)
        return members, state

    def get_config(self) -> dict[str, Any]:
        return {
            "forward_model": type(self.forward_model).__name__,
            "obs_op": type(self.obs_op).__name__,
            "analysis_step": type(self.analysis_step).__name__,
            "obs_source": (
                type(self.obs_source).__name__ if self.obs_source is not None else None
            ),
            "n_steps": self.n_steps,
            "n_members": self.n_members,
        }


class SmootherCycle(StatefulOperator):
    """4D-Var-shaped windowed smoother.

    Each call processes ``window`` forward steps, accumulating
    observations, then runs the analysis step on the whole window
    at once. This matches the 4D-Var control-variable formulation
    where the cost function is minimised over the full window
    jointly.

    Args:
        forward_model: Same protocol as `DACycle`.
        obs_op: Same protocol as `DACycle`.
        analysis_step: A smoother / variational solver. Called as
            ``analysis_step(window_forecasts, window_obs,
            obs_op=..., obs_err_cov=...)`` where the window arguments
            are lists.
        window: Number of forward steps per analysis window.
        stride: Number of windows to take per call (default 1).
        obs_source: Optional observation source; invoked per step
            inside the window. ``None`` skips the analysis step.
    """

    __config_mixin_auto__ = False

    def __init__(
        self,
        forward_model: ForwardModel,
        obs_op: ObservationOperator,
        analysis_step: AnalysisStep,
        window: int,
        stride: int = 1,
        obs_source: Operator | None = None,
    ) -> None:
        if not isinstance(forward_model, ForwardModel):
            raise TypeError("SmootherCycle.forward_model must satisfy ForwardModel.")
        if not isinstance(obs_op, ObservationOperator):
            raise TypeError("SmootherCycle.obs_op must satisfy ObservationOperator.")
        if not isinstance(analysis_step, AnalysisStep):
            raise TypeError("SmootherCycle.analysis_step must satisfy AnalysisStep.")
        if obs_source is not None and not isinstance(obs_source, Operator):
            raise TypeError("SmootherCycle.obs_source must be Operator or None.")
        if window < 1:
            raise ValueError(f"SmootherCycle.window must be >= 1, got {window}.")
        if stride < 1:
            raise ValueError(f"SmootherCycle.stride must be >= 1, got {stride}.")
        self.forward_model = forward_model
        self.obs_op = obs_op
        self.analysis_step = analysis_step
        self.obs_source = obs_source
        self.window = window
        self.stride = stride

    def _apply(self, carrier: Any, state: Any) -> tuple[Any, Any]:
        global_step = 0
        for _ in range(self.stride):
            forecasts: list[Any] = []
            observations: list[Any] = []
            for _ in range(self.window):
                carrier = self.forward_model.step(carrier, self.forward_model.dt)
                forecasts.append(carrier)
                if self.obs_source is not None:
                    observations.append(self.obs_source(global_step))
                global_step += 1
                state = _advance_da_state(state, dt=self.forward_model.dt)
            if self.obs_source is not None:
                carrier = self.analysis_step(
                    forecasts,
                    observations,
                    obs_op=self.obs_op,
                    obs_err_cov=getattr(state, "obs_err_cov", None),
                )
        return carrier, state

    def get_config(self) -> dict[str, Any]:
        return {
            "forward_model": type(self.forward_model).__name__,
            "obs_op": type(self.obs_op).__name__,
            "analysis_step": type(self.analysis_step).__name__,
            "obs_source": (
                type(self.obs_source).__name__ if self.obs_source is not None else None
            ),
            "window": self.window,
            "stride": self.stride,
        }


def _advance_da_state(state: Any, dt: float) -> Any:
    """Increment ``t`` and ``cycle_count`` if the state exposes them.

    Carrier-agnostic: any state object with mutable ``t`` /
    ``cycle_count`` attributes participates; objects without them are
    returned unchanged.
    """
    if state is None:
        return state
    if hasattr(state, "t"):
        state.t = state.t + dt
    if hasattr(state, "cycle_count"):
        state.cycle_count = state.cycle_count + 1
    return state
