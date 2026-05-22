"""Canonical `CarryState` subclasses for the cycle patterns.

Each cycle pattern brings its own state shape — pipekit-cycle ships
the three most common ones. Algorithm libraries are free to subclass
`pipekit.CarryState` directly with their own fields.

- `DAState` — clock + cycle bookkeeping for `DACycle` /
  `EnsembleDACycle`.
- `IterationState` — residual + iteration counter for `Recurrence`.
- `WindowState` — sliding-window bookkeeping for `WindowedCycle` /
  `SmootherCycle`.

See master plan Report 10, section 6.2.
"""

from __future__ import annotations

from typing import Any

from pipekit import CarryState


class DAState(CarryState):
    """State threaded through a DA cycle.

    Attributes:
        t: Current model time (seconds, or any user-defined unit).
        cycle_count: Number of completed cycles.
        obs_err_cov: Observation-error covariance (handed to the
            analysis step). Carrier-agnostic — typically a numpy /
            JAX array or a covariance object from the algorithm
            library.
        extras: Free-form dict for algorithm-specific state.
    """

    def __init__(
        self,
        t: float = 0.0,
        cycle_count: int = 0,
        obs_err_cov: Any = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        self.t = t
        self.cycle_count = cycle_count
        self.obs_err_cov = obs_err_cov
        self.extras = dict(extras) if extras is not None else {}


class IterationState(CarryState):
    """State threaded through a `Recurrence`.

    Attributes:
        iter_count: Number of completed iterations.
        residual_norm: Most recent residual norm (convergence check).
        extras: Free-form dict for algorithm-specific bookkeeping
            (learning rate schedule, line-search trace, …).
    """

    def __init__(
        self,
        iter_count: int = 0,
        residual_norm: float = float("inf"),
        extras: dict[str, Any] | None = None,
    ) -> None:
        self.iter_count = iter_count
        self.residual_norm = residual_norm
        self.extras = dict(extras) if extras is not None else {}


class WindowState(CarryState):
    """State threaded through a windowed / smoother cycle.

    Attributes:
        window_index: Zero-based index of the current window.
        last_analysis: Most recent analysis carrier (the bridge between
            windows). ``None`` before the first analysis is produced.
        extras: Free-form dict for algorithm-specific bookkeeping
            (windowed obs accumulator, adjoint trajectories, …).
    """

    def __init__(
        self,
        window_index: int = 0,
        last_analysis: Any = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        self.window_index = window_index
        self.last_analysis = last_analysis
        self.extras = dict(extras) if extras is not None else {}
