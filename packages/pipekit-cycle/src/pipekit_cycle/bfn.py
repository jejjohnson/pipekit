"""Back-and-Forth Nudging (BFN) — an adjoint-free assimilation cycle.

BFN (Auroux & Blum, 2008) repeatedly integrates the model **forward** then
**backward** over an assimilation window, relaxing the state toward the
observations on each pass, until the successive back-pass initial states
converge. It needs no adjoint — only the forward model run in both time
directions plus a nudging gain ``K`` — which makes it a cheap baseline
beside `SmootherCycle` / `Incremental4DVar`.

Sign convention: the nudging is **corrective on both passes** (``x <-
x + K(y - H x)``); ``way`` flips only the *model* integration direction
(``+dt`` forward, ``-dt`` backward, or a supplied ``backward_model``). This
is the contractive form that converges toward the observations.

See master plan Report 10 and ``.plans/pipekit_massh_gap_analysis.md`` §2.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pipekit import Operator, StatefulOperator

from pipekit_cycle.da import _advance_da_state
from pipekit_cycle.protocols import ForwardModel, ObservationOperator


class BFNCycle(StatefulOperator):
    """Back-and-Forth Nudging over an assimilation window.

    Each call runs up to ``max_iter`` forth/back sweeps over a ``window``
    of forward steps, relaxing toward observations each step, and stops
    once successive back-pass initial states agree within ``tol``. The
    number of sweeps actually run is recorded on ``self.iterations``.

    The forward model must support backward stepping — ``step(state, -dt)``
    — which holds for time-reversible cores and works through
    `CallableForward` (it forwards ``dt`` to your callable). A
    `NeuralForward` ignores ``dt``, so emulator cores must supply an
    explicit ``backward_model``.

    Args:
        forward_model: Object satisfying `ForwardModel` (forward pass).
        obs_op: Object satisfying `ObservationOperator`.
        nudging_gain: Callable ``innovation -> increment`` applying the
            gain ``K`` (e.g. a Gaspari-Cohn-tapered relaxation). The
            innovation is ``y - H(x)``.
        window: Number of forward steps per sweep.
        max_iter: Maximum forth/back sweeps.
        tol: Convergence threshold on the relative change of the back-pass
            initial state between sweeps.
        spectral_sigma: If set, Gaussian-filter the innovation before
            applying the gain (spectral nudging); needs ``scipy``.
        obs_source: Optional `Operator` invoked ``obs_source(step_index)``
            for the observation at each step. ``None`` skips nudging (the
            cycle is then a pure forth/back integration).
        backward_model: Optional `ForwardModel` for the backward pass when
            the forward model is not time-reversible.
        convergence_fn: Optional ``(new_x0, old_x0) -> float`` measuring the
            change between successive back-pass initial states. Defaults to a
            relative L2 norm (the only place the cycle reaches for ``numpy``);
            inject your own to keep the cycle dependency-free or to use a
            domain metric.

    Raises:
        TypeError: if the model / obs operator / obs source have the wrong
            type, or ``nudging_gain`` is not callable.
        ValueError: if ``window < 1`` or ``max_iter < 1``.
    """

    __config_mixin_auto__ = False
    forbid_in_yaml: ClassVar[bool] = True  # carries the nudging-gain closure

    def __init__(
        self,
        forward_model: ForwardModel,
        obs_op: ObservationOperator,
        nudging_gain: Callable[[Any], Any],
        *,
        window: int,
        max_iter: int = 10,
        tol: float = 1e-3,
        spectral_sigma: float | None = None,
        obs_source: Operator | None = None,
        backward_model: ForwardModel | None = None,
        convergence_fn: Callable[[Any, Any], float] | None = None,
    ) -> None:
        if not isinstance(forward_model, ForwardModel):
            raise TypeError("BFNCycle.forward_model must satisfy ForwardModel.")
        if not isinstance(obs_op, ObservationOperator):
            raise TypeError("BFNCycle.obs_op must satisfy ObservationOperator.")
        if not callable(nudging_gain):
            raise TypeError("BFNCycle.nudging_gain must be callable.")
        if obs_source is not None and not isinstance(obs_source, Operator):
            raise TypeError("BFNCycle.obs_source must be an Operator or None.")
        if backward_model is not None and not isinstance(backward_model, ForwardModel):
            raise TypeError("BFNCycle.backward_model must satisfy ForwardModel.")
        if window < 1:
            raise ValueError(f"BFNCycle.window must be >= 1, got {window}.")
        if max_iter < 1:
            raise ValueError(f"BFNCycle.max_iter must be >= 1, got {max_iter}.")
        self.forward_model = forward_model
        self.obs_op = obs_op
        self.nudging_gain = nudging_gain
        self.window = window
        self.max_iter = max_iter
        self.tol = tol
        self.spectral_sigma = spectral_sigma
        self.obs_source = obs_source
        self.backward_model = backward_model
        self.convergence_fn = convergence_fn
        self.iterations: int = 0

    def _relax(self, x: Any, obs: Any) -> Any:
        """Nudge ``x`` toward ``obs``: ``x + K(y - H x)`` (corrective)."""
        if obs is None:
            return x
        innov = obs - self.obs_op(x)
        if self.spectral_sigma is not None:
            from scipy.ndimage import gaussian_filter

            innov = gaussian_filter(innov, self.spectral_sigma)
        return x + self.nudging_gain(innov)

    def _integrate(self, x: Any, way: int, obs_seq: list[Any]) -> Any:
        """Integrate one sweep in time direction ``way`` (+1 / -1), nudging.

        The forward sweep **steps then relaxes** (obs applied at the new time
        level); the backward sweep **relaxes then steps**, so a reversed
        ``obs_seq`` lands each backward nudge on the *same* time level as its
        forward counterpart — correct for time-varying observations, where a
        uniform step-then-relax would shift every backward nudge by one step.
        """
        if way > 0:
            model = self.forward_model
            dt = model.dt
        else:
            model = self.backward_model or self.forward_model
            dt = -model.dt if self.backward_model is None else model.dt
        for obs in obs_seq:
            if way > 0:
                x = self._relax(model.step(x, dt), obs)
            else:
                x = model.step(self._relax(x, obs), dt)
        return x

    def _apply(self, carrier: Any, state: Any) -> tuple[Any, Any]:
        if self.obs_source is not None:
            window_obs = [self.obs_source(i) for i in range(self.window)]
        else:
            window_obs = [None] * self.window

        convergence = self.convergence_fn or _relative_change
        self.iterations = 0
        x0 = carrier
        end = carrier
        for _ in range(self.max_iter):
            self.iterations += 1
            end = self._integrate(x0, +1, window_obs)
            new_x0 = self._integrate(end, -1, list(reversed(window_obs)))
            change = convergence(new_x0, x0)
            x0 = new_x0
            if change < self.tol:
                break
        # propagate the converged initial state forward as the analysis
        end = self._integrate(x0, +1, window_obs)
        # advance the threaded clock / cycle count over the window, matching
        # DACycle / SmootherCycle so downstream operators see fresh metadata.
        for _ in range(self.window):
            state = _advance_da_state(state, self.forward_model.dt)
        return end, state

    def get_config(self) -> dict[str, Any]:
        return {
            "forward_model": type(self.forward_model).__name__,
            "obs_op": type(self.obs_op).__name__,
            "window": self.window,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "spectral_sigma": self.spectral_sigma,
            "obs_source": (
                type(self.obs_source).__name__ if self.obs_source is not None else None
            ),
            "backward_model": (
                type(self.backward_model).__name__
                if self.backward_model is not None
                else None
            ),
        }


def _relative_change(new: Any, old: Any) -> float:
    """Relative L2 change ``‖new - old‖ / (‖old‖ + ε)`` (carrier-agnostic)."""
    import numpy as np

    a = np.asarray(new, dtype=float)
    b = np.asarray(old, dtype=float)
    denom = float(np.linalg.norm(b)) + 1e-12
    return float(np.linalg.norm(a - b)) / denom
