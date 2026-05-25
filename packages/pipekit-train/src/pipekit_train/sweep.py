"""Hyperparameter sweep orchestration.

Two operators:

- `ParameterGrid` — Cartesian-product iterator over named parameter
  axes. Yields ``{name: value, ...}`` dicts.
- `HyperSweep` — orchestrates N independent `TrainingLoop` runs by
  calling a user-supplied ``loop_template(params) -> TrainingLoop``
  for each point in the search space, then returns the trial results
  ranked by a configured metric.

v0.2 ships grid + sequential execution. Optuna / Ray Tune sampler
adapters land in v0.2.1; parallel execution lands in v0.3. The
StatefulOperator wrapping means a sweep composes into `Sequential`
just like a `TrainingLoop` (carry-state is the current best trial).

See issue #14 and design `boundaries.md` Q5.
"""

from __future__ import annotations

import itertools
import math
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pipekit import CarryState, Operator, StatefulOperator

from pipekit_train.loop import TrainingLoop


# ---------------------------------------------------------------------
# ParameterGrid — iterate a Cartesian product of named axes.
# ---------------------------------------------------------------------


class ParameterGrid(Operator):
    """Cartesian product over named parameter axes.

    Args:
        grid: Mapping ``axis_name → list of values``. Every axis
            must have at least one value. An **empty mapping**
            (``{}``) is also allowed and yields **zero** combinations
            (``len(grid) == 0``, ``list(grid) == []``) — this
            departs from the mathematical Cartesian-product
            convention where ``∏ over zero axes == 1``, in favour
            of the pragmatic "no axes ⇒ no trials to run" reading
            that downstream `HyperSweep` consumers expect.

    Iteration yields one ``{axis_name: value, ...}`` dict per cell.
    ``len(grid)`` is the product of axis lengths for non-empty
    grids; 0 for the empty grid per the convention above.

    Example:
        >>> list(ParameterGrid({"lr": [1e-3, 1e-2], "width": [8, 16]}))
        [{'lr': 0.001, 'width': 8}, {'lr': 0.001, 'width': 16},
         {'lr': 0.01, 'width': 8}, {'lr': 0.01, 'width': 16}]
    """

    def __init__(self, grid: dict[str, list[Any]]) -> None:
        if not isinstance(grid, dict):
            raise TypeError(
                f"ParameterGrid.grid must be a dict; got {type(grid).__name__}."
            )
        for name, values in grid.items():
            if not isinstance(values, (list, tuple)):
                raise TypeError(
                    f"ParameterGrid axis {name!r} must be a list/tuple of "
                    f"values; got {type(values).__name__}."
                )
            if len(values) == 0:
                raise ValueError(
                    f"ParameterGrid axis {name!r} is empty — every axis "
                    "must have at least one value."
                )
        self.grid = dict(grid)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.grid:
            return
        axes = list(self.grid.keys())
        value_lists = [self.grid[a] for a in axes]
        for combo in itertools.product(*value_lists):
            yield dict(zip(axes, combo, strict=True))

    def __len__(self) -> int:
        if not self.grid:
            return 0
        return math.prod(len(v) for v in self.grid.values())

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            "ParameterGrid is iterated, not called. Use `list(grid)` "
            "or pass it to `HyperSweep(search_space=...)`."
        )


# ---------------------------------------------------------------------
# SweepResult — one per trial.
# ---------------------------------------------------------------------


@dataclass
class SweepResult:
    """One trial's result inside a sweep.

    Attributes:
        params: The parameter dict that produced this trial.
        trained_model_op: The trained `pipekit.Operator` returned by
            the trial's `TrainingLoop.run`.
        artifact: The trial's `TrainingArtifact` (or `_LocalArtifact`
            fallback when `pipekit-experiment` is absent).
        final_metric: The value of ``select_metric`` pulled from
            ``artifact.backend_info["final_metrics"]`` — the number
            this trial is ranked by.
        sweep_id: Stable identifier shared by every trial in this
            sweep (defaults to a uuid4 hex string).
        trial_index: **Global** 0-based index across the cumulative
            trial stream represented by the carry-state. For a fresh
            sweep it starts at 0; when a `HyperSweep` is composed in
            `Sequential` and resumes a `SweepCarryState`, this index
            continues from `state.completed` at entry — so
            `trial_index` always identifies the trial unambiguously
            regardless of staging.
    """

    params: dict[str, Any]
    trained_model_op: Operator
    artifact: Any
    final_metric: float
    sweep_id: str
    trial_index: int


# ---------------------------------------------------------------------
# SweepCarryState — best-so-far across trials.
# ---------------------------------------------------------------------


class SweepCarryState(CarryState):
    """Carry-state for `HyperSweep`.

    Threaded across trials inside ``_apply`` so the sweep composes
    into `Sequential` (e.g. coarse-then-fine sweep stages share
    best-so-far progress).

    Attributes:
        completed: Cumulative number of trials run across all
            `_apply` calls that have consumed this state.
        best_metric: Best metric value observed (None until the
            first trial completes).
        best_trial_index: **Global** index (in the same coordinate
            system as ``completed``) of the trial that produced
            ``best_metric``. When `_apply` is called with an
            incoming `SweepCarryState`, trials in the new stage
            are numbered ``state.completed, state.completed + 1, …``
            so ``best_trial_index`` always identifies the best trial
            unambiguously across coarse → fine sweep stages.
    """

    def __init__(
        self,
        completed: int = 0,
        best_metric: float | None = None,
        best_trial_index: int | None = None,
    ) -> None:
        self.completed = int(completed)
        self.best_metric = best_metric if best_metric is None else float(best_metric)
        self.best_trial_index = best_trial_index


# ---------------------------------------------------------------------
# HyperSweep — orchestrate N training runs.
# ---------------------------------------------------------------------


class HyperSweep(StatefulOperator):
    """Orchestrate N independent `TrainingLoop` runs.

    For each ``params`` dict yielded by ``search_space``, builds a
    `TrainingLoop` via ``loop_template(params)`` and runs it.
    Captures the trained model and artifact per trial. Returns a list
    of `SweepResult` sorted by ``select_metric`` in ``select_mode``
    order.

    v0.2 runs trials sequentially. Parallel execution (multiprocessing)
    lands in v0.3. Optuna / Ray Tune samplers land in v0.2.1 — for now
    any iterable of param dicts works as ``search_space``.

    Args:
        loop_template: Callable ``(params: dict) -> TrainingLoop``.
            Each trial's loop is built fresh; ``loop_template`` should
            be deterministic given ``params`` for reproducibility.
        search_space: A `ParameterGrid` or any iterable of param
            dicts. The latter accommodates Optuna trial generators in
            v0.2.1.
        select_metric: Key to look up in
            ``artifact.backend_info["final_metrics"]``. Default
            ``"loss"`` — match this to whatever your loss surfaces
            (e.g. ``"mse"`` for `MSE`).
        select_mode: ``"min"`` for losses, ``"max"`` for accuracies.
        n_trials: Optional cap on trials drawn from ``search_space``.
            ``None`` runs the entire search space. Useful when
            ``search_space`` is an unbounded sampler.
        sweep_id: Stable identifier for this sweep; default uuid4 hex.
            Surfaced on every `SweepResult` for tracker provenance.
        on_trial: Optional callback ``(SweepResult, SweepCarryState)
            -> None`` invoked after each trial completes. Useful for
            progress logging or early-stop heuristics.

    Returns from `run()`:
        ``list[SweepResult]`` sorted by ``select_metric``. Best result
        is ``results[0]``.

    Raises:
        ValueError: if the search space is empty, or if a trial's
            artifact doesn't include ``select_metric``.
        TypeError: if ``loop_template(params)`` doesn't return a
            `TrainingLoop`.
    """

    # The sweep itself isn't directly YAML-serialisable because the
    # loop_template is a user closure; mark accordingly.
    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        loop_template: Callable[[dict[str, Any]], TrainingLoop],
        search_space: Iterable[dict[str, Any]],
        select_metric: str = "loss",
        select_mode: Literal["min", "max"] = "min",
        n_trials: int | None = None,
        sweep_id: str | None = None,
        on_trial: Callable[[SweepResult, SweepCarryState], None] | None = None,
    ) -> None:
        if not callable(loop_template):
            raise TypeError(
                "HyperSweep.loop_template must be callable; got "
                f"{type(loop_template).__name__}."
            )
        if select_mode not in {"min", "max"}:
            raise ValueError(
                f"HyperSweep.select_mode must be 'min' or 'max'; got {select_mode!r}."
            )
        if n_trials is not None and n_trials < 1:
            raise ValueError(
                f"HyperSweep.n_trials must be >= 1 or None; got {n_trials}."
            )
        self.loop_template = loop_template
        self.search_space = search_space
        self.select_metric = select_metric
        self.select_mode = select_mode
        self.n_trials = n_trials
        self.sweep_id = sweep_id or uuid.uuid4().hex
        self.on_trial = on_trial
        self.initial_state_fn = SweepCarryState

    # --- StatefulOperator hook -----------------------------------------

    def _apply(
        self,
        carrier: Any = None,
        state: SweepCarryState | None = None,
    ) -> tuple[list[SweepResult], SweepCarryState]:
        """Run all trials; return sorted results + final carry-state.

        ``carrier`` is ignored. ``state`` may be a `SweepCarryState`
        from a prior `_apply` call (used in `Sequential` composition);
        completed counts accumulate.
        """
        del carrier
        state = state or SweepCarryState()
        # Trial indices are GLOBAL across the cumulative trial stream
        # (state.completed at entry), not stage-local. That keeps
        # `SweepResult.trial_index` and `state.best_trial_index`
        # meaningful when a HyperSweep is composed in Sequential and
        # resumes from a non-zero `state.completed` (e.g. coarse →
        # fine sweep stages). Fresh sweeps (state.completed == 0)
        # see indices 0, 1, 2, … as before.
        starting_count = state.completed

        results: list[SweepResult] = []
        for local_index, params in enumerate(self._iter_trials()):
            global_index = starting_count + local_index
            loop = self.loop_template(dict(params))
            if not isinstance(loop, TrainingLoop):
                raise TypeError(
                    "HyperSweep.loop_template must return a "
                    f"TrainingLoop; got {type(loop).__name__}."
                )
            trained_model_op, artifact = loop.run()
            metric_value = self._extract_metric(artifact, global_index, params)

            result = SweepResult(
                params=dict(params),
                trained_model_op=trained_model_op,
                artifact=artifact,
                final_metric=metric_value,
                sweep_id=self.sweep_id,
                trial_index=global_index,
            )
            results.append(result)

            # Update carry-state.
            state.completed += 1
            if state.best_metric is None or self._is_better(
                metric_value, state.best_metric
            ):
                state.best_metric = metric_value
                state.best_trial_index = global_index

            if self.on_trial is not None:
                self.on_trial(result, state)

        sorted_results = sorted(
            results,
            key=lambda r: r.final_metric,
            reverse=self.select_mode == "max",
        )
        return sorted_results, state

    # --- run convenience -----------------------------------------------

    def run(self) -> list[SweepResult]:
        """Run all trials and return them sorted by ``select_metric``."""
        results, _ = self._apply()
        return results

    # --- helpers --------------------------------------------------------

    def _iter_trials(self) -> Iterator[dict[str, Any]]:
        seen = 0
        for params in iter(self.search_space):
            if not isinstance(params, dict):
                raise TypeError(
                    "HyperSweep.search_space must yield dicts; got "
                    f"{type(params).__name__} on trial {seen}."
                )
            yield params
            seen += 1
            if self.n_trials is not None and seen >= self.n_trials:
                return
        if seen == 0:
            raise ValueError("HyperSweep.search_space is empty — no trials to run.")

    def _extract_metric(
        self,
        artifact: Any,
        trial_index: int,
        params: dict[str, Any],
    ) -> float:
        backend_info = getattr(artifact, "backend_info", {}) or {}
        final_metrics = backend_info.get("final_metrics", {}) or {}
        if self.select_metric not in final_metrics:
            raise ValueError(
                f"HyperSweep: trial {trial_index} (params={params!r}) "
                f"artifact has no metric {self.select_metric!r} in "
                f"backend_info['final_metrics']. Available keys: "
                f"{sorted(final_metrics)}."
            )
        return float(final_metrics[self.select_metric])

    def _is_better(self, new: float, current_best: float) -> bool:
        if self.select_mode == "min":
            return new < current_best
        return new > current_best

    def get_config(self) -> dict[str, Any]:
        return {
            "select_metric": self.select_metric,
            "select_mode": self.select_mode,
            "n_trials": self.n_trials,
            "sweep_id": self.sweep_id,
            "search_space": type(self.search_space).__name__,
            "loop_template": getattr(
                self.loop_template, "__name__", repr(self.loop_template)
            ),
        }
