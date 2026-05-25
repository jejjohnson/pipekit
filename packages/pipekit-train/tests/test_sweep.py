"""Unit tests for `pipekit_train.sweep`.

Covers `ParameterGrid` iteration / sizing / validation and
`HyperSweep` construction validation. End-to-end sweep behaviour
against the Equinox adapter lives in
``tests/adapters/test_sweep_equinox.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pipekit_train import (
    HyperSweep,
    ParameterGrid,
    SweepCarryState,
    SweepResult,
)
from pipekit_train.loop import TrainingLoop


# ---------------------------------------------------------------------
# ParameterGrid
# ---------------------------------------------------------------------


def test_parameter_grid_yields_cartesian_product():
    grid = ParameterGrid({"lr": [1e-3, 1e-2], "width": [8, 16]})
    combos = list(grid)
    assert len(combos) == 4
    assert {"lr": 1e-3, "width": 8} in combos
    assert {"lr": 1e-3, "width": 16} in combos
    assert {"lr": 1e-2, "width": 8} in combos
    assert {"lr": 1e-2, "width": 16} in combos


def test_parameter_grid_single_axis():
    grid = ParameterGrid({"lr": [1e-3, 1e-2, 1e-1]})
    combos = list(grid)
    assert combos == [{"lr": 1e-3}, {"lr": 1e-2}, {"lr": 1e-1}]


def test_parameter_grid_len_product_of_axes():
    grid = ParameterGrid({"a": [1, 2], "b": [3, 4, 5], "c": [True, False]})
    assert len(grid) == 12
    assert len(list(grid)) == 12


def test_parameter_grid_empty_grid_yields_nothing():
    grid = ParameterGrid({})
    assert list(grid) == []
    assert len(grid) == 0


def test_parameter_grid_rejects_empty_axis():
    with pytest.raises(ValueError, match="empty"):
        ParameterGrid({"lr": []})


def test_parameter_grid_rejects_non_list_axis():
    with pytest.raises(TypeError, match="list/tuple"):
        ParameterGrid({"lr": "not a list"})  # type: ignore[dict-item]


def test_parameter_grid_rejects_non_dict():
    with pytest.raises(TypeError, match="dict"):
        ParameterGrid([("lr", [1e-3])])  # type: ignore[arg-type]


def test_parameter_grid_apply_raises_typeerror():
    """ParameterGrid is iterated, not called."""
    grid = ParameterGrid({"lr": [1e-3]})
    with pytest.raises(TypeError, match="iterated"):
        grid()


def test_parameter_grid_iteration_is_reusable():
    grid = ParameterGrid({"lr": [1e-3, 1e-2]})
    assert list(grid) == list(grid)


# ---------------------------------------------------------------------
# SweepCarryState
# ---------------------------------------------------------------------


def test_sweep_carry_state_defaults():
    state = SweepCarryState()
    assert state.completed == 0
    assert state.best_metric is None
    assert state.best_trial_index is None


def test_sweep_carry_state_explicit_init():
    state = SweepCarryState(completed=3, best_metric=0.05, best_trial_index=1)
    assert state.completed == 3
    assert state.best_metric == 0.05
    assert state.best_trial_index == 1


# ---------------------------------------------------------------------
# HyperSweep — construction validation
# ---------------------------------------------------------------------


def _make_dummy_loop(_params: dict[str, Any]) -> TrainingLoop:
    # Not actually used — these tests cover construction validation.
    raise AssertionError("dummy loop_template should not be called")


def test_hyper_sweep_rejects_non_callable_loop_template():
    with pytest.raises(TypeError, match="callable"):
        HyperSweep(
            loop_template="not a function",  # type: ignore[arg-type]
            search_space=ParameterGrid({"lr": [1e-3]}),
        )


def test_hyper_sweep_rejects_bad_select_mode():
    with pytest.raises(ValueError, match="select_mode"):
        HyperSweep(
            loop_template=_make_dummy_loop,
            search_space=ParameterGrid({"lr": [1e-3]}),
            select_mode="other",  # type: ignore[arg-type]
        )


def test_hyper_sweep_rejects_nonpositive_n_trials():
    with pytest.raises(ValueError, match="n_trials"):
        HyperSweep(
            loop_template=_make_dummy_loop,
            search_space=ParameterGrid({"lr": [1e-3]}),
            n_trials=0,
        )


def test_hyper_sweep_default_sweep_id_is_unique():
    a = HyperSweep(
        loop_template=_make_dummy_loop,
        search_space=ParameterGrid({"lr": [1e-3]}),
    )
    b = HyperSweep(
        loop_template=_make_dummy_loop,
        search_space=ParameterGrid({"lr": [1e-3]}),
    )
    assert a.sweep_id != b.sweep_id
    assert len(a.sweep_id) > 0


def test_hyper_sweep_explicit_sweep_id_preserved():
    sweep = HyperSweep(
        loop_template=_make_dummy_loop,
        search_space=ParameterGrid({"lr": [1e-3]}),
        sweep_id="my-sweep-2026-05-25",
    )
    assert sweep.sweep_id == "my-sweep-2026-05-25"


def test_hyper_sweep_get_config_surfaces_key_fields():
    sweep = HyperSweep(
        loop_template=_make_dummy_loop,
        search_space=ParameterGrid({"lr": [1e-3]}),
        select_metric="mse",
        select_mode="min",
        n_trials=2,
        sweep_id="sweep-x",
    )
    config = sweep.get_config()
    assert config["select_metric"] == "mse"
    assert config["select_mode"] == "min"
    assert config["n_trials"] == 2
    assert config["sweep_id"] == "sweep-x"
    assert config["search_space"] == "ParameterGrid"
    assert config["loop_template"] == "_make_dummy_loop"


def test_hyper_sweep_forbid_in_yaml():
    """HyperSweep carries a user closure; it must not claim YAML round-trip."""
    assert HyperSweep.forbid_in_yaml is True


# ---------------------------------------------------------------------
# HyperSweep — orchestration tests with a fake loop (no JAX dep)
#
# We exercise the orchestration logic (_extract_metric, ranking,
# carry-state, on_trial callback, n_trials cap) without actually
# training anything. The Equinox-backed sweep test lives separately.
# ---------------------------------------------------------------------


class _FakeArtifact:
    def __init__(self, metric_value: float) -> None:
        self.backend_info = {"final_metrics": {"loss": float(metric_value)}}


class _FakeTrainedOp:
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params


class _FakeLoop:
    """Stand-in for `TrainingLoop`. Returns a deterministic metric
    derived from params so the sweep's ranking can be asserted."""

    def __init__(self, params: dict[str, Any], metric: float) -> None:
        self.params = params
        self.metric = metric

    def run(self) -> tuple[_FakeTrainedOp, _FakeArtifact]:
        return _FakeTrainedOp(self.params), _FakeArtifact(self.metric)


def _fake_loop_template(metric_fn: Any):
    def _build(params: dict[str, Any]) -> _FakeLoop:
        return _FakeLoop(params, metric_fn(params))

    return _build


def test_hyper_sweep_rejects_non_trainingloop_return(monkeypatch):
    """If loop_template returns something that isn't a TrainingLoop,
    HyperSweep raises a clear TypeError."""
    sweep = HyperSweep(
        loop_template=_fake_loop_template(lambda _p: 1.0),
        search_space=ParameterGrid({"lr": [1e-3]}),
        select_metric="loss",
    )
    with pytest.raises(TypeError, match="TrainingLoop"):
        sweep.run()


# To exercise the orchestration without a real TrainingLoop, swap out
# the isinstance check via monkeypatch — keeps the test JAX-free.
def test_hyper_sweep_orchestration_with_fake_loop(monkeypatch):
    """Mock the isinstance check; verify ranking + carry-state + sweep_id."""
    import pipekit_train.sweep as sweep_mod

    monkeypatch.setattr(
        sweep_mod, "TrainingLoop", _FakeLoop
    )  # bypass the strict isinstance gate

    # Params → metric: lr=1e-3 → loss=0.5; lr=1e-2 → loss=0.1; lr=1e-1 → loss=2.0
    metrics = {1e-3: 0.5, 1e-2: 0.1, 1e-1: 2.0}
    sweep = HyperSweep(
        loop_template=_fake_loop_template(lambda p: metrics[p["lr"]]),
        search_space=ParameterGrid({"lr": [1e-3, 1e-2, 1e-1]}),
        select_metric="loss",
        select_mode="min",
        sweep_id="ranked-sweep",
    )
    results = sweep.run()

    assert len(results) == 3
    assert isinstance(results[0], SweepResult)
    assert results[0].params["lr"] == 1e-2  # best (min)
    assert results[0].final_metric == 0.1
    assert results[-1].params["lr"] == 1e-1  # worst
    assert all(r.sweep_id == "ranked-sweep" for r in results)
    # trial_index reflects original iteration order, not sorted order.
    assert {r.trial_index for r in results} == {0, 1, 2}


def test_hyper_sweep_select_mode_max(monkeypatch):
    """select_mode='max' inverts the ranking (for accuracy-like metrics)."""
    import pipekit_train.sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "TrainingLoop", _FakeLoop)

    metrics = {1: 0.7, 2: 0.95, 3: 0.4}

    def loop_template(params: dict[str, Any]) -> _FakeLoop:
        return _FakeLoop(params, metrics[params["seed"]])

    sweep = HyperSweep(
        loop_template=loop_template,
        search_space=ParameterGrid({"seed": [1, 2, 3]}),
        select_metric="loss",
        select_mode="max",
    )
    results = sweep.run()
    assert results[0].params["seed"] == 2  # best (max)
    assert results[0].final_metric == 0.95
    assert results[-1].params["seed"] == 3  # worst


def test_hyper_sweep_n_trials_caps_iteration(monkeypatch):
    """n_trials=2 over a 5-cell grid runs only the first 2."""
    import pipekit_train.sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "TrainingLoop", _FakeLoop)

    call_count = 0

    def loop_template(params: dict[str, Any]) -> _FakeLoop:
        nonlocal call_count
        call_count += 1
        return _FakeLoop(params, 1.0)

    sweep = HyperSweep(
        loop_template=loop_template,
        search_space=ParameterGrid({"lr": [1, 2, 3, 4, 5]}),
        select_metric="loss",
        n_trials=2,
    )
    results = sweep.run()
    assert len(results) == 2
    assert call_count == 2  # only 2 trials actually ran


def test_hyper_sweep_on_trial_callback_fires_per_trial(monkeypatch):
    """on_trial sees each completed SweepResult + the running SweepCarryState."""
    import pipekit_train.sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "TrainingLoop", _FakeLoop)

    seen: list[tuple[int, int, float | None]] = []

    def _on_trial(result: SweepResult, state: SweepCarryState) -> None:
        seen.append((result.trial_index, state.completed, state.best_metric))

    sweep = HyperSweep(
        loop_template=_fake_loop_template(lambda p: float(p["lr"])),
        search_space=ParameterGrid({"lr": [3.0, 1.0, 2.0]}),
        select_metric="loss",
        on_trial=_on_trial,
    )
    sweep.run()
    # Three trials in iteration order:
    #   trial 0: lr=3.0 → loss 3.0, best=3.0
    #   trial 1: lr=1.0 → loss 1.0, best=1.0 (improved)
    #   trial 2: lr=2.0 → loss 2.0, best=1.0 (no improvement)
    assert [s[0] for s in seen] == [0, 1, 2]
    assert [s[1] for s in seen] == [1, 2, 3]
    assert seen[0][2] == 3.0
    assert seen[1][2] == 1.0
    assert seen[2][2] == 1.0


def test_hyper_sweep_missing_metric_raises_clear_error(monkeypatch):
    import pipekit_train.sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "TrainingLoop", _FakeLoop)

    sweep = HyperSweep(
        loop_template=_fake_loop_template(lambda _p: 1.0),
        search_space=ParameterGrid({"lr": [1e-3]}),
        select_metric="this_key_does_not_exist",
    )
    with pytest.raises(ValueError, match="this_key_does_not_exist"):
        sweep.run()


def test_hyper_sweep_empty_search_space_raises(monkeypatch):
    import pipekit_train.sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "TrainingLoop", _FakeLoop)

    sweep = HyperSweep(
        loop_template=_fake_loop_template(lambda _p: 1.0),
        search_space=iter([]),  # explicitly empty
        select_metric="loss",
    )
    with pytest.raises(ValueError, match="empty"):
        sweep.run()


def test_hyper_sweep_rejects_non_dict_param_yield(monkeypatch):
    import pipekit_train.sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "TrainingLoop", _FakeLoop)

    sweep = HyperSweep(
        loop_template=_fake_loop_template(lambda _p: 1.0),
        search_space=iter([{"lr": 1e-3}, "not a dict"]),  # type: ignore[list-item]
        select_metric="loss",
    )
    with pytest.raises(TypeError, match="dicts"):
        sweep.run()


def test_hyper_sweep_is_stateful_operator():
    """HyperSweep is a StatefulOperator so it composes into Sequential."""
    from pipekit import StatefulOperator

    sweep = HyperSweep(
        loop_template=_make_dummy_loop,
        search_space=ParameterGrid({"lr": [1e-3]}),
    )
    assert isinstance(sweep, StatefulOperator)
    assert HyperSweep._is_stateful is True


# ---------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------


def test_resumed_sweep_uses_global_trial_indices(monkeypatch):
    """Regression: PR #22 P2 review — when `_apply` runs with an
    incoming `SweepCarryState` (Sequential composition / coarse-fine
    staging), the trial_index on each new SweepResult AND
    state.best_trial_index must continue the global trial-stream
    numbering (state.completed at entry), not restart at 0.
    """
    import pipekit_train.sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "TrainingLoop", _FakeLoop)

    # Simulate that a prior sweep stage already completed 5 trials
    # and found best_metric=0.4 at global index 3.
    incoming = SweepCarryState(completed=5, best_metric=0.4, best_trial_index=3)

    metrics = {"a": 0.9, "b": 0.1, "c": 0.5}
    sweep = HyperSweep(
        loop_template=_fake_loop_template(lambda p: metrics[p["v"]]),
        search_space=ParameterGrid({"v": ["a", "b", "c"]}),
        select_metric="loss",
        select_mode="min",
    )

    results, state_out = sweep._apply(state=incoming)

    # The 3 new trials should be globally indexed 5, 6, 7.
    new_indices = sorted(r.trial_index for r in results)
    assert new_indices == [5, 6, 7]

    # state.completed accumulates: 5 + 3 = 8.
    assert state_out.completed == 8

    # The "b" trial scored 0.1, which beats the incoming best of 0.4.
    # Its global index is 6 (1st new trial = 5, 2nd = 6 for "b"), so
    # state.best_trial_index must be 6 — NOT 1 (the stage-local index).
    assert state_out.best_metric == 0.1
    assert state_out.best_trial_index == 6


def test_fresh_sweep_trial_indices_start_at_zero(monkeypatch):
    """Sanity: a fresh sweep (no incoming state) still numbers
    0, 1, 2, … so the regression above hasn't broken the base case.
    """
    import pipekit_train.sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "TrainingLoop", _FakeLoop)

    metrics = {"a": 0.5, "b": 0.2}
    sweep = HyperSweep(
        loop_template=_fake_loop_template(lambda p: metrics[p["v"]]),
        search_space=ParameterGrid({"v": ["a", "b"]}),
        select_metric="loss",
        select_mode="min",
    )
    results = sweep.run()
    assert sorted(r.trial_index for r in results) == [0, 1]
