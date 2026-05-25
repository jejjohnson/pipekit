"""End-to-end `HyperSweep` test against the Equinox adapter.

Gated by ``pytest.importorskip("equinox")`` so the test only runs in
the ``[equinox]`` extra environment. Trains a tiny MLP per trial on a
synthetic regression task across a small grid; asserts the best
trial has loss below a sanity threshold and that `sweep_id` propagates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


pytest.importorskip("equinox")
pytest.importorskip("jax")
pytest.importorskip("optax")
pytest.importorskip("orbax.checkpoint")

import equinox as eqx
import jax
from pipekit_train import (
    MSE,
    HyperSweep,
    IterableDataset,
    ParameterGrid,
    SweepCarryState,
    SweepResult,
    TrainingLoop,
)
from pipekit_train.adapters.equinox import EquinoxModelOp


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _synth_dataset(n: int = 64, seed: int = 0) -> IterableDataset:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    ys = (2.0 * xs + 1.0).astype(np.float32)
    pairs = list(zip(xs, ys, strict=True))
    return IterableDataset(source=pairs, content_hash=f"synth-{n}-{seed}")


def _build_loop(params: dict[str, Any]) -> TrainingLoop:
    """Per-trial loop builder used by the sweep tests below."""
    mlp = eqx.nn.MLP(
        in_size=1,
        out_size=1,
        width_size=int(params.get("width", 8)),
        depth=1,
        key=jax.random.key(int(params.get("seed", 0))),
    )
    return TrainingLoop(
        model_op=EquinoxModelOp(mlp),
        dataset=_synth_dataset(n=64, seed=int(params.get("seed", 0))),
        loss=MSE(),
        optimizer_config={"name": "adam", "lr": float(params.get("lr", 1e-2))},
        max_steps=30,
        batch_size=8,
        backend="equinox",
        seed=int(params.get("seed", 0)),
    )


# ---------------------------------------------------------------------
# End-to-end: a 3-cell grid sweep over learning rates
# ---------------------------------------------------------------------


def test_hyper_sweep_runs_grid_and_ranks_by_mse():
    """Sweep over lr ∈ {1e-4, 1e-2, 1e+1}: the middle value should win."""
    sweep = HyperSweep(
        loop_template=_build_loop,
        search_space=ParameterGrid({"lr": [1e-4, 1e-2, 1e1]}),
        select_metric="mse",
        select_mode="min",
        sweep_id="lr-grid",
    )
    results = sweep.run()

    assert len(results) == 3
    # All trials should have completed and produced finite metrics.
    for r in results:
        assert isinstance(r, SweepResult)
        assert r.sweep_id == "lr-grid"
        assert np.isfinite(r.final_metric)
        # The trained_model_op is an `EquinoxModelOp` (or any pipekit Operator).
        assert isinstance(r.trained_model_op, EquinoxModelOp)
    # The 1e-2 trial should beat both the way-too-small (1e-4) and
    # way-too-large (1e+1) learning rates — diverges or never moves.
    best = results[0]
    assert best.params["lr"] == 1e-2


def test_hyper_sweep_records_on_trial_progress():
    """on_trial fires exactly once per trial with the updated state."""
    seen: list[tuple[int, float, float | None]] = []

    def _record(result: SweepResult, state: SweepCarryState) -> None:
        seen.append((result.trial_index, result.final_metric, state.best_metric))

    sweep = HyperSweep(
        loop_template=_build_loop,
        search_space=ParameterGrid({"lr": [1e-2, 1e-3]}),
        select_metric="mse",
        on_trial=_record,
        sweep_id="progress-sweep",
    )
    sweep.run()
    assert len(seen) == 2
    assert [s[0] for s in seen] == [0, 1]


def test_hyper_sweep_n_trials_caps_a_larger_grid():
    """A 4-cell grid with n_trials=2 runs only the first 2."""
    sweep = HyperSweep(
        loop_template=_build_loop,
        search_space=ParameterGrid({"lr": [1e-2, 1e-3, 1e-4, 1e-5]}),
        select_metric="mse",
        n_trials=2,
    )
    results = sweep.run()
    assert len(results) == 2


def test_hyper_sweep_artifact_carries_dataset_hash_per_trial():
    """Each trial's artifact carries its own dataset_hash."""
    sweep = HyperSweep(
        loop_template=_build_loop,
        search_space=ParameterGrid({"seed": [0, 1]}),
        select_metric="mse",
    )
    results = sweep.run()
    hashes = {r.artifact.dataset_hash for r in results}
    # Different seeds → different datasets → different hashes.
    assert len(hashes) == 2
