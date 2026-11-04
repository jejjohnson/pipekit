"""End-to-end `HyperSweep` test against the Equinox adapter.

Gated by ``pytest.importorskip("equinox")`` so the test only runs in
the ``[equinox]`` extra environment. Trains a tiny MLP per trial on a
synthetic regression task across a small grid; asserts the best
trial has loss below a sanity threshold and that `sweep_id` propagates.
"""

from __future__ import annotations

import warnings
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


# ---------------------------------------------------------------------
# Trials rank on validation metrics when they have them
# ---------------------------------------------------------------------


def _disagreeing_sweep(**kwargs: Any) -> tuple[HyperSweep, list[float]]:
    """Two trials whose training and validation rankings disagree.

    The validation targets are the untrained model's outputs, so the frozen trial
    (lr=1e-12) is perfect on validation and poor on training, and the other trial
    is the reverse.
    """
    init = eqx.nn.MLP(1, 1, 8, 1, key=jax.random.key(0))
    train = _synth_dataset()
    xs = np.random.default_rng(1).uniform(-1, 1, (16, 1)).astype(np.float32)
    targets = np.asarray(jax.vmap(init)(xs))
    val = IterableDataset(source=list(zip(xs, targets, strict=True)), content_hash="v")
    val_mse: list[float] = []

    class RecordVal:
        def on_eval_end(self, loop, state, metrics) -> None:
            val_mse.append(metrics["mse"])

    def build(params: dict[str, Any]) -> TrainingLoop:
        return TrainingLoop(
            model_op=EquinoxModelOp(init),
            dataset=train,
            val_dataset=val,
            loss=MSE(),
            optimizer_config={"name": "adam", "lr": params["lr"]},
            max_steps=40,
            eval_every_n_steps=15,
            batch_size=8,
            backend="equinox",
            callbacks=(RecordVal(),),
            seed=0,
        )

    space = [{"lr": 1e-12}, {"lr": 1e-2}]
    return HyperSweep(build, space, select_metric="mse", **kwargs), val_mse


def test_sweep_ranks_on_validation_when_trials_have_it():
    sweep, val_mse = _disagreeing_sweep()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = sweep.run()

    assert results[0].params == {"lr": 1e-12}
    # two periodic evals and one final eval per trial; the last one is what ranks
    assert len(val_mse) == 6
    assert results[0].final_metric == pytest.approx(val_mse[2])
    assert results[1].final_metric == pytest.approx(val_mse[5])
    assert results[0].artifact.backend_info["final_eval_metrics"]["mse"] < 1e-6


def test_sweep_select_split_train_keeps_the_training_ranking():
    sweep, _ = _disagreeing_sweep(select_split="train")
    assert sweep.run()[0].params == {"lr": 1e-2}


def test_sweep_without_validation_warns_once_and_ranks_on_training():
    sweep = HyperSweep(_build_loop, [{"lr": 1e-3}, {"lr": 1e-2}], select_metric="mse")
    with pytest.warns(UserWarning, match="last training batch") as caught:
        results = sweep.run()
    assert len([w for w in caught if "last training batch" in str(w.message)]) == 1
    assert results[0].params == {"lr": 1e-2}


def test_sweep_select_split_eval_requires_validation_metrics():
    sweep = HyperSweep(
        _build_loop, [{"lr": 1e-3}], select_metric="mse", select_split="eval"
    )
    with pytest.raises(ValueError, match="final_eval_metrics"):
        sweep.run()


def test_sweep_rejects_an_unknown_select_split():
    with pytest.raises(ValueError, match="select_split"):
        HyperSweep(_build_loop, [{"lr": 1e-3}], select_split="val")
