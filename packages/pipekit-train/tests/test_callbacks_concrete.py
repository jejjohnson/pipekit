"""Tests for the bundled callbacks (`EarlyStopping`, `Checkpoint`, `LogToExperiment`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pipekit import Const
from pipekit_train import Checkpoint, EarlyStopping, LogToExperiment


# --- Helpers --------------------------------------------------------------


@dataclass
class _FakeState:
    model: Any = None
    step: int = 0


class _RecordingTracker:
    """In-memory ExperimentTracker for tests."""

    def __init__(self) -> None:
        self.starts: list[tuple[str, dict]] = []
        self.metrics_log: list[tuple[Any, dict, int]] = []
        self.ends: list[tuple[Any, str]] = []
        self._run_id = 0

    def start_run(self, name: str, config: dict[str, Any]) -> Any:
        self._run_id += 1
        run = ("run", self._run_id)
        self.starts.append((name, dict(config)))
        return run

    def log_metrics(self, run: Any, metrics: dict[str, float], step: int) -> None:
        self.metrics_log.append((run, dict(metrics), step))

    def end_run(self, run: Any, status: str) -> None:
        self.ends.append((run, status))


class _RecordingRegistry:
    """In-memory ModelRegistry for tests."""

    def __init__(self) -> None:
        self.stored: list[Any] = []

    def store(self, model_op: Any) -> str:
        self.stored.append(model_op)
        return f"hash{len(self.stored)}"


# --- EarlyStopping --------------------------------------------------------


def test_early_stopping_does_not_stop_on_improvement():
    cb = EarlyStopping(metric="val_loss", patience=2)
    state = _FakeState()
    cb.on_eval_end(None, state, {"val_loss": 1.0})
    cb.on_eval_end(None, state, {"val_loss": 0.8})
    cb.on_eval_end(None, state, {"val_loss": 0.6})
    assert not cb.should_stop


def test_early_stopping_triggers_after_patience_exhausted():
    cb = EarlyStopping(metric="val_loss", patience=2)
    state = _FakeState()
    cb.on_eval_end(None, state, {"val_loss": 1.0})  # best=1.0
    cb.on_eval_end(None, state, {"val_loss": 1.5})  # wait=1
    assert not cb.should_stop
    cb.on_eval_end(None, state, {"val_loss": 1.2})  # wait=2 → stop
    assert cb.should_stop


def test_early_stopping_mode_max():
    cb = EarlyStopping(metric="acc", patience=1, mode="max")
    state = _FakeState()
    cb.on_eval_end(None, state, {"acc": 0.5})  # best=0.5
    cb.on_eval_end(None, state, {"acc": 0.4})  # wait=1 → stop
    assert cb.should_stop


def test_early_stopping_min_delta_required_improvement():
    cb = EarlyStopping(metric="loss", patience=1, min_delta=0.1)
    state = _FakeState()
    cb.on_eval_end(None, state, {"loss": 1.0})  # best=1.0
    cb.on_eval_end(None, state, {"loss": 0.95})  # not enough → wait=1 → stop
    assert cb.should_stop


def test_early_stopping_silently_skips_when_metric_missing():
    cb = EarlyStopping(metric="val_loss", patience=1)
    state = _FakeState()
    cb.on_eval_end(None, state, {"other_metric": 0.5})
    assert not cb.should_stop
    assert cb.best is None


def test_early_stopping_reset_clears_state():
    cb = EarlyStopping(metric="loss", patience=1)
    state = _FakeState()
    cb.on_eval_end(None, state, {"loss": 1.0})
    cb.on_eval_end(None, state, {"loss": 2.0})
    assert cb.should_stop
    cb.reset()
    assert not cb.should_stop and cb.best is None and cb.wait == 0


def test_early_stopping_rejects_bad_args():
    with pytest.raises(ValueError, match="mode"):
        EarlyStopping(metric="m", mode="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(metric="m", patience=0)
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStopping(metric="m", min_delta=-0.1)


def test_early_stopping_runtime_state_not_in_config():
    cb = EarlyStopping(metric="m")
    config = cb.get_config()
    assert "should_stop" not in config
    assert "best" not in config
    assert "wait" not in config


def test_early_stopping_apply_raises():
    cb = EarlyStopping(metric="m")
    with pytest.raises(TypeError, match="callback"):
        cb(None)


# --- Checkpoint -----------------------------------------------------------


def test_checkpoint_should_snapshot_predicate():
    cb = Checkpoint(every_n_steps=100)
    assert not cb.should_snapshot(0)  # step 0 never snapshots
    assert not cb.should_snapshot(50)
    assert cb.should_snapshot(100)
    assert cb.should_snapshot(200)
    assert not cb.should_snapshot(150)


def test_checkpoint_stores_final_when_registry_set():
    reg = _RecordingRegistry()
    cb = Checkpoint(every_n_steps=100, registry=reg)
    model = Const(7)
    cb.on_train_end(loop=None, state=_FakeState(model=model))
    assert reg.stored == [model]
    assert cb.last_hash == "hash1"
    assert cb.last_uri == "registry://hash1"


def test_checkpoint_no_store_when_registry_unset():
    cb = Checkpoint(every_n_steps=100)
    cb.on_train_end(loop=None, state=_FakeState(model=Const(7)))
    assert cb.last_hash is None and cb.last_uri is None


def test_checkpoint_no_op_when_state_has_no_model():
    reg = _RecordingRegistry()
    cb = Checkpoint(every_n_steps=100, registry=reg)
    cb.on_train_end(loop=None, state=object())  # no `model` attr
    assert reg.stored == []


def test_checkpoint_rejects_bad_registry():
    class _NoStore:
        pass

    cb = Checkpoint(every_n_steps=100, registry=_NoStore())
    with pytest.raises(TypeError, match="ModelRegistry"):
        cb.on_train_end(loop=None, state=_FakeState(model=Const(7)))


def test_checkpoint_rejects_invalid_args():
    with pytest.raises(ValueError, match="every_n_steps"):
        Checkpoint(every_n_steps=0)
    with pytest.raises(ValueError, match="keep_last"):
        Checkpoint(keep_last=0)


def test_checkpoint_runtime_state_not_in_config():
    cb = Checkpoint(every_n_steps=100)
    config = cb.get_config()
    assert "last_uri" not in config
    assert "last_hash" not in config


def test_checkpoint_forbid_in_yaml():
    """Carrying a runtime registry handle, Checkpoint must not round-trip."""
    assert Checkpoint.forbid_in_yaml is True


def test_checkpoint_apply_raises():
    cb = Checkpoint(every_n_steps=100)
    with pytest.raises(TypeError, match="callback"):
        cb(None)


# --- LogToExperiment ------------------------------------------------------


def test_log_to_experiment_full_lifecycle():
    tracker = _RecordingTracker()
    cb = LogToExperiment(tracker=tracker, run_name="test_run")
    cb.on_train_begin(loop=None, state=_FakeState())
    assert cb.run == ("run", 1)
    assert tracker.starts == [("test_run", {})]

    cb.on_step_end(None, _FakeState(step=10), {"loss": 0.5})
    cb.on_eval_end(None, _FakeState(step=10), {"val_loss": 0.6})
    cb.on_train_end(None, _FakeState(step=100))

    assert tracker.metrics_log == [
        (("run", 1), {"loss": 0.5}, 10),
        (("run", 1), {"eval/val_loss": 0.6}, 10),
    ]
    assert tracker.ends == [(("run", 1), "completed")]


def test_log_to_experiment_merges_loop_config_with_extra():
    tracker = _RecordingTracker()
    cb = LogToExperiment(tracker=tracker, run_name="r", config={"extra": "x"})

    class _Loop:
        def get_config(self):
            return {"backend": "equinox", "max_steps": 10}

    cb.on_train_begin(loop=_Loop(), state=_FakeState())
    started = tracker.starts[0][1]
    assert started == {"backend": "equinox", "max_steps": 10, "extra": "x"}


def test_log_to_experiment_no_metric_log_before_start_run():
    tracker = _RecordingTracker()
    cb = LogToExperiment(tracker=tracker, run_name="r")
    cb.on_step_end(None, _FakeState(step=5), {"loss": 1.0})
    assert tracker.metrics_log == []


def test_log_to_experiment_skips_empty_metrics():
    tracker = _RecordingTracker()
    cb = LogToExperiment(tracker=tracker, run_name="r")
    cb.on_train_begin(None, _FakeState())
    cb.on_step_end(None, _FakeState(step=10), {})
    cb.on_eval_end(None, _FakeState(step=10), {})
    assert tracker.metrics_log == []


def test_log_to_experiment_rejects_bad_tracker():
    class _Bad:
        pass

    with pytest.raises(TypeError, match="ExperimentTracker"):
        LogToExperiment(tracker=_Bad(), run_name="r")


def test_log_to_experiment_runtime_state_not_in_config():
    tracker = _RecordingTracker()
    cb = LogToExperiment(tracker=tracker, run_name="r")
    config = cb.get_config()
    assert "run" not in config


def test_log_to_experiment_forbid_in_yaml():
    assert LogToExperiment.forbid_in_yaml is True


def test_log_to_experiment_apply_raises():
    tracker = _RecordingTracker()
    cb = LogToExperiment(tracker=tracker, run_name="r")
    with pytest.raises(TypeError, match="callback"):
        cb(None)
