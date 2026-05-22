"""Tests for `pipekit_experiment.run`."""

from __future__ import annotations

from datetime import datetime

from pipekit_experiment import Run, RunArtifacts, RunMetrics


def test_run_dataclass_defaults():
    r = Run(run_id="abc", pipeline_hash="h1")
    assert r.run_id == "abc"
    assert r.pipeline_hash == "h1"
    assert r.status == "running"
    assert r.ended_at is None


def test_run_to_dict_serialises_datetimes():
    r = Run(run_id="abc", pipeline_hash="h1")
    d = r.to_dict()
    assert isinstance(d["started_at"], str)
    # Round-trip parse
    datetime.fromisoformat(d["started_at"])
    assert d["ended_at"] is None


def test_run_metrics_log_and_latest():
    m = RunMetrics()
    m.log("loss", 0.5, step=0)
    m.log("loss", 0.4, step=1)
    m.log("acc", 0.8, step=1)
    assert m.values["loss"] == [(0, 0.5), (1, 0.4)]
    assert m.latest() == {"loss": 0.4, "acc": 0.8}


def test_run_metrics_clear():
    m = RunMetrics()
    m.log("loss", 0.5, 0)
    m.clear()
    assert m.values == {}
    assert m.latest() == {}


def test_run_artifacts_attach_and_get():
    a = RunArtifacts()
    a.attach("model", "s3://bucket/m.bin")
    assert a.get("model") == "s3://bucket/m.bin"
    assert a.get("missing") is None
    a.attach("model", "s3://bucket/m2.bin")  # overwrite
    assert a.get("model") == "s3://bucket/m2.bin"


def test_run_artifacts_clear():
    a = RunArtifacts()
    a.attach("model", "s3://x")
    a.clear()
    assert a.items == {}
