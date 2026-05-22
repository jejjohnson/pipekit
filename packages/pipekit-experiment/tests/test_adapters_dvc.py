"""Tests for `pipekit_experiment.adapters.dvc`.

The DVC adapter shells out to the ``dvc`` CLI. We don't require DVC
to be installed in CI — the tests exercise the pointer-file parsing
and the missing-binary error path, and mock subprocess for the CLI
calls.
"""

from __future__ import annotations

import pytest
from pipekit_experiment.adapters.dvc import DVCDatasetVersioning


def test_hash_reads_md5_from_pointer(tmp_path):
    target = tmp_path / "data.parquet.dvc"
    target.write_text("outs:\n- md5: abc123\n  size: 1024\n  path: data.parquet\n")
    dvc = DVCDatasetVersioning(repo=tmp_path)
    assert dvc.hash("data.parquet") == "abc123"


def test_hash_reads_hash_field(tmp_path):
    target = tmp_path / "x.dvc"
    target.write_text("outs:\n- hash: deadbeef\n  path: x\n")
    dvc = DVCDatasetVersioning(repo=tmp_path)
    assert dvc.hash("x") == "deadbeef"


def test_hash_missing_pointer_raises(tmp_path):
    dvc = DVCDatasetVersioning(repo=tmp_path)
    with pytest.raises(FileNotFoundError):
        dvc.hash("absent")


def test_hash_malformed_pointer_raises(tmp_path):
    (tmp_path / "x.dvc").write_text("nothing useful here\n")
    dvc = DVCDatasetVersioning(repo=tmp_path)
    with pytest.raises(ValueError):
        dvc.hash("x")


def test_record_hash_in_sink(tmp_path):
    (tmp_path / "x.dvc").write_text("outs:\n- md5: aa\n  path: x\n")
    dvc = DVCDatasetVersioning(repo=tmp_path)
    sink: dict[str, str] = {}
    dvc.record_hash_in("x", sink)
    assert sink == {"x": "aa"}


def test_missing_dvc_executable_raises(tmp_path):
    dvc = DVCDatasetVersioning(
        repo=tmp_path, dvc_executable="dvc-that-definitely-does-not-exist"
    )
    with pytest.raises(RuntimeError, match="not found on PATH"):
        dvc.add("anything")


def test_pointer_path_accepts_dvc_suffix(tmp_path):
    (tmp_path / "x.dvc").write_text("outs:\n- md5: zz\n  path: x\n")
    dvc = DVCDatasetVersioning(repo=tmp_path)
    # Passing the .dvc path directly should also work.
    assert dvc.hash("x.dvc") == "zz"
