"""Tests for `pipekit_train.writer.JSONLWriter`."""

from __future__ import annotations

import json
from pathlib import Path

from pipekit_train import JSONLWriter


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_jsonl_writer_appends_records(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = JSONLWriter(path=path, flush_every=1)
    writer.write(0, {"loss": 1.0})
    writer.write(1, {"loss": 0.5, "acc": 0.9})
    writer.close()

    records = _read_jsonl(path)
    assert [r["step"] for r in records] == [0, 1]
    assert records[0]["metrics"] == {"loss": 1.0}
    assert records[1]["metrics"] == {"loss": 0.5, "acc": 0.9}
    assert "timestamp" in records[0]


def test_jsonl_writer_coerces_to_float(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = JSONLWriter(path=path, flush_every=1)
    writer.write(0, {"int_metric": 7})
    writer.close()

    record = _read_jsonl(path)[0]
    assert record["metrics"]["int_metric"] == 7.0
    assert isinstance(record["metrics"]["int_metric"], float)


def test_jsonl_writer_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "metrics.jsonl"
    writer = JSONLWriter(path=path, flush_every=1)
    writer.write(0, {"loss": 1.0})
    writer.close()
    assert path.exists()


def test_jsonl_writer_close_is_idempotent(tmp_path):
    writer = JSONLWriter(path=tmp_path / "metrics.jsonl", flush_every=1)
    writer.write(0, {"loss": 1.0})
    writer.close()
    writer.close()  # second close is a no-op


def test_jsonl_writer_flush_every_buffers_writes(tmp_path):
    """flush_every=2 means the file is flushed every 2 writes."""
    path = tmp_path / "metrics.jsonl"
    writer = JSONLWriter(path=path, flush_every=2)
    writer.write(0, {"loss": 1.0})
    writer.write(1, {"loss": 0.5})  # this triggers the flush
    # Records are written immediately; flush only forces fsync.
    # We can't easily observe the buffer, but we can check both lines
    # are present after the second write.
    assert len(_read_jsonl(path)) == 2
    writer.close()
