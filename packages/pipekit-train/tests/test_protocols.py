"""Structural checks for `pipekit_train` protocols.

`Loss`, `Callback`, and `MetricWriter` are runtime-checkable
Protocols; structural conformance must succeed without inheritance.
"""

from __future__ import annotations

from typing import Any

from pipekit_train import Callback, JSONLWriter, Loss, MetricWriter


class _PlainLoss:
    def __call__(self, predicted: Any, target: Any) -> float:
        return 0.0


class _PlainCallback:
    def on_train_begin(self, loop: Any, state: Any) -> None: ...

    def on_step_end(self, loop: Any, state: Any, metrics: dict[str, float]) -> None: ...

    def on_epoch_end(
        self, loop: Any, state: Any, metrics: dict[str, float]
    ) -> None: ...

    def on_eval_end(
        self, loop: Any, state: Any, eval_metrics: dict[str, float]
    ) -> None: ...

    def on_train_end(self, loop: Any, state: Any) -> None: ...


class _PlainWriter:
    def write(self, step: int, metrics: dict[str, float]) -> None: ...

    def close(self) -> None: ...


def test_loss_runtime_checkable():
    assert isinstance(_PlainLoss(), Loss)


def test_callback_runtime_checkable():
    assert isinstance(_PlainCallback(), Callback)


def test_metric_writer_runtime_checkable():
    assert isinstance(_PlainWriter(), MetricWriter)


def test_jsonl_writer_satisfies_metric_writer(tmp_path):
    writer = JSONLWriter(path=tmp_path / "metrics.jsonl")
    assert isinstance(writer, MetricWriter)
