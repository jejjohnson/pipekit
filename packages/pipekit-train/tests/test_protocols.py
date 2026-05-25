"""Structural checks for `pipekit_train` protocols.

`Loss` and `MetricWriter` are runtime-checkable Protocols;
structural conformance must succeed without inheritance.

`Callback` is **not** runtime-checkable on purpose — the design
documents the five hooks as optional and adapters dispatch via
``getattr``. The Callback tests therefore pin the partial-
implementation contract by checking method presence directly.
"""

from __future__ import annotations

from typing import Any

from pipekit_train import Callback, JSONLWriter, Loss, MetricWriter


class _PlainLoss:
    def __call__(self, predicted: Any, target: Any) -> float:
        return 0.0


class _FullCallback:
    """A callback implementing all five hooks."""

    def on_train_begin(self, loop: Any, state: Any) -> None: ...

    def on_step_end(self, loop: Any, state: Any, metrics: dict[str, float]) -> None: ...

    def on_epoch_end(
        self, loop: Any, state: Any, metrics: dict[str, float]
    ) -> None: ...

    def on_eval_end(
        self, loop: Any, state: Any, eval_metrics: dict[str, float]
    ) -> None: ...

    def on_train_end(self, loop: Any, state: Any) -> None: ...


class _StepOnlyCallback:
    """A callback implementing only `on_step_end`. Must still be valid."""

    def on_step_end(self, loop: Any, state: Any, metrics: dict[str, float]) -> None: ...


class _PlainWriter:
    def write(self, step: int, metrics: dict[str, float]) -> None: ...

    def close(self) -> None: ...


_HOOKS = (
    "on_train_begin",
    "on_step_end",
    "on_epoch_end",
    "on_eval_end",
    "on_train_end",
)


def test_loss_runtime_checkable():
    assert isinstance(_PlainLoss(), Loss)


def test_metric_writer_runtime_checkable():
    assert isinstance(_PlainWriter(), MetricWriter)


def test_jsonl_writer_satisfies_metric_writer(tmp_path):
    writer = JSONLWriter(path=tmp_path / "metrics.jsonl")
    assert isinstance(writer, MetricWriter)


def test_callback_protocol_describes_full_hook_surface():
    """The Protocol enumerates all five hooks as the type-checker contract."""
    for hook in _HOOKS:
        assert hasattr(Callback, hook), f"Callback Protocol missing {hook!r}"


def test_full_callback_has_all_hooks():
    cb = _FullCallback()
    for hook in _HOOKS:
        assert callable(getattr(cb, hook, None)), f"missing {hook!r}"


def test_partial_callback_is_valid_at_runtime():
    """A callback that implements only one hook is the supported model.

    Adapters dispatch via ``getattr(cb, hook, None)`` — a partial
    implementation just means the missing hooks are no-ops.
    """
    cb = _StepOnlyCallback()
    assert callable(getattr(cb, "on_step_end", None))
    for hook in _HOOKS:
        if hook == "on_step_end":
            continue
        assert getattr(cb, hook, None) is None
