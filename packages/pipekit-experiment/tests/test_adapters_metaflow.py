"""Tests for `pipekit_experiment.adapters.metaflow`.

We test the adapter without requiring metaflow itself — `run` and the
``decorate=False`` form of `as_step` don't touch metaflow.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pipekit import Const, Identity, Operator
from pipekit_experiment.adapters.metaflow import MetaflowStepAdapter


class Concat(Operator):
    """Two-input operator: concatenate two strings."""

    def _apply(self, a, b):
        return a + b


def test_run_reads_inputs_writes_output():
    ns = SimpleNamespace(left="hello, ", right="world")
    out = MetaflowStepAdapter.run(
        Concat(), ns, name="greeting", inputs=["left", "right"]
    )
    assert out == "hello, world"
    assert ns.greeting == "hello, world"


def test_as_step_undecorated_callable():
    step = MetaflowStepAdapter.as_step(
        Const(7), name="seven", inputs=[], decorate=False
    )
    ns = SimpleNamespace()
    step(ns)
    assert ns.seven == 7
    assert step.__name__ == "seven"


def test_as_step_writes_to_namespace_attr():
    step = MetaflowStepAdapter.as_step(
        Identity(), name="passthrough", inputs=["x"], decorate=False
    )
    ns = SimpleNamespace(x=42)
    step(ns)
    assert ns.passthrough == 42


def test_run_rejects_non_operator():
    with pytest.raises(TypeError):
        MetaflowStepAdapter.run(
            lambda x: x,  # type: ignore[arg-type]
            SimpleNamespace(x=1),
            name="out",
            inputs=["x"],
        )


def test_as_step_rejects_non_operator():
    with pytest.raises(TypeError):
        MetaflowStepAdapter.as_step(
            lambda x: x,  # type: ignore[arg-type]
            name="out",
            inputs=["x"],
            decorate=False,
        )


def test_as_step_decorated_requires_metaflow(monkeypatch):
    """decorate=True should raise a clean ImportError when metaflow absent."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "metaflow":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="metaflow"):
        MetaflowStepAdapter.as_step(Identity(), name="x", inputs=[], decorate=True)
