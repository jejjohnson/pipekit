"""Tests for `pipekit.combine` — Group F."""

from __future__ import annotations

import pytest
from pipekit import Fanout, Identity, Operator


def test_fanout_returns_dict_of_outputs(Scale_cls, AddConst_cls):
    op = Fanout({"a": Scale_cls(2.0), "b": AddConst_cls(1.0)})
    assert op(3.0) == {"a": 6.0, "b": 4.0}


def test_fanout_each_branch_sees_same_input(Scale_cls):
    op = Fanout({"x": Identity(), "y": Scale_cls(0.0)})
    out = op(42)
    assert out["x"] == 42
    assert out["y"] == 0


def test_fanout_rejects_empty():
    with pytest.raises(ValueError):
        Fanout({})


def test_fanout_rejects_non_operator():
    with pytest.raises(TypeError):
        Fanout({"a": lambda x: x})  # type: ignore[dict-item]


def test_fanout_is_an_operator():
    assert isinstance(Fanout({"a": Identity()}), Operator)


def test_fanout_get_config_lists_branches():
    op = Fanout({"a": Identity()})
    cfg = op.get_config()
    assert "branches" in cfg
    assert "a" in cfg["branches"]


def test_fanout_repr():
    op = Fanout({"a": Identity(), "b": Identity()})
    assert "Fanout" in repr(op)
    assert "a" in repr(op)
