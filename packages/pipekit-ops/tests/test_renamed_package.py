"""States saved while the package was imported as `pipekit` still load."""

from __future__ import annotations

import json

import pytest
from pipekit_ops import (
    Const,
    Identity,
    Retry,
    Sequential,
    Try,
    allowed,
    dumps,
    loads,
    loads_sandboxed,
    register,
    unregister,
)
from pipekit_ops._base.operator import current_module


def _as_saved_before_0_2(text: str) -> str:
    """Rewrite every module path from ``pipekit_ops…`` back to ``pipekit…``."""

    def rewrite(node: object) -> object:
        if isinstance(node, dict):
            out = {key: rewrite(value) for key, value in node.items()}
            module = out.get("module")
            if isinstance(module, str) and module.startswith("pipekit_ops"):
                out["module"] = "pipekit" + module.removeprefix("pipekit_ops")
            exception = out.get("exception")
            if isinstance(exception, str) and exception.startswith("pipekit_ops."):
                out["exception"] = "pipekit" + exception.removeprefix("pipekit_ops")
            return out
        if isinstance(node, list):
            return [rewrite(value) for value in node]
        return node

    return json.dumps(rewrite(json.loads(text)))


@pytest.mark.parametrize(
    ("saved", "today"),
    [
        ("pipekit", "pipekit_ops"),
        ("pipekit.control", "pipekit_ops.control"),
        ("pipekit._base.sequential", "pipekit_ops._base.sequential"),
        ("pipekit_cycle.cycle", "pipekit_cycle.cycle"),
        ("pipekitish", "pipekitish"),
    ],
)
def test_current_module(saved: str, today: str) -> None:
    assert current_module(saved) == today


def _pipeline() -> Sequential:
    guarded = Try(primary=Const(3), fallback=Identity(), on=(KeyError,))
    retried = Retry(guarded, attempts=2, base_delay=0.0, on=(OSError,))
    return Sequential([retried, Identity()])


def test_a_pre_0_2_state_loads() -> None:
    old = _as_saved_before_0_2(dumps(_pipeline()))
    assert '"pipekit.control"' in old
    back = loads(old)
    assert back(0) == 3
    assert dumps(back) == dumps(_pipeline())


def test_a_pre_0_2_state_loads_sandboxed() -> None:
    classes = (Sequential, Retry, Try, Const, Identity)
    before = set(allowed())
    for cls in classes:
        register(cls)
    try:
        back = loads_sandboxed(_as_saved_before_0_2(dumps(_pipeline())))
    finally:
        for cls in classes:
            if f"{cls.__module__}.{cls.__name__}" not in before:
                unregister(cls)
    assert back(0) == 3
