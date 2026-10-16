"""Execute the code blocks of docs/tutorials/pipekit-under-jit.md in order."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest


pytest.importorskip("equinox")
pytest.importorskip("jax")

TUTORIAL = Path(__file__).resolve().parents[3] / "docs/tutorials/pipekit-under-jit.md"


def test_tutorial_code_blocks_run_top_to_bottom(monkeypatch):
    if not TUTORIAL.exists():
        pytest.skip("docs not available (installed package, not a checkout)")
    blocks = re.findall(r"```python\n(.*?)```", TUTORIAL.read_text(), flags=re.S)
    assert len(blocks) >= 5
    # Run in a real module: dataclasses (and so Equinox) resolve classes
    # through sys.modules[cls.__module__].
    module = types.ModuleType("pipekit_under_jit_tutorial")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"{TUTORIAL.name}[block {i}]", "exec"), module.__dict__)
        except Exception as err:  # pragma: no cover - surfaced with context
            raise AssertionError(f"tutorial block {i} failed:\n{block}") from err
