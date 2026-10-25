"""`pipekit.on_import`: run code when a module is imported, without importing it."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pipekit import on_import


@pytest.fixture
def fake_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A never-imported module on sys.path; its name is unique per test."""
    name = f"pk_fake_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text("VALUE = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    sys.modules.pop(name, None)


def test_runs_after_a_later_import(fake_module: str) -> None:
    seen: list[int] = []
    on_import(fake_module, lambda module: seen.append(module.VALUE))
    assert fake_module not in sys.modules  # registering does not import
    assert seen == []
    importlib.import_module(fake_module)
    assert seen == [42]
    importlib.reload(sys.modules[fake_module])
    assert seen == [42]  # once


def test_runs_now_when_already_imported(fake_module: str) -> None:
    importlib.import_module(fake_module)
    seen: list[int] = []
    on_import(fake_module, lambda module: seen.append(module.VALUE))
    assert seen == [42]


def test_a_failing_callback_warns_and_the_import_succeeds(fake_module: str) -> None:
    def boom(module: object) -> None:
        raise ValueError("nope")

    on_import(fake_module, boom)
    with pytest.warns(RuntimeWarning, match="nope"):
        module = importlib.import_module(fake_module)
    assert module.VALUE == 42


def test_the_wrapped_loader_keeps_the_real_loaders_api(fake_module: str) -> None:
    on_import(fake_module, lambda module: None)
    module = importlib.import_module(fake_module)
    assert module.__spec__.loader.get_filename(fake_module).endswith(".py")
