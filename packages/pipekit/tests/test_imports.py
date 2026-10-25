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


# -- review fixes (Codex, #117) --------------------------------------------------------


def test_waits_for_another_threads_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module mid-import in another thread is seen whole by the callback."""
    import threading
    import time

    name = f"pk_slow_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text("import time\ntime.sleep(0.5)\nDONE = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        worker = threading.Thread(target=importlib.import_module, args=(name,))
        worker.start()
        while name not in sys.modules:
            time.sleep(0.01)
        seen: list[bool] = []
        on_import(name, lambda module: seen.append(getattr(module, "DONE", False)))
        worker.join()
        assert seen == [True]
    finally:
        sys.modules.pop(name, None)


def test_runs_for_a_namespace_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = f"pk_ns_{tmp_path.name.replace('-', '_')}"
    (tmp_path / name).mkdir()  # no __init__.py
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        seen: list[str] = []
        on_import(name, lambda module: seen.append(module.__name__))
        module = importlib.import_module(name)
        assert seen == [name]
        assert list(module.__path__) == [str(tmp_path / name)]
    finally:
        sys.modules.pop(name, None)


def test_warnings_as_errors_never_abort_the_import(
    fake_module: str, capsys: pytest.CaptureFixture[str]
) -> None:
    import warnings

    ran: list[str] = []

    def boom(module: object) -> None:
        raise ValueError("nope")

    on_import(fake_module, boom)
    on_import(fake_module, lambda module: ran.append("second"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        module = importlib.import_module(fake_module)
    assert module.VALUE == 42
    assert ran == ["second"]
    assert "nope" in capsys.readouterr().err
