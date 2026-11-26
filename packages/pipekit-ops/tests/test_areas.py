"""The areas of pipekit-ops, their import rule, and what a saved operator records."""

from __future__ import annotations

import ast
import importlib
import pickle
from pathlib import Path

import pipekit_ops
import pytest
from pipekit_ops import Operator, dumps, loads
from pipekit_ops.core import resolve_loaded
from pipekit_ops.testing import public_operators


ROOT = Path(pipekit_ops.__file__).parent
#: an area imports only from the ones before it; ``testing`` from all of them
AREAS = ("carriers", "core", "saving", "operators", "execution", "observe", "testing")
#: the module a state was saved under before the areas
FORMER = {
    "Branch": "control",
    "Cache": "cache",
    "Const": "blocks",
    "Fanout": "combine",
    "Get": "dicts",
    "Graph": "_base.graph",
    "Map": "execution",
    "Quarantine": "qc",
    "Reduce": "seq",
    "Sequential": "_base.sequential",
    "Tap": "observe",
    "ThreadMap": "parallel",
    "Timeout": "flow",
}


def imported_areas(path: Path) -> set[str]:
    """The areas that a file imports, inside its functions too."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        parts = node.module.split(".")
        if parts[0] != "pipekit_ops":
            continue
        names = parts[1:2] or [alias.name for alias in node.names]
        found.update(name for name in names if name in AREAS)
    return found


def test_the_package_has_these_areas_and_no_other_module() -> None:
    found = {
        path.name
        for path in ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }
    assert found == set(AREAS)
    assert {path.name for path in ROOT.glob("*.py")} == {"__init__.py"}


@pytest.mark.parametrize("area", AREAS)
def test_an_area_imports_only_from_the_ones_before_it(area: str) -> None:
    allowed = set(AREAS[: AREAS.index(area) + 1])
    for path in (ROOT / area).rglob("*.py"):
        beyond = imported_areas(path) - allowed
        assert not beyond, f"{path.relative_to(ROOT)} imports {sorted(beyond)}"


@pytest.mark.parametrize("area", AREAS)
def test_the_names_of_an_area_are_unique_and_importable(area: str) -> None:
    module = importlib.import_module(f"pipekit_ops.{area}")
    assert len(set(module.__all__)) == len(module.__all__)
    for name in module.__all__:
        assert getattr(module, name) is not None


def test_every_public_name_has_an_area() -> None:
    homes = {
        name
        for area in AREAS[:-1]
        for name in importlib.import_module(f"pipekit_ops.{area}").__all__
    }
    assert set(pipekit_ops.__all__) <= homes


# --- what a saved operator records ---


def operators() -> list[type[Operator]]:
    return list(public_operators(pipekit_ops).values())


def test_every_operator_is_recorded_under_the_package() -> None:
    assert operators()
    for cls in operators():
        assert cls.__module__ == "pipekit_ops", cls


def test_every_public_exception_is_recorded_under_the_package() -> None:
    for name in pipekit_ops.__all__:
        value = getattr(pipekit_ops, name)
        if isinstance(value, type) and issubclass(value, BaseException):
            assert value.__module__ == "pipekit_ops", value


@pytest.mark.parametrize(("name", "module"), FORMER.items())
def test_a_state_saved_before_the_areas_loads(name: str, module: str) -> None:
    cls = getattr(pipekit_ops, name)
    assert resolve_loaded(f"pipekit_ops.{module}", name) is cls


def test_a_saved_pipeline_names_the_package() -> None:
    op = pipekit_ops.Sequential([pipekit_ops.Identity(), pipekit_ops.Const(1)])
    assert op.state["module"] == "pipekit_ops"
    assert loads(dumps(op)).state == op.state
    assert pickle.loads(pickle.dumps(op)).state == op.state


def test_an_exception_saved_before_the_areas_loads() -> None:
    op = pipekit_ops.Try(
        pipekit_ops.Identity(), pipekit_ops.Const(0), on=(pipekit_ops.QCError,)
    )
    old = dumps(op).replace('"pipekit_ops.QCError"', '"pipekit_ops.qc.QCError"')
    assert "pipekit_ops.qc.QCError" in old
    assert loads(old).state == op.state


def test_a_class_is_recorded_only_where_the_module_holds_it() -> None:
    class Stray(Operator):
        def _apply(self, x: object) -> object:
            return x

    with pytest.raises(ValueError, match="no Stray"):
        pipekit_ops.core.record_under("pipekit_ops", Stray)
