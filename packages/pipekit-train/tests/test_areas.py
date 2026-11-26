"""The areas of pipekit-train, their import rule, and what a saved operator records."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pipekit_train
import pytest
from pipekit_ops import Operator, dumps, loads
from pipekit_ops.core import resolve_loaded
from pipekit_ops.testing import public_operators


ROOT = Path(pipekit_train.__file__).parent
#: an area imports only from the ones before it
AREAS = ("data", "losses", "loop", "sweep", "backends")
#: the backends, each behind its extra, as (area, module the extra brings)
BACKENDS = (("equinox", "equinox"), ("numpyro", "numpyro"), ("blackjax", "blackjax"))
#: the module a state was saved under before the areas
FORMER = {
    "CachedDataset": "dataset",
    "Composite": "loss",
    "EarlyStopping": "callbacks",
    "HyperSweep": "sweep",
    "MSE": "loss",
    "ParameterGrid": "sweep",
    "TrainingLoop": "loop",
    "ValidationStep": "loop",
}


def imported_areas(path: Path) -> set[str]:
    """The areas that a file imports, inside its functions too."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        parts = node.module.split(".")
        if parts[0] != "pipekit_train":
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
    assert {path.name for path in ROOT.glob("*.py")} == {
        "__init__.py",
        "_signatures.py",
    }


@pytest.mark.parametrize("area", AREAS)
def test_an_area_imports_only_from_the_ones_before_it(area: str) -> None:
    allowed = set(AREAS[: AREAS.index(area) + 1])
    for path in (ROOT / area).rglob("*.py"):
        beyond = imported_areas(path) - allowed
        assert not beyond, f"{path.relative_to(ROOT)} imports {sorted(beyond)}"


def test_the_package_imports_no_backend() -> None:
    assert "backends" not in imported_areas(ROOT / "__init__.py")
    for area in AREAS[:-1]:
        for path in (ROOT / area).rglob("*.py"):
            assert "backends" not in imported_areas(path), path


@pytest.mark.parametrize("area", AREAS[:-1])
def test_the_names_of_an_area_are_unique_and_importable(area: str) -> None:
    module = importlib.import_module(f"pipekit_train.{area}")
    assert len(set(module.__all__)) == len(module.__all__)
    for name in module.__all__:
        assert getattr(module, name) is not None


def test_every_public_name_has_an_area() -> None:
    homes = {
        name
        for area in AREAS[:-1]
        for name in importlib.import_module(f"pipekit_train.{area}").__all__
    }
    assert set(pipekit_train.__all__) <= homes


@pytest.mark.parametrize(("area", "extra"), BACKENDS)
def test_a_backend_records_its_operators_under_its_area(area: str, extra: str) -> None:
    pytest.importorskip(extra)
    module = importlib.import_module(f"pipekit_train.backends.{area}")
    assert len(set(module.__all__)) == len(module.__all__)
    found = public_operators(module)
    assert found
    for cls in found.values():
        assert cls.__module__ == module.__name__, cls


def test_the_former_backend_modules_still_load() -> None:
    pytest.importorskip("equinox")
    from pipekit_train.backends.equinox import EquinoxModelOp

    assert resolve_loaded("pipekit_train.adapters.equinox", "EquinoxModelOp") is (
        EquinoxModelOp
    )


# --- what a saved operator records ---


def operators() -> list[type[Operator]]:
    return list(public_operators(pipekit_train).values())


def test_every_operator_is_recorded_under_the_package() -> None:
    assert operators()
    for cls in operators():
        assert cls.__module__ == "pipekit_train", cls


@pytest.mark.parametrize(("name", "module"), FORMER.items())
def test_a_state_saved_before_the_areas_loads(name: str, module: str) -> None:
    cls = getattr(pipekit_train, name)
    assert resolve_loaded(f"pipekit_train.{module}", name) is cls


def test_a_saved_loss_names_the_package() -> None:
    op = pipekit_train.MSE()
    assert op.state["module"] == "pipekit_train"
    assert loads(dumps(op)).state == op.state
    old = dumps(op).replace('"pipekit_train"', '"pipekit_train.loss"')
    assert loads(old).state == op.state
