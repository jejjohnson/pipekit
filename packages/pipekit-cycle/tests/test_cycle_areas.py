"""The areas of pipekit-cycle, their import rule, and what a saved operator records."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pipekit_cycle
import pytest
from pipekit_ops import dumps, loads
from pipekit_ops.core import resolve_loaded
from pipekit_ops.testing import public_operators


ROOT = Path(pipekit_cycle.__file__).parent
#: an area imports only from the ones before it
AREAS = ("protocols", "cycles", "components", "assimilation")
#: the module a state was saved under before the areas
FORMER = {
    "BFNCycle": "bfn",
    "CallableForward": "forward",
    "Cycle": "cycle",
    "DACycle": "da",
    "IdentityObs": "obs",
    "Recurrence": "cycle",
    "SmootherCycle": "da",
}


def imported_areas(path: Path) -> set[str]:
    """The areas that a file imports, inside its functions too."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        parts = node.module.split(".")
        if parts[0] != "pipekit_cycle":
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
    module = importlib.import_module(f"pipekit_cycle.{area}")
    assert len(set(module.__all__)) == len(module.__all__)
    for name in module.__all__:
        assert getattr(module, name) is not None


def test_every_public_name_has_one_area() -> None:
    seen: dict[str, str] = {}
    for area in AREAS:
        for name in importlib.import_module(f"pipekit_cycle.{area}").__all__:
            assert name not in seen, f"{name} is in {seen[name]} and {area}"
            seen[name] = area
    assert set(pipekit_cycle.__all__) == set(seen)


# --- what a saved operator records ---


def test_every_operator_is_recorded_under_the_package() -> None:
    found = public_operators(pipekit_cycle)
    assert found
    for cls in found.values():
        assert cls.__module__ == "pipekit_cycle", cls


@pytest.mark.parametrize(("name", "module"), FORMER.items())
def test_a_state_saved_before_the_areas_loads(name: str, module: str) -> None:
    cls = getattr(pipekit_cycle, name)
    assert resolve_loaded(f"pipekit_cycle.{module}", name) is cls


def test_the_adapter_a_cycle_saves_for_a_forward_model_loads_from_its_old_module() -> (
    None
):
    # Cycle wraps a plain ForwardModel in this private operator, and saves it
    cls = pipekit_cycle._ForwardModelAdapter
    assert cls.__module__ == "pipekit_cycle"
    assert resolve_loaded("pipekit_cycle.cycle", cls.__name__) is cls


def test_a_saved_operator_names_the_package() -> None:
    op = pipekit_cycle.IdentityObs()
    assert op.state["module"] == "pipekit_cycle"
    assert loads(dumps(op)).state == op.state
    old = dumps(op).replace('"pipekit_cycle"', '"pipekit_cycle.obs"')
    assert loads(old).state == op.state
