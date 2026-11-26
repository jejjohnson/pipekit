"""The areas of pipekit-evaluate and their import rule."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pipekit_evaluate
import pytest


ROOT = Path(pipekit_evaluate.__file__).parent
#: an area imports only from the ones before it
AREAS = ("scoring", "reports", "benchmark")


def imported_areas(path: Path) -> set[str]:
    """The areas that a file imports, inside its functions too."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        parts = node.module.split(".")
        if parts[0] != "pipekit_evaluate":
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
    module = importlib.import_module(f"pipekit_evaluate.{area}")
    assert len(set(module.__all__)) == len(module.__all__)
    for name in module.__all__:
        assert getattr(module, name) is not None


def test_a_name_has_one_area() -> None:
    seen: dict[str, str] = {}
    for area in AREAS:
        for name in importlib.import_module(f"pipekit_evaluate.{area}").__all__:
            assert name not in seen, f"{name} is in {seen[name]} and {area}"
            seen[name] = area


def test_every_public_name_has_an_area() -> None:
    homes = {
        name
        for area in AREAS
        for name in importlib.import_module(f"pipekit_evaluate.{area}").__all__
    }
    assert set(pipekit_evaluate.__all__) - {"benchmark"} <= homes
