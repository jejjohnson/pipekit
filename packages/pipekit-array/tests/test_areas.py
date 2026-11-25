"""The areas of pipekit-array, their import rule, and what a saved operator records."""

from __future__ import annotations

import ast
import importlib
import pickle
from pathlib import Path

import pipekit_array
import pytest
from pipekit_ops import Operator, dumps, loads
from pipekit_ops.testing import public_operators


ROOT = Path(pipekit_array.__file__).parent
#: an area imports only from the ones before it; ``testing`` from all of them
AREAS = ("arrays", "named", "kernels", "operators", "bridges", "testing")
#: the areas that need an extra are imported by name, never by the package
BY_NAME = ("arrays", "testing")
#: the module a state was saved under before the areas
FORMER = {
    "ApplyAlong": "dim_operators",
    "ArgReduce": "dim_operators",
    "BatchAlong": "batching",
    "ConcatenateAlong": "combinators",
    "Estimator": "estimator",
    "ISel": "dim_operators",
    "KernelOp": "kernel",
    "MeanScalar": "reductions",
    "StackAlong": "combinators",
    "Window": "dim_operators",
}


def area_of(path: Path) -> str:
    """The area of a source file, or its name at the top of the package."""
    return path.relative_to(ROOT).parts[0].removesuffix(".py")


def imported_areas(path: Path) -> set[str]:
    """The areas that a file imports, inside its functions too."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        parts = node.module.split(".")
        if parts[0] != "pipekit_array":
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
    assert {path.name for path in ROOT.glob("*.py")} == {"__init__.py", "_recorded.py"}


@pytest.mark.parametrize("area", AREAS)
def test_an_area_imports_only_from_the_ones_before_it(area: str) -> None:
    allowed = set(AREAS[: AREAS.index(area) + 1])
    for path in (ROOT / area).rglob("*.py"):
        beyond = imported_areas(path) - allowed
        assert not beyond, f"{path.relative_to(ROOT)} imports {sorted(beyond)}"


def test_the_package_needs_no_extra() -> None:
    assert not imported_areas(ROOT / "__init__.py") & set(BY_NAME)


def test_named_needs_no_extra() -> None:
    # arrays needs NumPy and array-api-compat, and named carriers do not
    for area in ("named", "operators"):
        for path in (ROOT / area).rglob("*.py"):
            assert "arrays" not in imported_areas(path), path


@pytest.mark.parametrize("area", ["arrays", "named", "kernels", "operators", "testing"])
def test_the_names_of_an_area_are_unique_and_importable(area: str) -> None:
    if area in ("arrays", "kernels"):  # elementwise is a kernel on plain arrays
        pytest.importorskip("array_api_compat")
    module = importlib.import_module(f"pipekit_array.{area}")
    assert len(set(module.__all__)) == len(module.__all__)
    for name in module.__all__:
        assert getattr(module, name) is not None


def test_no_private_module_is_needed_for_a_public_name() -> None:
    top = set(pipekit_array.__all__) - {"BackendError", "Estimator"}
    homes = [
        importlib.import_module(f"pipekit_array.{area}")
        for area in ("named", "kernels", "operators")
    ]
    assert top <= {name for home in homes for name in home.__all__}


# --- what a saved operator records ---


def operators() -> list[type[Operator]]:
    return list(public_operators(pipekit_array).values())


def test_every_operator_is_recorded_under_the_package() -> None:
    for cls in operators():
        assert cls.__module__ == "pipekit_array", cls


@pytest.mark.parametrize(("name", "module"), FORMER.items())
def test_a_state_saved_before_the_areas_loads(name: str, module: str) -> None:
    from pipekit_ops._base.operator import resolve_loaded

    cls = getattr(pipekit_array, name)
    assert resolve_loaded(f"pipekit_array.{module}", name) is cls


def test_every_operator_had_a_module_before() -> None:
    for cls in operators():
        assert getattr(cls, "__former_modules__", ()), cls


def test_a_saved_pipeline_names_the_package() -> None:
    op = pipekit_array.ISel("time", 0)
    assert op.state["module"] == "pipekit_array"
    assert loads(dumps(op)).state == op.state
    assert pickle.loads(pickle.dumps(op)).state == op.state


def test_the_einx_operator_loads_from_its_former_module() -> None:
    from pipekit_array.bridges.einx import Einx
    from pipekit_ops._base.operator import resolve_loaded

    assert resolve_loaded("pipekit_array.einx", "Einx") is Einx
