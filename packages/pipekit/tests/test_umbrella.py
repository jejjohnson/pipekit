"""The pipekit umbrella: no code, every workspace package, pointers for moved names."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pipekit
import pipekit_ops
import pytest


ROOT = Path(__file__).resolve().parents[3]


def test_the_umbrella_exports_only_its_version() -> None:
    assert pipekit.__all__ == ["__version__"]


def test_the_umbrella_installs_every_workspace_package() -> None:
    umbrella = tomllib.loads((ROOT / "packages/pipekit/pyproject.toml").read_text())
    wanted = {
        path.parent.name
        for path in (ROOT / "packages").glob("*/pyproject.toml")
        if path.parent.name != "pipekit"
    }
    declared = {
        req.split(">")[0].split("[")[0] for req in umbrella["project"]["dependencies"]
    }
    assert declared == wanted


@pytest.mark.parametrize("name", ["Operator", "Sequential", "Graph", "Retry", "dumps"])
def test_a_moved_name_points_at_pipekit_ops(name: str) -> None:
    assert name in pipekit_ops.__all__
    with pytest.raises(ImportError, match=rf"from pipekit_ops import {name}`"):
        getattr(pipekit, name)


def test_from_import_of_a_moved_name_raises_the_same_message() -> None:
    with pytest.raises(ImportError, match="moved to pipekit_ops"):
        from pipekit import Operator  # noqa: F401


def test_an_unknown_name_is_still_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'NotAThing'"):
        pipekit.NotAThing  # noqa: B018


#: the last release of each package that still imported `pipekit`
LAST_BEFORE_PIPEKIT_OPS = {
    "pipekit-array": (0, 0, 9),
    "pipekit-cycle": (0, 0, 4),
    "pipekit-experiment": (0, 0, 5),
    "pipekit-evaluate": (0, 1, 1),
    "pipekit-train": (0, 1, 5),
    "pipekit-jax": (0, 0, 6),
}


def _workspace_requirements() -> list[tuple[str, str]]:
    """``(owner, requirement)`` for every requirement on another workspace package."""
    found = []
    for path in sorted((ROOT / "packages").glob("*/pyproject.toml")):
        project = tomllib.loads(path.read_text())["project"]
        groups = [project.get("dependencies", [])]
        groups += project.get("optional-dependencies", {}).values()
        for requirements in groups:
            for req in requirements:
                if req.startswith("pipekit") and not req.startswith(project["name"]):
                    found.append((project["name"], req))
    return found


@pytest.mark.parametrize(("owner", "requirement"), _workspace_requirements())
def test_requirements_exclude_releases_that_import_pipekit(
    owner: str, requirement: str
) -> None:
    name = requirement.split(">=")[0].split("[")[0]
    assert ">=" in requirement, f"{owner}: {requirement} needs a lower bound"
    bound = tuple(int(part) for part in requirement.split(">=")[1].split("."))
    # pipekit-ops never shipped a release that imported `pipekit`
    last = LAST_BEFORE_PIPEKIT_OPS.get(name, (0, 1, 5))
    assert bound > last, f"{owner}: {requirement} still accepts {last}"
