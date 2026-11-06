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
